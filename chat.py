import sys
import re

import pyperclip

from globals import *
import conversation




 ######   #######  ##    ##  ######  ########    ###    ##    ## ########  ######
##    ## ##     ## ###   ## ##    ##    ##      ## ##   ###   ##    ##    ##    ##
##       ##     ## ####  ## ##          ##     ##   ##  ####  ##    ##    ##
##       ##     ## ## ## ##  ######     ##    ##     ## ## ## ##    ##     ######
##       ##     ## ##  ####       ##    ##    ######### ##  ####    ##          ##
##    ## ##     ## ##   ### ##    ##    ##    ##     ## ##   ###    ##    ##    ##
 ######   #######  ##    ##  ######     ##    ##     ## ##    ##    ##     ######

BOT_FOLDER = 'bots'
CHAT_FOLDER = 'chats'
FILENAME = 'help'

CONFIG = f'{BOT_FOLDER}/bots'
AUTOSAVE = True
AUTOLOAD = True
FILENUM = 1

TODAY = datetime.now().strftime('%Y-%m-%d')
DATE = TODAY

AI_NAME = 'DefaultAI'
try:
    USER_NAME = os.getlogin()
except Exception:
    USER_NAME = 'DefaultUser'

SWITCH = True
FIRST_MESSAGE = True

DEFAULT = 'help'
SAVED_VARS = ['USER_NAME', 'DEFAULT', 'FILENAME', 'SWITCH']
config = load_file(CONFIG, output=False)
BOTS = config['bots']
for k, v in config['variables'].items():
    # Import certain variables from bots file
    if k in SAVED_VARS:
        globals()[k] = v


def save_config():
    vars = {k: globals()[k] for k in SAVED_VARS}
    data = {'variables': vars, 'bots': BOTS}
    save_file(data, CONFIG, output=False)

def create_filename():
    prefix = FILENAME+'_' if FILENAME else ''
    return f'{CHAT_FOLDER}/{prefix}{DATE}_{FILENUM}'

def filename_vars(filename):
    if '/' in filename:
        _, filename = filename.split('/')
    pattern = r'([a-zA-Z]*)_?(\d\d\d\d-\d\d-\d\d)?_?(\d*)'
    a = re.search(pattern, filename)
    d = a.group(2) if a.group(2) else DATE
    n = int(a.group(3)) if a.group(3) else 0
    return a.group(1), d, n

def filename_num(filename):
    return filename_vars(filename)[2] # Last element

def latest_file_today(string):
    all = os.listdir(CHAT_FOLDER)
    files = [f.split('.')[0] for f in all if f.startswith(f'{string}_{DATE}')]

    if files:
        return f'{CHAT_FOLDER}/{max(files, key=filename_num)}'
    else:
        return f'{BOT_FOLDER}/{string}'

def switch_file(assistant):
    global FILENAME, DATE, FILENUM
    file = latest_file_today(assistant)
    chat.reset(f'{BOT_FOLDER}/{assistant}')
    chat.ai_name = f"{assistant.capitalize()}Bot"
    FILENAME, DATE, FILENUM = filename_vars(file)
    FILENUM += 1
    save_config() # Save new default filename

def load_latest(string):
    global FILENAME, DATE, FILENUM
    file = latest_file_today(string)
    if file.startswith(BOT_FOLDER):
        chat.reset(file)
        chat.ai_name = f"{string.capitalize()}Bot"
    else:
        chat.load(file)
    FILENAME, DATE, FILENUM = filename_vars(file)
    if FILENUM == 0:
        FILENUM = 1

def switch_context(user_input):
    bot_string = '\n'.join( [f'{k}: {v}' for k, v in BOTS.items()] )
    prompt = f"You are a routing system for chatbots. You connect the users request to the chatbot which is best for it. Give the name of the best chatbot for this interaction and say nothing else. If none of the chatbots fit or you are unsure, say 'idk'. Choose from these chatbots:\n{bot_string}"

    assistant = conversation.get_complete(prompt, user_input, max_tokens=16, print_result=False)
    if DEBUG: console.log(f'{assistant=}')
    assistant = assistant.strip()

    if assistant in BOTS: # exact match
        if DEBUG: console.log(f'Assistant {create_filename()} (high certainty, string is key)')
        return assistant
    
    for bot in BOTS: # starts with bot, 'bot', or "bot"
        if assistant.startswith(bot) or assistant.startswith(f"'{bot}'") or assistant.startswith(f"\"{bot}\""):
            if DEBUG: console.log(f'Assistant {create_filename()} (moderate certainty, string startswith bot)')
            return bot
        
    for bot in BOTS: # bot is anywhere in the string
        if bot in assistant:
            if DEBUG: console.log(f'Assistant {create_filename()} (low certainty, bot in string)')
            return bot
            
    if assistant != 'idk': # unknown response
        if DEBUG: console.print(f'Context switch unknown response: {assistant}')

    return False




##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##

if __name__ == '__main__':
    print_markdown(f'## BlueChat')

    if len(sys.argv) == 2: # If there's an argument...
        # Load an existing conversation
        arg = sys.argv[1]
        if arg.endswith('.json') or arg.endswith('.yaml'):
            arg, _ = arg.split('.')
        chat = conversation.Conversation(filename=arg, user=USER_NAME, ai=AI_NAME)

    elif AUTOLOAD:
        latest = latest_file_today(FILENAME)
        chat = conversation.Conversation(filename=latest, user=USER_NAME, ai=AI_NAME)
        FILENAME, DATE, FILENUM = filename_vars(latest)
        chat.ai_name = FILENAME.capitalize() + 'Bot'
        if FILENUM == 0:
            FILENUM = 1

    else: # Setup a new conversation
        chat = conversation.Conversation(user=USER_NAME, ai=AI_NAME)

    while True:
        # Get a new input
        user_input = chat.input()
        if not user_input:
            clear_screen()
            chat.print_messages(chat.messages)

        elif user_input.startswith('!'):
            words = user_input.split(' ')
            command = words[0][1:]
            if len(words) > 1:
                parm = words[1]
            else:
                parm = ''

            # Process commands
            if command in ['exit', 'e', 'x']: # Exit program
                console.print('Goodbye')
                break
            
            elif command in ['restart', 'r']: # Restart chat
                console.print('Restarting conversation')
                chat.reset(f'{BOT_FOLDER}/{FILENAME}')
                FILENAME, DATE, FILENUM = filename_vars(latest_file_today(FILENAME))
                FILENUM += 1

            elif command in ['redo', 'undo', 'u']: # Undo messages
                if len(chat.messages) > 1:
                    console.print('Removing last 2 messages')
                    chat.messages = chat.messages[:-2]
                    chat.history = chat.history[:-2]
                else:
                    console.print('No messages to remove')

            elif command in ['summary', 'sum']: # Summarize chat
                if parm:
                    console.print(f'Summary: {chat.summarize_messages(int(parm))}')
                else:
                    console.print(f'Summary: {chat.summarize_messages()}')

            elif command in ['print', 'messages']: # Print chat memory
                chat.print_messages(chat.messages)

            elif command in ['history', 'h']: # Print chat history
                chat.print_messages(chat.history)

            elif command in ['save', 's']: # Save chat to file
                if parm:
                    chat.save(parm)
                    console.print(f"Data saved to {parm}")
                else:
                    chat.save(create_filename())

            elif command in ['load', 'l']: # Load chat from file
                FIRST_MESSAGE = False # If we just loaded, we don't want to switch context right away
                if parm:
                    load_latest(parm)
                else:
                    console.print('Error: No filename specified')

            elif command in ['model', 'm']: # Set GPT model
                if parm in ['3', '3.5', 'gpt-3', 'gpt-3.5-turbo']:
                    chat.model = 'gpt-3.5-turbo'
                elif parm in ['4', 'gpt-4']:
                    chat.model = 'gpt-4'
                console.print(f"Model={chat.model}")

            elif command in ['translate', 't']: # Translate mode
                if parm:
                    chat.translate = True
                    chat.ai_lang = parm
                    if len(words) == 3:
                        chat.user_lang = words[2]
                    else:
                        chat.user_lang = 'English'
                else:
                    chat.translate = not chat.translate

                console.print(f'Translate={chat.translate}, AI={chat.ai_lang}, User={chat.user_lang}')

            elif command in ['debug', 'd']: # Toggle debug mode
                DEBUG = not DEBUG
                console.print(f'Debug={DEBUG}')

            elif command in ['auto', 'a']: # Auto-switch context mode
                if parm: # turn auto-switch on and take user input
                    user_input = ' '.join(words[1:]) # everything after !a
                    SWITCH = True
                    FIRST_MESSAGE = True
                else: # toggle auto-switching
                    if FIRST_MESSAGE and SWITCH:
                        SWITCH = False
                        FIRST_MESSAGE = False
                    else:
                        SWITCH = True
                        FIRST_MESSAGE = True
                    console.print(f'Auto-switch={SWITCH}')

            elif command in ['copy', 'c']: # Copy chat messages to clipboard
                if parm:
                    if parm == 'all':
                        n = len(chat.messages)
                    else:
                        n = int(parm)
                else:
                    n = 2
                pyperclip.copy(chat.messages_string(chat.messages[-n:], divider='\n\n'))
                console.print(f'Copied {n} messages to clipboard\n')

            elif command in ['paste', 'p', 'v']: # Paste clipboard to a message
                user_input = pyperclip.paste()
                console.print(user_input)

            elif command in BOTS: # New bot chat by name (!help, !code, !math, etc...)
                switch_file(command)

        if user_input and not user_input.startswith('!'): # !p and !a can change user_input
            if SWITCH and FIRST_MESSAGE:
                with status_spinner('Checking context...'):
                    switch = switch_context(user_input)
                    if switch and switch != FILENAME:
                        switch_file(switch)

            FIRST_MESSAGE = False

            answer = chat.get_dialogue(user_input)

            if AUTOSAVE:
                chat.save(create_filename(), output=False)
