import sys
import re

import pyperclip

from globals import *
import conversation
import threading
import time




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
FOLDER = BOT_FOLDER

CONFIG_FILE = 'bots'
CONFIG = f'{BOT_FOLDER}/{CONFIG_FILE}'
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

SAVED_VARS = ['USER_NAME', 'FILENAME', 'SWITCH']
config = load_file(CONFIG)
BOTS = config['bots']
for k, v in config['variables'].items():
    # Import certain variables from bots file
    if k in SAVED_VARS:
        globals()[k] = v

files = [f.split('.')[0] for f in os.listdir(BOT_FOLDER)]
ALL_BOTS = [f for f in files if f != CONFIG_FILE]
ALL_BOTS.sort()
PRINT_START = 1
PARAGRAPHS = None
RESETS = 0

def save_config():
    vars = {k: globals()[k] for k in SAVED_VARS}
    data = {'variables': vars, 'bots': BOTS}
    save_file(data, CONFIG)

def create_filename():
    prefix = FILENAME+'_' if FILENAME else ''
    return f'{CHAT_FOLDER}/{prefix}{DATE}_{FILENUM}'

def filename_vars(filename):
    if '/' in filename:
        _, filename = filename.split('/')
    match filename.split('_'):
        case [f, d, n]:
            return f, d, int(n)
        case [f, n]:
            return f, int(n)
        case [f]:
            return f, DATE, 0

def print_filename(filename=None):
    if filename is None:
        filename = f'{CHAT_FOLDER}/{FILENAME}_{DATE}_{FILENUM}'

    folder, file = filename.split('/')
    match file.capitalize().split('_'):
        case [f, d, n]:
            s = f'[od.dim]{folder} /[/] [bold]{f}[/] [od.dim]-[/] [od.white]{d}[/] [od.dim]-[/] [bold]{n}'
        case [f, n]:
            s = f'[od.dim]{folder} /[/] [bold]{f}[/] [od.dim]-[/] [bold]{n}'
        case [f]:
            s = f'[od.dim]{folder} /[/] [bold]{f}'
    print_rule(s)

def filename_num(filename):
    return filename_vars(filename)[2] # Last element

def latest_file_today(string):
    all = os.listdir(CHAT_FOLDER)
    files = [f.split('.')[0] for f in all if f.startswith(f'{string}_{DATE}')]

    if files:
        return f'{CHAT_FOLDER}/{max(files, key=filename_num)}'
    else:
        return f'{BOT_FOLDER}/{string}'

def load_latest(string, reset=False):
    global FILENAME, DATE, FILENUM, PRINT_START
    file = latest_file_today(string)
    filename, DATE, FILENUM = filename_vars(file)
    if filename != FILENAME:
        FILENAME = filename
        save_config()

    if file.startswith(BOT_FOLDER) or reset:
        file = f'{BOT_FOLDER}/{filename}'
        chat.reset(file, ai=string.capitalize()+'Bot')
    else:
        chat.load(file)
    print_filename(file)

    if len(chat.messages) > 1:
        PRINT_START = len(chat.messages)-2
        chat.print_messages(chat.messages, PRINT_START, None)
    else:
        PRINT_START = 1

    if FILENUM == 0:
        FILENUM = 1
    elif reset:
        FILENUM += 1

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

def screen_reset():
    global RESETS
    RESETS += 1
    clear_screen()
    print_filename()
    chat.print_messages(chat.messages, PRINT_START, None)
    if PARAGRAPHS is not None:
        console.print(chat.ai_prefix())
        if PARAGRAPHS != '':
            print_markdown(PARAGRAPHS)
            console.print('')
    else:
        console.print('\r'+chat.user_prefix(), end='')

def watch_window_size():
    last = os.get_terminal_size().columns
    reset_flag = False
    while True:
        current = os.get_terminal_size().columns
        if current != last: # size changed, set flag to reset later
            reset_flag = True
        elif reset_flag: # reset once size stops changing (current == last)
            screen_reset()
            reset_flag = False
        last = current
        time.sleep(0.1)




##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##

if __name__ == '__main__':
    print_rule('[bold]BlueChat')
    # console.print('[bold]Chatbots:[/] '+', '.join([s.capitalize() for s in ALL_BOTS]) )

    if len(sys.argv) == 2: # If there's an argument...
        # Load an existing conversation
        arg = sys.argv[1]
        if arg.endswith('.json') or arg.endswith('.yaml'):
            arg, _ = arg.split('.')
        chat = conversation.Conversation(filename=arg, user=USER_NAME, ai=AI_NAME)

    elif AUTOLOAD:
        chat = conversation.Conversation(system='', user=USER_NAME, ai=AI_NAME)
        load_latest(FILENAME)

    else: # Setup a new conversation
        chat = conversation.Conversation(user=USER_NAME, ai=AI_NAME)

    window_size_thread = threading.Thread(target=watch_window_size, daemon=True)
    window_size_thread.start()

    last_input_reset = False
    while True:
        # Get a new input
        user_input = chat.input(prefix=(not last_input_reset))
        last_input_reset = False
        if not user_input:
            screen_reset()
            last_input_reset = True
            continue
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
                _, DATE, FILENUM = filename_vars(latest_file_today(FILENAME))
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

            elif command in ['variables', 'vars', 'v']: # Print variables
                console.print({k: v for k, v in globals().items() if k.isupper()})
                chat.print_vars()

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

            elif command in ALL_BOTS: # New bot chat by name (!help, !code, !math, etc...)
                FIRST_MESSAGE = False # Just loaded, don't switch context right away
                load_latest(command, reset=True)

        if user_input and not user_input.startswith('!'): # !p and !a can change user_input
            if SWITCH and FIRST_MESSAGE:
                with status_spinner('Checking context...'):
                    switch = switch_context(user_input)
                    if switch and switch != FILENAME:
                        load_latest(switch, reset=True)

            FIRST_MESSAGE = False

            PARAGRAPHS = ''
            for p in chat.get_dialogue(user_input):
                PARAGRAPHS += p
            PARAGRAPHS = None

            if AUTOSAVE:
                chat.save(create_filename())
