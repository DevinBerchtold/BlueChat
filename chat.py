import sys
import re
from datetime import datetime

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

CONFIG = 'bots'
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
    return f'{prefix}{DATE}_{FILENUM}'

def parse_filename(filename):
    global FILENAME, DATE, FILENUM

    pattern = r'([a-zA-Z]*)_?(\d\d\d\d-\d\d-\d\d)?_?(\d*)'
    a = re.search(pattern, filename)
    if a.group(3) == '':
        n = 0
    else:
        n = int(a.group(3))

    FILENAME, DATE, FILENUM = a.group(1), a.group(2), n
    if DATE == None: DATE = TODAY
    if DEBUG: console.log(f'<Parse Name={FILENAME}, Date={DATE}, Number={FILENUM}')

def latest_file_today(string):
    all = os.listdir(FOLDER)
    files = [f.split('.')[0] for f in all if f.startswith(f'{string}_{DATE}')]

    if files:
        return sorted(files)[-1]
    else:
        return string

def load_latest(string):
    global FILENUM
    file = latest_file_today(string)
    if file == string:
        chat.reset(file)
        chat.ai_name = FILENAME.capitalize() + 'Bot'
    else:
        chat.load(file)
    parse_filename(file)
    if FILENUM == 0:
        FILENUM = 1

def switch_context(user_input):
    if DEBUG: console.log('Checking context')
    
    bot_string = '\n'.join( [f'{k}: {v}' for k, v in BOTS.items()] )
    prompt = f"You are a routing system for chatbots. You connect the users request to the chatbot which is best for it. Give the name of the best chatbot for this interaction and say nothing else. If none of the chatbots fit or you are unsure, say 'idk'. Choose from these chatbots:\n{bot_string}"

    assistant = conversation.get_complete(prompt, user_input, max_tokens=16, print_result=False)
    if DEBUG: console.log(f'{assistant=}')
    assistant = assistant.strip()
    global FILENAME, FILENUM
    if assistant in BOTS.keys():
        if assistant != FILENAME:
            file = latest_file_today(assistant)
            chat.reset(assistant)
            chat.ai_name = assistant.capitalize() + 'Bot'
            parse_filename(file)
            FILENUM += 1
            save_config() # Save new default filename

            if DEBUG: console.log(f'Switching to {create_filename()}')
            return assistant
    else:
        if assistant != 'idk':
            console.print(f'Context switch unknown response: {assistant}')
    return False




##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##

print_markdown(f'## BlueChat')

if len(sys.argv) == 2: # If there's an argument...
    # Load an existing conversation
    arg = sys.argv[1]
    if arg.endswith('.json') or arg.endswith('.yaml'):
        arg = arg.split('.')[0]
    chat = conversation.Conversation(filename=arg, user=USER_NAME, ai=AI_NAME)

elif AUTOLOAD:
    latest = latest_file_today(FILENAME)
    chat = conversation.Conversation(filename=latest, user=USER_NAME, ai=AI_NAME)
    parse_filename(latest)
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
        input_split = user_input.split(' ')
        command = input_split[0]
        if len(input_split) > 1:
            parm = input_split[1]
        else:
            parm = ''

        # Process commands
        if command == '!exit' or command == '!e' or command == '!x':
            console.print('Goodbye')
            break

        elif command == '!restart' or command == '!r':
            console.print('Restarting conversation')
            chat.reset(filename=FILENAME)
            parse_filename(latest_file_today(FILENAME))
            FILENUM += 1

        elif command == '!redo' or command == '!undo' or command == '!u':
            if len(chat.messages) > 1:
                console.print('Removing last 2 messages')
                chat.messages = chat.messages[:-2]
                chat.history = chat.history[:-2]
            else:
                console.print('No messages to remove')

        elif command == '!summary' or command == '!sum':
            if parm:
                console.print(f'Summary: {chat.summarize_messages(int(parm))}')
            else:
                console.print(f'Summary: {chat.summarize_messages()}')

        elif command == '!print' or command == '!messages':
            chat.print_messages(chat.messages)

        elif command == '!history' or command == '!h':
            chat.print_messages(chat.history)

        elif command == '!save' or command == '!s':
            if parm:
                chat.save(parm)
                console.print(f"Data saved to {parm}")
            else:
                chat.save(create_filename())

        elif command == '!load' or command == '!l':
            FIRST_MESSAGE = False # If we just loaded, we don't want to switch context right away
            if parm:
                load_latest(parm)
            else:
                console.print('Error: No filename specified')

        elif command == '!model' or command == '!m':
            if parm in ['3', '3.5', 'gpt-3', 'gpt-3.5-turbo']:
                chat.model = 'gpt-3.5-turbo'
            elif parm in ['4', 'gpt-4']:
                chat.model = 'gpt-4'
            console.print(f"Model={chat.model}")

        elif command == '!translate' or command == '!t':
            if user_input == '!translate' or user_input == '!t':
                chat.translate = not chat.translate
            else:
                if len(input_split) == 2:
                    chat.translate = True
                    chat.ai_lang = parm
                    chat.user_lang = 'English'
                elif len(input_split) == 3:
                    chat.translate = True
                    chat.ai_lang = parm
                    chat.user_lang = input_split[2]

            console.print(f'Translate={chat.translate}, AI={chat.ai_lang}, User={chat.user_lang}')

        elif command == '!debug' or command == '!d':
            DEBUG = not DEBUG
            console.print(f'Debug={DEBUG}')

        elif command == '!auto' or command == '!a':
            if FIRST_MESSAGE and SWITCH:
                SWITCH = False
                FIRST_MESSAGE = False
            else:
                SWITCH = True
                FIRST_MESSAGE = True
            console.print(f'Auto-switch={SWITCH}')

        elif command == '!copy' or command == '!c':
            if parm:
                if parm == 'all':
                    n = len(chat.messages)
                else:
                    n = int(parm)
            else:
                n = 2
            pyperclip.copy(chat.messages_string(chat.messages[-n:], divider='\n\n'))
            console.print(f'Copied {n} messages to clipboard\n')

        elif command == '!paste' or command == '!p' or command == '!v':
            user_input = pyperclip.paste()
            console.print(user_input)
        
    if user_input and not user_input.startswith('!'): # Not else. !p can change user_input
        if SWITCH and FIRST_MESSAGE:
            switch_context(user_input)
        FIRST_MESSAGE = False

        answer = chat.get_dialogue(user_input)

        if AUTOSAVE:
            chat.save(create_filename(), output=False)
