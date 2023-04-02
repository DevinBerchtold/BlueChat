import os
import sys
import re
from datetime import datetime

import conversation




 ######   #######  ##    ##  ######  ########    ###    ##    ## ########  ######
##    ## ##     ## ###   ## ##    ##    ##      ## ##   ###   ##    ##    ##    ##
##       ##     ## ####  ## ##          ##     ##   ##  ####  ##    ##    ##
##       ##     ## ## ## ##  ######     ##    ##     ## ## ## ##    ##     ######
##       ##     ## ##  ####       ##    ##    ######### ##  ####    ##          ##
##    ## ##     ## ##   ### ##    ##    ##    ##     ## ##   ###    ##    ##    ##
 ######   #######  ##    ##  ######     ##    ##     ## ##    ##    ##     ######

AUTOSAVE = True
AUTOLOAD = True
# DEFAULT_FILENAME = 'help'
FILENAME = 'help'
FILENUM = 1

TODAY = datetime.now().strftime('%Y-%m-%d')
DATE = TODAY

# USER_NAME = 'DefaultUser'
AI_NAME = 'DefaultAI'
try:
    USER_NAME = os.getlogin()
except Exception:
    USER_NAME = 'DefaultUser'

SWITCH = True
FIRST_MESSAGE = True

settings_data = conversation.yaml.safe_load(open(f'{conversation.FOLDER}/bots.yaml', 'r', encoding='utf8'))
bots = settings_data['bots']
for k, v in settings_data['variables'].items():
    # Import certain variables from bots file
    if k in ['USER_NAME', 'AI_NAME', 'FILENAME', 'SWITCH']:
        globals()[k] = v
bot_string = '\n'.join( [f'{k}: {v}' for k, v in bots.items()] )

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
    if conversation.DEBUG: print(f'<Parse Name={FILENAME}, Date={DATE}, Number={FILENUM}')

def switch_context(user_input):
    if conversation.DEBUG: print('Checking context...')
    prompt = f"You are the first interface to the user for BlueChat, a series of helpful chatbots. Your job is to route the user's request to the bot which can best handle the request. The following bots are available:\n{bot_string}\nRespond with the name of the best bot for this interaction. If none of the bots are suited for the request or you are unsure, say 'idk'. Do not talk to the user. Only say a bot name or 'idk'"

    assistant = conversation.get_complete(prompt, user_input)
    if conversation.DEBUG: print(f'<{assistant=}>')
    if assistant in bots.keys():
        global FILENAME, FILENUM
        FILENAME=assistant

        file = latest_file_today(assistant)
        chat.load(assistant)
        parse_filename(file)
        FILENUM += 1

        if conversation.DEBUG: print(f'Switching to {create_filename()}...')

        return assistant
    else:
        return False

def latest_file_today(string):
    all = os.listdir(conversation.FOLDER)
    files = [f.split('.')[0] for f in all if f.startswith(f'{string}_{DATE}')]

    if files:
        return sorted(files)[-1]
    else:
        return string

def load_latest(string, output=True):
    global FILENUM
    file = latest_file_today(string)
    chat.load(file, output=output)
    parse_filename(file)
    if FILENUM == 0:
        FILENUM = 1




##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##

if len(sys.argv) == 2: # If there's an argument...
    # Load an existing conversation
    arg = sys.argv[1]
    if arg.endswith('.json') or arg.endswith('.yaml'):
        arg = arg.split('.')[0]
    chat = conversation.Conversation(filename=arg, user=USER_NAME, ai=AI_NAME)

elif AUTOLOAD:
    # FILENAME = ''
    latest = latest_file_today(FILENAME)
    chat = conversation.Conversation(filename=latest, user=USER_NAME, ai=AI_NAME)
    parse_filename(latest)
    if FILENUM == 0:
        FILENUM = 1
    # latest = latest_file_today(DEFAULT_FILENAME)
    # if latest:
    #     chat = conversation.Conversation(filename=latest, user=USER_NAME, ai=AI_NAME)
    #     parse_filename(latest)
    # else:
    #     chat = conversation.Conversation(filename=DEFAULT_FILENAME, user=USER_NAME, ai=AI_NAME)
    #     FILENAME = DEFAULT_FILENAME
    #     DATE = TODAY
    #     FILENUM = 1

else: # Setup a new conversation
    chat = conversation.Conversation(user=USER_NAME, ai=AI_NAME)

# try:
while True:
    # Get a new input
    user_input = chat.input()

    if user_input.startswith('!'):
        input_split = user_input.split(' ')
        command = input_split[0]
        if len(input_split) > 1:
            parm = input_split[1]
        else:
            parm = ''

        # Process commands
        if command == '!exit' or command == '!x':
            print('Goodbye')
            break

        elif command == '!restart' or command == '!r':
            print('Restarting conversation')
            chat.load(FILENAME)
            parse_filename(latest_file_today(FILENAME))
            FILENUM += 1

        elif command == '!redo' or command == '!undo' or command == '!u':
            if len(chat.messages) > 1:
                print('Removing last 2 messages')
                chat.messages = chat.messages[:-2]
                chat.history = chat.history[:-2]
            else:
                print('No messages to remove')

        elif command == '!summary' or command == '!sum':
            if parm:
                print(f'<Summary: {chat.summarize(int(parm))}>')
            else:
                print(f'<Summary: {chat.summarize()}>')

        elif command == '!print' or command == '!messages' or command == '!p':
            for m in chat.messages:
                print(m['role']+': '+m['content'])

        elif command == '!history' or command == '!h':
            for m in chat.history:
                print(m['role']+': '+m['content'])

        elif command == '!save' or command == '!s':
            if parm:
                chat.save(parm)
                print(f"Data saved to {parm}")
            else:
                chat.save(create_filename())

        elif command == '!load' or command == '!l':
            FIRST_MESSAGE = False # If we just loaded, we don't want to switch context right away
            if parm:
                load_latest(parm)
            else:
                print('Error: No filename specified')

        elif command == '!model' or command == '!m':
            if parm:
                conversation.MODEL = parm
            print(f"<Model: {conversation.MODEL}>")

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

            print(f'<Translate={chat.translate}, AI={chat.ai_lang}, User={chat.user_lang}>')

        elif command == '!debug' or command == '!d':
            conversation.DEBUG = not conversation.DEBUG
            print(f'<Debug={conversation.DEBUG}>')

        elif command == '!switch' and command == '!c':
            SWITCH = not SWITCH
            print(f'<Switch={SWITCH}>')
        
    else: # Doesn't start with '!'
        if SWITCH: # Switch context if necessary
            if FIRST_MESSAGE:
                switch_context(user_input)
        FIRST_MESSAGE = False

        answer = chat.get_dialogue(user_input)

        if AUTOSAVE:
            chat.save(create_filename(), output=False)

# except Exception as e: # Unexpected error. Dump all data
#     print(e)
#     time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#     save(f"crash_{time_string}")
