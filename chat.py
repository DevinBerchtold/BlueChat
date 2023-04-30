import os
import sys
import threading
import time
import re

import pyperclip

import files
import console
import conversation




 ######   #######  ##    ##  ######  ########    ###    ##    ## ########  ######
##    ## ##     ## ###   ## ##    ##    ##      ## ##   ###   ##    ##    ##    ##
##       ##     ## ####  ## ##          ##     ##   ##  ####  ##    ##    ##
##       ##     ## ## ## ##  ######     ##    ##     ## ## ## ##    ##     ######
##       ##     ## ##  ####       ##    ##    ######### ##  ####    ##          ##
##    ## ##     ## ##   ### ##    ##    ##    ##     ## ##   ###    ##    ##    ##
 ######   #######  ##    ##  ######     ##    ##     ## ##    ##    ##     ######

DEBUG = False

BOT_FOLDER = 'bots'
CHAT_FOLDER = 'chats'
FILENAME = 'help'
FOLDER = BOT_FOLDER

CONFIG_FILE = 'bots'
CONFIG = f'{BOT_FOLDER}/{CONFIG_FILE}'
AUTOSAVE = True
AUTOLOAD = True
FILENUM = 1
DATE = files.TODAY

AI_NAME = 'DefaultAI'
try:
    USER_NAME = os.getlogin()
except Exception:
    USER_NAME = 'DefaultUser'

SWITCH = True
FIRST_MESSAGE = True

SAVED_VARS = ['USER_NAME', 'FILENAME', 'SWITCH']
config = files.load_file(CONFIG)
BOTS = config['bots']
for k, v in config['variables'].items():
    # Import certain variables from bots file
    if k in SAVED_VARS:
        globals()[k] = v

ALL_BOTS = sorted([
    f.split('.')[0]
    for f in os.listdir(BOT_FOLDER)
    if f.split('.')[0] != CONFIG_FILE
])
PRINT_START = 1




######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######

def save_config():
    vars = {k: globals()[k] for k in SAVED_VARS}
    data = {'variables': vars, 'bots': BOTS}
    files.save_file(data, CONFIG)

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

def filename_num(filename):
    return filename_vars(filename)[2] # Last element

def latest_file_today(string):
    all = os.listdir(CHAT_FOLDER)
    chat_files = [f.split('.')[0] for f in all if f.startswith(f'{string}_{DATE}')]

    if chat_files:
        return f'{CHAT_FOLDER}/{max(chat_files, key=filename_num)}'
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
    console.print_filename(file)

    if len(chat.messages) > 1:
        PRINT_START = len(chat.history)-2
        chat.print_messages(PRINT_START, None)
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

def screen_reset(chat):
    return console.reset(chat,
        f'{CHAT_FOLDER}/{FILENAME}_{DATE}_{FILENUM}',PRINT_START)

def watch_window_size():
    last = os.get_terminal_size().columns
    reset_flag = False
    while True:
        current = os.get_terminal_size().columns
        if current != last: # size changed, set flag to reset later
            reset_flag = True
        elif reset_flag: # reset once size stops changing (current == last)
            screen_reset(chat)
            reset_flag = False
        last = current
        time.sleep(0.1)




 ######   #######  ##     ## ##     ##    ###    ##    ## ########   ######
##    ## ##     ## ###   ### ###   ###   ## ##   ###   ## ##     ## ##    ##
##       ##     ## #### #### #### ####  ##   ##  ####  ## ##     ## ##
##       ##     ## ## ### ## ## ### ## ##     ## ## ## ## ##     ##  ######
##       ##     ## ##     ## ##     ## ######### ##  #### ##     ##       ##
##    ## ##     ## ##     ## ##     ## ##     ## ##   ### ##     ## ##    ##
 ######   #######  ##     ## ##     ## ##     ## ##    ## ########   ######

def restart_command(chat, parm, words):
    """Restart the chat by clearing memory and creating a new save file."""
    global DATE, FILENUM
    console.print('Restarting conversation')
    chat.reset(f'{BOT_FOLDER}/{FILENAME}')
    _, DATE, FILENUM = filename_vars(latest_file_today(FILENAME))
    FILENUM += 1

def undo_command(chat, parm, words):
    """Undo the last messages in the chat. Defaults to last 2 messages if parameter is not specified."""
    if parm:
        n = int(parm)
    else:
        n = 2
    if len(chat.messages) >= n:
        chat.messages = chat.messages[:-n]
        chat.history = chat.history[:-n]
        console.print(f'Removing last {n} messages')
    else:
        console.print('No messages to remove')

def summary_command(chat, parm, words):
    """Summarize the last tokens of the conversation. Summarizes the entire conversation if number of tokens is not specified."""
    if parm:
        s = chat.summarize_messages(int(parm))
    else:
        s = chat.summarize_messages()
    console.print(f'Summary: {s}')

def history_command(chat, parm, words):
    """Print the conversation system messages and full chat history."""
    console.print_filename(f'{CHAT_FOLDER}/{FILENAME}_{DATE}_{FILENUM}')
    chat.print_systems()
    chat.print_messages()

def save_command(chat, parm, words):
    """Save the chat to the given file. Automatically adds file extension and names the file if a filename is not specified."""
    if parm:
        chat.save(parm)
        console.print(f"Data saved to {parm}")
    else:
        chat.save(create_filename())

def load_command(chat, parm, words):
    """Load the chat from the given file. Loads the latest file if a bot name is specified, or the exact file if a filename is specified."""
    global FIRST_MESSAGE
    FIRST_MESSAGE = False # Just loaded, don't switch context right away
    if parm:
        if '_' in parm:
            if '/' in parm:
                chat.load(parm)
            else:
                chat.load(f'{CHAT_FOLDER}/{parm}')
        else:
            load_latest(parm)
    else:
        console.print('Error: No filename specified')

def model_command(chat, parm, words):
    """Set the LLM model (e.g. gpt-3.5-turbo, gpt-4) to be used in the chat. If a model is not specified, prints the current model. Accepts abbreviations like 'gpt-3.5' or just '3'"""
    if parm in ['3', '3.5', 'gpt-3', 'gpt-3.5-turbo']:
        chat.model = 'gpt-3.5-turbo'
    elif parm in ['4', 'gpt-4']:
        chat.model = 'gpt-4'
    console.print(f"Model={chat.model}")

def translate_command(chat, parm, words):
    """Toggle translate mode if no parameters are specified. If specified, the first parameter is the AI language and the second parameter is the user's language."""
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

def debug_command(chat, parm, words):
    """Toggle debug mode for all modules."""
    DEBUG = not DEBUG
    files.DEBUG = DEBUG
    console.DEBUG = DEBUG
    conversation.DEBUG = DEBUG
    console.print(f'Debug={DEBUG}')

def auto_command(chat, parm, words):
    """Toggle auto-switch context mode. Text after the command is sent as a message."""
    global FIRST_MESSAGE, SWITCH
    if parm: # turn auto-switch on and take user input
        user_input = ' '.join(words[1:]) # everything after !a
        SWITCH = True
        FIRST_MESSAGE = True
        return user_input
    else: # toggle auto-switching
        if FIRST_MESSAGE and SWITCH:
            SWITCH = False
            FIRST_MESSAGE = False
        else:
            SWITCH = True
            FIRST_MESSAGE = True
        console.print(f'Auto-switch={SWITCH}')

def variable_command(chat, parm, words):
    """Set or print variables for global or chat settings. Sets a variable specified in the format `variable=value`. If no variable is specified, prints out all variables."""
    if parm:
        var, string = parm.split('=')
        try: value = int(string)
        except ValueError:
            try: value = float(string)
            except ValueError:
                if string == 'True': value = True
                elif string == 'False': value = False
                else: value = string

        if var.isupper() and var in globals():
            globals()[var] = value
            console.print(f'global {var}={value}')
        elif var.islower() and hasattr(chat, var):
            setattr(chat, var, value)
            console.print(f'chat.{var}={value}')
        else:
            console.print('Unrecognized variable')
    else:
        console.print({k: v for k, v in globals().items() if k.isupper()})
        chat.print_vars()

def copy_command(chat, parm, words):
    """Copy a number of chat messages, all messages, or code blocks to the clipboard. Copies the last 2 messages by default."""
    n = None
    if parm:
        if parm == 'all':
            n = len(chat.messages)
        elif parm == 'code':
            text = chat.messages[-1]['content']
            pattern = r'```[a-z]+\s*([\s\S]*?)\s*```'
            matches = re.findall(pattern, text)
            pyperclip.copy('\n\n'.join(matches))
            console.print(f'Copied {len(matches)} code blocks to clipboard\n')
        else:
            n = int(parm)
    else:
        n = 2
    if n:
        pyperclip.copy(chat.messages_string(chat.messages[-n:], divider='\n\n'))
        console.print(f'Copied {n} messages to clipboard\n')

def paste_command(chat, parm, words):
    """Paste the clipboard content as a message. Text after the command is prepended to the message before it is sent."""
    user_input = pyperclip.paste()
    if parm: # Prepend everything after !p
        user_input = ' '.join(words[1:]) + '\n' + user_input
    console.print(user_input)
    return user_input

def exit_command(chat, parm, words):
    """Exit the application."""
    console.print('Goodbye')
    sys.exit()

COMMANDS = (
    ('help', 'h'),
    ('exit', 'e', 'x', 'quit', 'q'),
    ('restart', 'r'),
    ('undo', 'u', 'redo'),
    ('summary', 'sum'),
    ('history', 'hi', 'messages', 'me', 'print', 'pr'),
    ('save', 's'),
    ('load', 'l'),
    ('model', 'm'),
    ('translate', 't'),
    ('debug', 'd'),
    ('auto', 'a'),
    ('variable', 'v', 'var'),
    ('copy', 'c'),
    ('paste', 'p')
)

def help_command(chat, parm, words):
    """Print the help screen."""
    help_strings = []
    for cmds in COMMANDS:
        n = len(cmds)
        name_list = []
        skip = False
        for i, c in enumerate(cmds):
            if skip: skip = False
            else:
                if i < n-1 and c.startswith(cmds[i+1]): # command 2 is abbreviation for command 1:
                    name_list.append(f'[od.yellow]!{cmds[i+1]}[/od.yellow]{c[len(cmds[i+1]):]}')
                    skip = True
                else:
                    name_list.append('!'+c)
        names = ', '.join(name_list)
        doc = globals()[f'{cmds[0]}_command'].__doc__
        help_strings.append(f'[bold]{names}:[/] {doc}')
    console.print_columns(help_strings)
    console.print('')

commands = {}
for cmds in COMMANDS:
    string = f'{cmds[0]}_command'
    func = globals()[string]
    for cmd in cmds:
        commands[cmd] = func




##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##

if __name__ == '__main__':
    console.print_rule('[bold]BlueChat')
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

    if console.RICH:
        window_size_thread = threading.Thread(target=watch_window_size, daemon=True)
        window_size_thread.start()

    last_input_reset = False
    while True:
        # Get a new input
        user_input = chat.input(prefix=(not last_input_reset))
        last_input_reset = False
        if not user_input:
            screen_reset(chat)
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
            if command in commands:
                ret = commands[command](chat, parm, words)
                if ret:
                    user_input = ret
            elif command in ALL_BOTS: # New bot chat by name (!help, !code, !math, etc...)
                FIRST_MESSAGE = False # Just loaded, don't switch context right away
                load_latest(command, reset=True)

        if user_input and not user_input.startswith('!'): # !p and !a can change user_input
            if SWITCH and FIRST_MESSAGE:
                with console.status('Checking context...'):
                    switch = switch_context(user_input)
                    if switch and switch != FILENAME:
                        load_latest(switch, reset=True)

            FIRST_MESSAGE = False

            chat.get_dialogue(user_input)

            if AUTOSAVE:
                chat.save(create_filename())
