import os
import sys
import threading
import time
import re
import inspect

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

def restart_command(chat, *_):
    """Restart the chat by clearing memory and creating a new save file."""
    global DATE, FILENUM
    console.print('Restarting conversation')
    chat.reset(f'{BOT_FOLDER}/{FILENAME}')
    _, DATE, FILENUM = filename_vars(latest_file_today(FILENAME))
    FILENUM += 1

def undo_command(chat, num, *_):
    """Undo and remove the last `num` messages in the chat. Removes the last 2 messages if `num` is not specified."""
    if num:
        n = int(num)
    else:
        n = 2
    if len(chat.messages) >= n:
        chat.messages = chat.messages[:-n]
        chat.history = chat.history[:-n]
        console.print(f'Removing last {n} messages')
    else:
        console.print('No messages to remove')

def summary_command(chat, tokens, *_):
    """Summarize the last `tokens` of the conversation. Summarizes the entire conversation if `tokens` is not specified."""
    if tokens:
        s = chat.summarize_messages(int(tokens))
    else:
        s = chat.summarize_messages()
    console.print(f'Summary: {s}')

def history_command(chat, *_):
    """Print the conversation system messages and full chat history."""
    console.print_filename(f'{CHAT_FOLDER}/{FILENAME}_{DATE}_{FILENUM}')
    chat.print_systems()
    chat.print_messages()

def save_command(chat, filename, *_):
    """Save the chat to `filename`. Automatically adds file extension and names the file if `filename` is not specified."""
    if filename:
        if '/' in filename:
            file = filename
        else:
            file = f'{CHAT_FOLDER}/{filename}'
    else:
        file = create_filename()
    chat.save(file)
    console.print(f"Data saved to {file}")

def load_command(chat, filename, *_):
    """Load the chat from `filename`. Loads the latest file if `filename` is a bot name or the exact file if it's a filename."""
    global FIRST_MESSAGE
    FIRST_MESSAGE = False # Just loaded, don't switch context right away
    if filename:
        if '_' in filename:
            if '/' in filename:
                chat.load(filename)
            else:
                chat.load(f'{CHAT_FOLDER}/{filename}')
        else:
            if filename in ALL_BOTS:
                load_latest(filename)
            else:
                if '/' in filename:
                    chat.load(filename)
                else:
                    chat.load(f'{CHAT_FOLDER}/{filename}')
    else:
        console.print('Error: No filename specified')

def model_command(chat, model, reset):
    """Set the LLM model to be used in the chat if `model` is specified, or prints the current model otherwise. Model can be 'gpt-3.5', 'gpt-4', or 'bison' or an abbreviation like '3', '4', or 'b'. If `reset` is 'r' or 'redo', the last message is regenerated with the new model."""
    if model in ['3', '3.5', 'gpt-3', 'gpt-3.5-turbo']:
        chat.model = 'gpt-3.5-turbo'
    elif model in ['4', 'gpt-4-turbo']:
        chat.model = 'gpt-4-1106-preview'
    elif model in ['gpt-4']:
        chat.model = 'gpt-4'
    elif model in ['32', '32k']:
        chat.model = 'gpt-4-32k'
    elif model in ['2', 'b', 'palm', 'bison', 'models/chat-bison-001']:
        chat.model = 'models/chat-bison-001'
    elif model in ['u', 'unicorn', 'models/chat-unicorn-001']:
        chat.model = 'models/chat-unicorn-001'
    else:
        chat.model = model

    if reset in ['r', 'redo', 'reset']:
        user_message = chat.messages[-2]['content']
        if len(chat.messages) >= 2:
            chat.messages = chat.messages[:-2]
            chat.history = chat.history[:-2]
        console.print(f'Redoing with {chat.model}...')
        return user_message
    console.print(f"Model={chat.model}")

def translate_command(chat, ai, user):
    """Toggle translate mode if no parameters are specified. If specified, `ai` is the AI language and `user` is the users language."""
    if ai:
        chat.translate = True
        chat.ai_lang = ai
        if user:
            chat.user_lang = user
        else:
            chat.user_lang = 'English'
    else:
        chat.translate = not chat.translate

    console.print(f'Translate={chat.translate}, AI={chat.ai_lang}, User={chat.user_lang}')

def debug_command(*_):
    """Toggle debug mode for all modules."""
    global DEBUG
    DEBUG = not DEBUG
    files.DEBUG = DEBUG
    console.DEBUG = DEBUG
    conversation.DEBUG = DEBUG
    console.print(f'Debug={DEBUG}')

def auto_command(_, *text):
    """Toggle auto-switch context mode. If `text` is specified, context is checked and `text` is sent as a message."""
    global FIRST_MESSAGE, SWITCH
    text = [w for w in text if w is not None]
    if text: # turn auto-switch on and take user input
        user_input = ' '.join(text) # everything after !a
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

def variable_command(chat, parm, *_):
    """Set or print variables for global or chat settings. If `parm` is specified in the format `variable=value`, set the variable. If not specified, print all variables."""
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

def copy_command(chat, parm, *_):
    """Copy the last `parm` chat messages to the clipboard if `parm` is a number or 'all'. Copy code blocks from the last message if `parm` is 'code'. Copies 2 messages by default."""
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

def paste_command(_, *text):
    """Paste the clipboard content and send as a message. If `text` is specified, `text` is prepended to the message before sending."""
    user_input = pyperclip.paste()
    text = [w for w in text if w is not None]
    if text: # Prepend everything after !p
        user_input = ' '.join(text) + '\n\n' + user_input
    console.print(user_input)
    return user_input

def exit_command(*_):
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

def help_command(*_):
    """Print the help screen."""
    help_strings = []
    for cmds in COMMANDS:
        n = len(cmds)
        name_list = []
        skip = False
        for i, c in enumerate(cmds):
            if skip: skip = False
            else: # Command n+1 might be alias for command n
                if i < n-1 and c.startswith(cmds[i+1]):
                    name_list.append(f'[bold][od.cyan]!{cmds[i+1]}[/]{c[len(cmds[i+1]):]}[/]')
                    skip = True # Skip the next one
                else:
                    name_list.append(f'[bold]!{c}[/]')
        names = '[od.dim]|[/]'.join(name_list)
        func = globals()[f'{cmds[0]}_command']
        doc = func.__doc__

        # Create command parms from function arguments
        parms = inspect.signature(func).parameters
        parms = [p for p in parms if p not in ('chat', '_')]
        if parms: parm_string = ' '+' '.join(parms)
        else: parm_string = ''
        doc_replaced = re.sub(r'`(.*?)`', r'[od.yellow]\1[/]', doc) # Highlight `parms`

        help_strings.append(f'{names}[bold][od.yellow]{parm_string}[/][/]: {doc_replaced}')
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
            words = user_input[1:].split(' ') # remove ! and split
            command = words[0]
            args = words[1:]
            if len(args) < 2: # All commands take 2 optional arguments (for now)
                args += [None] * (2-len(args)) # Extend to minimum length

            # Process commands
            if command in commands:
                ret = commands[command](chat, *args)
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
