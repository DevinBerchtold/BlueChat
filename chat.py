import os
import sys
import json
import re
import time
from datetime import datetime

import openai
import tiktoken




 ######   #######  ##    ##  ######  ########    ###    ##    ## ########  ######
##    ## ##     ## ###   ## ##    ##    ##      ## ##   ###   ##    ##    ##    ##
##       ##     ## ####  ## ##          ##     ##   ##  ####  ##    ##    ##
##       ##     ## ## ## ##  ######     ##    ##     ## ## ## ##    ##     ######
##       ##     ## ##  ####       ##    ##    ######### ##  ####    ##          ##
##    ## ##     ## ##   ### ##    ##    ##    ##     ## ##   ###    ##    ##    ##
 ######   #######  ##    ##  ######     ##    ##     ## ##    ##    ##     ######

YAML = True
if YAML:
    import yaml
JSON = True

DEBUG = False
MAX_TOKENS = 4000
DIALOGUE_TRIES = 3
VALIDATE_FORMAT = False

SUMMARIZE = True
SUMMARIZED = False
SUMMARIZE_LEN = 1000

# Translate mode translates requests and responses
TRANSLATE = False
USER_LANG = 'English'
AI_LANG = 'Chinese'

AUTOSAVE = True
AUTOLOAD = True
AUTOLOAD_DEFAULT = 'help'
FILENAME = datetime.now().strftime('%Y-%m-%d')

# MODEL = "gpt-3.5-turbo" # Cheaper
MODEL = "gpt-4" # Better
STREAM = True
AI_NAME = 'Red'
USER_NAME = 'Blue'
try:
    USER_ID = os.getlogin()
except Exception:
    USER_ID = 'unknown'

openai.api_key = os.environ["OPENAI_API_KEY"]

# Describes the world the story is set in plus more specific details of the room the character is in. Used for AI description and included at the beginning of every chat for context
setting = "It's a slow afternoon at the pet store."

# Describe the character from the character's perspective (The character is 'you')
red_character = f"You are {AI_NAME}, an old, grumpy, widowed man. You're here to buy food for your late wife's cat that you're taking care of. You will never adopt a dog. You would be shocked to learn a dog could speak."
# that only speaks in haiku's. You get embarrassed if your haiku has the wrong number of syllables

# Describe what the character should know about the user. From the characters perspective (User is 'he/she/they')
blue_character = f"They are {USER_NAME}, a cute little dog sitting in the middle of their square, gray cell. There is an enclosure with towels in the back for them to sleep in private."

# How the character knows the user? What caused this conversation to start? Does the AI have a goal related to the user?
red_to_blue = "You walk down the aisle of the pet store, towards the back you notice a sign that reads 'ADOPTIONS'. You see a little dog."

# Use this to ask the AI to describe the scene
blue_perspective = "You sit in the middle of your square, gray kennel. There is an enclosure with towels in the back for you to sleep in private."

response_directive = f"Give the response like this:\n(<adjective describing {AI_NAME}'s tone>) <what {AI_NAME} said>"

system_directive = {"role": "system", "content": f"{setting} {red_character} {red_to_blue} {blue_character} What does {AI_NAME} say? {response_directive}"}

system_backup = {"role": "system", "content": f"{setting} {red_character} {red_to_blue} {blue_character} What does {AI_NAME} say?"}

messages = [
    system_directive
]

history = []




######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######

def remove_spaces(string):
    lines = string.split('\n') # split into lines
    lines = [l.rstrip() for l in lines] # remove trailing spaces
    return '\n'.join(lines) # join and return

encoding = tiktoken.encoding_for_model(MODEL)

def chat_complete(messages, model=MODEL, temperature=0.8, prefix='', print_result=True):
    for _ in range(5):
        try:
            t0 = time.time()
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                user=USER_ID,
                stream=(STREAM and print_result)
            )
            full_string = ''
            if STREAM and print_result:
                sys.stdout.write(prefix)
                sys.stdout.flush()
                
                # Iterate through the stream of events
                collected_chunks = [] # Store all the chunks in a list
                for chunk in response:
                    collected_chunks.append(chunk)  # Save the event response
                    chunk_string = chunk['choices'][0]['delta'].get('content', '')  # Extract the message

                    if chunk_string != '':
                        full_string += chunk_string
                        sys.stdout.write(chunk_string)
                        sys.stdout.flush()

                print('')

                total_time = time.time()-t0
                if DEBUG: # print debug info
                    chunks = len(collected_chunks)
                    tokens = len(encoding.encode(full_string))
                    finish = chunk['choices'][0]['finish_reason']
                    print(f'<chat {chunks=}, {tokens=}, {finish=}, time={total_time:.3}s>')

                return remove_spaces(full_string) # Remove trailing spaces and return

            else:
                full_string = response['choices'][0]['message']['content']
                if print_result:
                    print(prefix+full_string+'\n')
                
                total_time = time.time()-t0

                if DEBUG: # print debug info
                    completion = response['usage']['completion_tokens']
                    prompt = response['usage']['prompt_tokens']
                    total = response['usage']['total_tokens']
                    finish = response['choices'][0]['finish_reason']
                    print(f'<chat tokens=({prompt}, {completion}, {total}), {finish=}, time={total_time:.3}s>')


                return remove_spaces(full_string) # Remove trailing spaces and return

        except Exception as e:
            print(f"Error: {e} Trying again in 1 second...")
            time.sleep(1)
    raise ConnectionError("Failed to access OpenAI API after 5 attempts.")

def get_response(request):
    complete_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": request }
    ]
    answer = chat_complete(complete_messages, print_result=False)
    return answer

def messages_string(messages_to_summarize):
    string = ""

    for m in messages_to_summarize:
        s = m['content']
        if m['role'] == 'system':
            # string += f' System: {s}'
            string += f' {s}'
        if m['role'] == 'user':
            string += f' {USER_NAME}: {s}'
        if m['role'] == 'assistant':
            string += f' {AI_NAME}: {s}'

    return string.strip()

def get_translation(request, from_lang, to_lang):
    complete_messages = [
        {"role": "system", "content": f"Translate the {from_lang} input to {to_lang}. Preserve the meaning, tone, and formatting."},
        {"role": "user", "content": request }
    ]
    answer = chat_complete(complete_messages, print_result=True)
    return answer

def get_summary(request):
    complete_messages = [
        # {"role": "system", "content": "Summarize this conversation. Be concise."},
        {"role": "system", "content": "Summarize this conversation."},
        {"role": "user", "content": request }
    ]
    answer = chat_complete(complete_messages, print_result=False)
    return answer

def get_dialogue(user_input):
    message = {"role": "user", "content": user_input}
    messages.append(message)
    history.append(message)

    answer = ''

    for n in range(DIALOGUE_TRIES):
        if n == DIALOGUE_TRIES-1:
            messages[0] = system_backup
            print("<Parenthesis failed! Moving to backup>")
            
        answer = chat_complete(messages, prefix=f"\n({len(history)+1}) {AI_NAME}: ")

        # check for match
        if VALIDATE_FORMAT:
            if re.match(r"\([\w ]{1,32}\).*", answer):
                break
        else: # Not validating, ignore everything else in loop
            break

        if re.match(r"(I'm sorry, )?I (didn't|don't) understand what you (mean|meant)", answer) or re.match(r"(I'm sorry, )?I (didn't|don't|did not|do not) understand what (you were|you are|you're) trying to say", answer):
            print("OpenAI didn't understand that. Please try again.")
            messages.pop()
            history.pop()
            return False

        if DEBUG:
            print('<OpenAI gave unexpected result>')
            print(answer)

    message = { "role": "assistant", "content": answer }
    messages.append(message)
    history.append(message)
    if VALIDATE_FORMAT:
        messages[0] = system_directive

    return answer

def summarize(n=5000, delete=False):
    """Summarize the first n tokens worth of conversation. Default n > 4096 means summarize everything"""
    i = 1
    if num_tokens_from_messages(messages) > n:
        while num_tokens_from_messages(messages[:i]) < n:
            i += 1
        string = messages_string(messages[1:i])
        if delete:
            del messages[1:i]
    else:
        string = messages_string(messages[1:])
        if delete:
            del messages[1:]

    string.strip()
    if DEBUG:
        print(f'<Summarizing: {string}>')

    return get_summary(string)

def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages. Copied from OpenAI example"""
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

if YAML: # YAML copy for readability. Only define functions if needed
    def str_presenter(dumper, data):
        s = '|' if '\n' in data else None # Change style to | if multiple lines
        # print(f'{s} {data[:10]}')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=s)

    def dict_presenter(dumper, data):
        f = all([s.isupper() for s in data.keys()]) # True if all keys in data are uppercase
        return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=f)

    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(dict, dict_presenter)

def load(filename):
    global messages, history
    messages, history = [], []
    if os.path.isfile(f'conversations/{filename}.json'):
        print(f"Loading {filename}.json")
        data = json.load(open(f'conversations/{filename}.json', encoding='utf8'))
    elif YAML:
        if os.path.isfile(f'conversations/{filename}.yaml'):
            print(f"Loading {filename}.yaml")
            data = yaml.safe_load(open(f'conversations/{filename}.yaml', 'r', encoding='utf8'))
        else:
            return False
    else:
        return False
    
    # We loaded a file, extract the data
    if isinstance(data, dict):
        for k, v in data.items():
            if k == 'global_vars':
                for x, y in v.items(): # for each GLOBAL VAR
                    if x not in ['DEBUG', 'USER_ID']: # Don't restore these
                        globals()[x] = y
            else: # messages, history
                globals()[k] = v
    elif isinstance(data, list): # List of messages format
        messages = data
    else:
        print('Unknown Data')
    
    if history == []: # No history? Copy from messages
        history = [m for m in messages if m['role'] != 'system']

    else: # We have history. Connect to messages so they are in sync
        n = 1 # Count the number of identical messages
        while (history[-n] == messages[-n]) and (n < len(history)):
            n += 1
        history[-n+1:] = messages[-n+1:] # Reassign so they are the same object

    if messages[-1]['role'] == 'user': # If last response is user...
        messages.pop() # Remove it so user can respond
        history.pop() # History too

    if len(messages) > 1:
        if messages[1]['role'] == 'system': # Second messages is system...
            global SUMMARIZE, SUMMARIZED # This means it was summarized
            SUMMARIZE = True
            SUMMARIZED = True

        # Print last messages from loaded file
        # print(f"\n(0) {messages[0]['content']}") # Print system message
        if len(messages) > 2:
            l = len(history) # Print last question and answer
            print(f"\n({l-1}) {USER_NAME}: {messages[-2]['content']}")
            print(f"\n({l}) {AI_NAME}: {messages[-1]['content']}")

    return True

def save(filename):
    if not os.path.exists("conversations"):
        os.makedirs('conversations')

    # All (uppercase) global variables minus built-in variables (__)
    global_vars = {k: v for k, v in globals().items() if not k.startswith("__") and k.isupper()}
    data = {'global_vars': global_vars, 'messages': messages, 'history': history}

    if YAML:
        yaml.safe_dump(data, open(f'conversations/{filename}.yaml', 'w', encoding='utf8'), allow_unicode=True, sort_keys=False, width=float("inf"))
    if JSON:
        json.dump(data, open(f'conversations/{filename}.json', 'w', encoding='utf8'), indent='\t', ensure_ascii=False)




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
    load(arg)

elif AUTOLOAD:
    filename = datetime.now().strftime('%Y-%m-%d')
    if not load(filename):
        load(AUTOLOAD_DEFAULT)

else: # Setup a new conversation (Blue)
    print("You wake up in your kennel at the back of the pet store.")
    print(get_response(f"{setting} {blue_perspective} Write a couple sentences to describe the scene"))
    print("You see an old man.")

# try:
while True:
    # Get a new input
    user_input = remove_spaces(input(f'\n({len(history)+1}) {USER_NAME}: '))

    if user_input.startswith('!'):
        input_split = user_input.split(' ')
        command = input_split[0]
        if len(input_split) > 1:
            parm = input_split[1]
        else:
            parm = ''

        # Process commands
        if command == '!exit':
            print('Goodbye')
            break

        elif command == '!restart':
            print('Restarting conversation')
            messages = [messages[0]]
            history = []

        elif command == '!redo' or command == '!undo':
            if len(messages) > 1:
                print('Removing last 2 messages')
                messages = messages[:-2]
                history = history[:-2]
            else:
                print('No messages to remove')

        elif command == '!summary':
            if parm:
                print(f'<Summary: {summarize(int(parm))}>')
            else:
                print(f'<Summary: {summarize()}>')

        elif command == '!print' or command == '!messages':
            for m in messages:
                print(m['role']+': '+m['content'])

        elif command == '!history':
            for m in history:
                print(m['role']+': '+m['content'])

        elif command == '!save':
            if parm:
                save(parm)
                print(f"Data saved to {parm}")
            else:
                save(FILENAME)
                print(f"Data saved to {FILENAME}")

        elif command == '!load':
            if parm:
                load(parm)
            else:
                load(FILENAME)

        elif command == '!model':
            if parm:
                MODEL = parm
            print(f"<Model: {MODEL}>")

        elif command == '!translate':
            if user_input == '!translate':
                TRANSLATE = not TRANSLATE
            else:
                if len(input_split) == 2:
                    TRANSLATE = True
                    AI_LANG = parm
                    USER_LANG = 'English'
                elif len(input_split) == 3:
                    TRANSLATE = True
                    AI_LANG = parm
                    USER_LANG = input_split[2]

            print(f'<Translate={TRANSLATE}, AI={AI_LANG}, User={USER_LANG}>')

        elif command == '!debug':
            DEBUG = not DEBUG
            print(f'<{DEBUG=}>')
        
    else: # Doesn't start with '!'
        if TRANSLATE:
            user_input = get_translation(user_input, USER_LANG, AI_LANG)

        # This isn't a command, process dialogue
        answer = get_dialogue(user_input)
        while not answer:
            user_input = remove_spaces(input(f'\n({len(history)}) {USER_NAME}: '))
            answer = get_dialogue(user_input)

        if TRANSLATE:
            answer = get_translation(answer, AI_LANG, USER_LANG)

        if AUTOSAVE:
            save(FILENAME)

        # Summarize or forget if approaching token limit
        if SUMMARIZE:
            total_tokens = num_tokens_from_messages(messages)
            if DEBUG:
                print(f'<{total_tokens=}>')
            if total_tokens > MAX_TOKENS:
                summary = summarize(SUMMARIZE_LEN, True)
                if DEBUG: print(f'<Summary: {summary}>')

                if SUMMARIZED: # Already summarized?
                    # Replace old summary system message
                    messages[1] = {"role": "system", "content": summary}
                else:
                    # Add a new summary system message
                    messages.insert(1, {"role": "system", "content": summary})
                    SUMMARIZED = True

        else: # If SUMMARIZE = False, just forget oldest messages
            while num_tokens_from_messages(messages) > MAX_TOKENS:
                last = messages[1]['content']
                if DEBUG: print(f'<Forgetting: {last}>')
                del messages[1]

# except Exception as e: # Unexpected error. Dump all data
#     print(e)
#     time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#     save(f"crash_{time_string}")
