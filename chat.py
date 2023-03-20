import os
import sys
import json
import re
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

DEBUG = False
MAX_TOKENS = 4000
DIALOGUE_TRIES = 3
VALIDATE_FORMAT = False

SUMMARIZE = True
SUMMARIZED = False
SUMMARIZE_LEN = 1000

# MODEL = "gpt-3.5-turbo" # Cheaper
MODEL = "gpt-4" # Better

openai.api_key = os.environ["OPENAI_API_KEY"]

# Describes the world the story is set in plus more specific details of the room the character is in. Used for AI description and included at the beginning of every chat for context
setting = "It's a slow afternoon at the pet store."

# Describe the character from the character's perspective (The character is 'you')
red_character = "You are Red, an old, grumpy, widowed man. You're here to buy food for your late wife's cat that you're taking care of. You will never adopt a dog. You would be shocked to learn a dog could speak."
# that only speaks in haiku's. You get embarrassed if your haiku has the wrong number of syllables

# Describe what the character should know about the blue (the player). From the characters perspective (Blue is 'he/she/they')
blue_character = "They are Blue, a cute little dog sitting in the middle of their square, gray cell. There is an enclosure with towels in the back for them to sleep in private."

# How the character knows the player? What caused this conversation to start? Does Red have a goal related to the player?
red_to_blue = "You walk down the aisle of the pet store, towards the back you notice a sign that reads 'ADOPTIONS'. You see a little dog."

# Use this to ask the AI to describe the scene
blue_perspective = "You sit in the middle of your square, gray kennel. There is an enclosure with towels in the back for you to sleep in private."

response_directive = "Give the response like this:\n(<adjective describing Red's tone>) <what Red said>"

system_directive = {"role": "system", "content": f"{setting} {red_character} {red_to_blue} {blue_character} What does Red say? {response_directive}"}

system_backup = {"role": "system", "content": f"{setting} {red_character} {red_to_blue} {blue_character} What does Red say?"}

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

def get_response(request):
    complete_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": request }
    ]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=complete_messages
    )
    answer = response['choices'][0]['message']['content']
    return answer

def messages_string(messages_to_summarize):
    string = ""
    
    for m in messages_to_summarize:
        s = m['content']
        if m['role'] == 'system':
            # string += f' System: {s}'
            string += f' {s}'
        if m['role'] == 'user':
            string += f' Blue: {s}'
        if m['role'] == 'assistant':
            string += f' Red: {s}'
    
    return string.strip()

def get_summary(request):
    complete_messages = [
        # {"role": "system", "content": "Summarize this conversation. Be concise."},
        {"role": "system", "content": "Summarize this conversation."},
        {"role": "user", "content": request }
    ]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=complete_messages
    )
    answer = response['choices'][0]['message']['content']

    return answer

def get_dialogue(user_input):
    messages.append( {"role": "user", "content": user_input} )
    history.append( {"role": "user", "content": user_input} )

    for n in range(DIALOGUE_TRIES):
        if n == DIALOGUE_TRIES-1:
            messages[0] = system_backup
            print("<Parenthesis failed! Moving to backup>")
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            temperature=0.6
        )

        answer = response['choices'][0]['message']['content']

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
            print(response['choices'][0]['message']['content'])

    choice = response['choices'][0]
    answer = choice['message']['content']

    messages.append( { "role": "assistant", "content": answer } )
    history.append( { "role": "assistant", "content": answer } )
    if VALIDATE_FORMAT:
        messages[0] = system_directive

    if DEBUG: # print debug info
        completion = response['usage']['completion_tokens']
        prompt = response['usage']['prompt_tokens']
        total = response['usage']['total_tokens']
        finish = choice['finish_reason']
        print(f'<dialogue tokens=({prompt}, {completion}, {total}) finish={finish}>')
        
    print(f'({len(history)-1}) Red: {answer}')
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

ENCODING = tiktoken.encoding_for_model(MODEL)
def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages. Copied from OpenAI example"""
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(ENCODING.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def file(user_input=''):
    if re.match(r"!\w*$", user_input): # Just '!command', no filename
    # if user_input == '!save':
        filename = input('Filename: ') # Ask for filename
    else: # '!command filename'
        filename = user_input.split(' ')[1] # Filename is after space

    if filename.endswith('.json'): # Remove .json if specified
        filename = filename.split('.')[0]

    return filename

def load(filename):
    global messages, history
    messages = json.load(open(f'{filename}.json'))
    if os.path.isfile(f'{filename}_h.json'):
        history = json.load(open(f'{filename}_h.json'))
    else:
        history = [m for m in messages if m['role'] != 'system']
    if messages[-1]['role'] == 'user': # If last response is user...
        messages.pop() # Remove it so user can respond
        history.pop() # History too




##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##

print(sys.argv)
if len(sys.argv) == 2: # If there's an argument...
    # Load an existing conversation
    arg = sys.argv[1]
    if arg.endswith('.json'):
        arg = arg.split('.')[0]
    print(f'Loading {arg}.json')
    load(arg)
    # Print last message from loaded file
    m = messages[-1]['content']
    print(f'({len(history)-1}) {m}')

else: # Setup a new conversation
    print("You wake up in your kennel at the back of the pet store.")

    print(get_response(f"{setting} {blue_perspective} Write a couple sentences to describe the scene"))

    print("You see an old man.")

try:
    while True:
        # Get a new input
        user_input = input(f'({len(history)}) Blue: ').strip()

        # Process commands
        if user_input == '!exit':
            print('Goodbye')
            break

        if user_input == '!summary':
            print(f'<Summary: {summarize()}>')
            continue

        if user_input == '!print' or user_input == '!messages':
            for m in messages:
                print(m['role']+': '+m['content'])
            continue

        if user_input == '!history':
            for m in history:
                print(m['role']+': '+m['content'])
            continue

        if user_input.startswith('!save'):
            filename = file(user_input)
            json.dump(messages, open(f'{filename}.json', 'w'), indent='\t')
            json.dump(history, open(f'{filename}_h.json', 'w'), indent='\t')
            continue

        if user_input.startswith('!load'):
            filename = file(user_input)
            load(filename)
            continue

        if user_input == '!debug':
            DEBUG = not DEBUG
            print(f'{DEBUG=}')
            continue

        # This isn't a command, process dialogue
        while not get_dialogue(user_input):
            user_input = input(f'({len(history)}) Blue: ').strip()

        # Summarize or forget if approaching token limit
        if SUMMARIZE:
            if num_tokens_from_messages(messages) > MAX_TOKENS:
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

except Exception as e: # Unexpected error. Dump all data
    print(e)
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    json.dump(messages, open(f'crash_{time}.json', 'w'), indent='\t')
    json.dump(history, open(f'crash_{time}_h.json', 'w'), indent='\t')    
    print(f'Data saved to crash_{time}.json')
