import os
import json
import openai
import re
import tiktoken

 ######   #######  ##    ##  ######  ########    ###    ##    ## ########  ######
##    ## ##     ## ###   ## ##    ##    ##      ## ##   ###   ##    ##    ##    ##
##       ##     ## ####  ## ##          ##     ##   ##  ####  ##    ##    ##
##       ##     ## ## ## ##  ######     ##    ##     ## ## ## ##    ##     ######
##       ##     ## ##  ####       ##    ##    ######### ##  ####    ##          ##
##    ## ##     ## ##   ### ##    ##    ##    ##     ## ##   ###    ##    ##    ##
 ######   #######  ##    ##  ######     ##    ##     ## ##    ##    ##     ######

DEBUG = False
LOAD = False
MAX_TOKENS = 4000
SUMMARIZE = False
SUMMARIZE_LEN = 2000

# Run in command line:
# setx OPENAI_API_KEY "your key"
openai.api_key = os.environ["OPENAI_API_KEY"]

# Describes the world the story is set in plus more specific details of the room the character is in. Used for AI description and included at the beginning of every chat for context
setting = "It's a slow afternoon at the pet store."

# Describe the character from the character's perspective (The character is 'you')
red_character = "You are Red, an old, grumpy, widowed man. You're here to buy food for your late wife's cat that you're taking care of. You will never adopt a dog. You would be shocked to learn a dog can talk."
# that only speaks in haiku's. You get embarrassed if your haiku has the wrong number of syllables

# Describe what the character should know about the blue (the player). From the characters perspective (Blue is 'he/she/they')
blue_character = "They are Blue, a cute little dog sitting in the middle of their square, gray cell. There is an enclosure with towels in the back for them to sleep in private."

# How the character knows the player? What caused this conversation to start? Does Red have a goal related to the player?
red_to_blue = "You walk down the aisle of the pet store, towards the back you notice a sign that reads 'ADOPTIONS'. You see a little dog."

# Use this to "describe the scene"
blue_perspective = "You sit in the middle of your square, gray kennel. There is an enclosure with towels in the back for you to sleep in private."

response_directive = "Give the response like this:\n(<adjective describing Red's tone>) <what Red said>"

system_directive = {"role": "system", "content": f"{setting} {red_character} {red_to_blue} {blue_character} What does Red say? {response_directive}"}

system_backup = {"role": "system", "content": f"{setting} {red_character} {red_to_blue} {blue_character} What does Red say?"}

messages = [
    system_directive
]

# LOAD = True
filename = 'alien.json'
if LOAD:
    messages = json.load(open(filename))




######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######

def get_response(request):
#     if DEBUG:
#         return f'turbo: {get_response_turbo(request)}\ndavinci: {get_response_davinci(request)}'
#     else:
#         return get_response_turbo(request)
#
# def get_response_turbo(request):

    complete_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": request }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=complete_messages
    )
    answer = response['choices'][0]['message']['content']
    return answer

def messages_string(messages_to_summarize):
    string = ""
    
    for m in messages_to_summarize:
        s = m['content']
        if m['role'] == 'system':
            string += f' System: {s}'
        if m['role'] == 'user':
            string += f' Blue: {s}'
        if m['role'] == 'assistant':
            string += f' Red: {s}'
    
    return string.strip()

def get_summary(request):
    complete_messages = [
        # {"role": "system", "content": "Summarize this conversation to Red as though he forgot it. Refer to Red as 'You' and refer to Blue as 'They'. Be concise."},
        {"role": "system", "content": "Summarize this conversation. Be concise."},
        {"role": "user", "content": request }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=complete_messages
    )
    answer = response['choices'][0]['message']['content']

    return answer

# def get_response_davinci(request):
#     print(f'request: {request}')
#     response = openai.Completion.create(
#         model="text-davinci-003",
#         max_tokens=64,
#         prompt=request
#     )
#     answer = response['choices'][0]['text']
#     return answer

def get_dialogue(user_input):
    messages.append( {"role": "user", "content": user_input} )

    MAX = 5
    for n in range(MAX):
        if n == MAX-1:
            messages[0] = system_backup
            print("<Parenthesis failed! Moving to backup>")
        response = openai.ChatCompletion.create(
            #model="text-davinci-003",
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.6
        )

        answer = response['choices'][0]['message']['content']

        # check for match
        if re.match(r"\([\w ]{1,32}\).*", answer):
            break

        if re.match(r"(I'm sorry, )?I (didn't|don't) understand what you (mean|meant)", answer) or re.match(r"(I'm sorry, )?I (didn't|don't|did not|do not) understand what (you were|you are|you're) trying to say", answer):
            print("OpenAI didn't understand that. Please try again.")
            messages.pop()
            return False
        
        if DEBUG:
            print('<OpenAI gave unexpected result>')
            print(response['choices'][0]['message']['content'])

    choice = response['choices'][0]
    answer = choice['message']['content']

    messages.append( { "role": "assistant", "content": answer } )
    messages[0] = system_directive

    if DEBUG: # print debug info
        completion = response['usage']['completion_tokens']
        prompt = response['usage']['prompt_tokens']
        total = response['usage']['total_tokens']
        finish = choice['finish_reason']
        print(f'<dialogue tokens=({prompt}, {completion}, {total}) finish={finish}>')
        
    print(f'({len(messages)-1}) Red: {answer}')
    return answer

def summarize(n=5000):
    """Summarize the first n tokens worth of conversation. Default n > 4096 means summarize everything"""
    i = 1
    if num_tokens_from_messages(messages) > n:
        while num_tokens_from_messages(messages[:i]) < n:
            i += 1
        string = messages_string(messages[1:i])
    else:
        string = messages_string(messages)
        
    print(f'<Summary: {get_summary(string)}>')

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages. Copied from OpenAI example"""
    # encoding = tiktoken.get_encoding("cl100k_base")
    # if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens




##     ##    ###    #### ##    ##
###   ###   ## ##    ##  ###   ##
#### ####  ##   ##   ##  ####  ##
## ### ## ##     ##  ##  ## ## ##
##     ## #########  ##  ##  ####
##     ## ##     ##  ##  ##   ###
##     ## ##     ## #### ##    ##

# Chat GPT
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

print("You wake up in your kennel at the back of the pet store.")

print(get_response(f"{setting} {blue_perspective} Write a couple sentences to describe the scene"))

print("You see an old man.")

while True:
    user_input = input(f'({len(messages)}) Blue: ')

    if user_input == '!summary':
        summarize()
        continue

    if user_input == '!print':
        for m in messages:
            print(m['role']+': '+m['content'])
        continue

    if user_input == '!repr':
        print(repr(messages))
        continue

    if user_input == '!save':
        file = input('Filename: ')
        json.dump(messages, open(file, 'w'), indent='\t')
        continue

    if user_input == '!debug':
        DEBUG = not DEBUG
        continue

    while not get_dialogue(user_input):
        user_input = input(f'({len(messages)}) Blue: ')

    if SUMMARIZE:
        # TODO: Run summarize when almost out of memory
        pass
    else: # If SUMMARIZE = False, just forget oldest messages
        while num_tokens_from_messages(messages) > MAX_TOKENS:
            last = messages[1]['content']
            if DEBUG: print(f'<Forgetting: {last}>')
            del messages[1]
