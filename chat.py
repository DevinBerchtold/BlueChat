import os
import sys
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
FILENAME = datetime.now().strftime('%Y-%m-%d')

# Describes the world the story is set in plus more specific details of the room the character is in. Used for AI description and included at the beginning of every chat for context
setting = "It's a slow afternoon at the pet store."

# Describe the character from the character's perspective (The character is 'you')
red_character = "You are {AI_NAME}, an old, grumpy, widowed man. You're here to buy food for your late wife's cat that you're taking care of. You will never adopt a dog. You would be shocked to learn a dog could speak."
# that only speaks in haiku's. You get embarrassed if your haiku has the wrong number of syllables

# Describe what the character should know about the user. From the characters perspective (User is 'he/she/they')
blue_character = "They are {USER_NAME}, a cute little dog sitting in the middle of their square, gray cell. There is an enclosure with towels in the back for them to sleep in private."

# How the character knows the user? What caused this conversation to start? Does the AI have a goal related to the user?
red_to_blue = "You walk down the aisle of the pet store, towards the back you notice a sign that reads 'ADOPTIONS'. You see a little dog."

# Use this to ask the AI to describe the scene
blue_perspective = "You sit in the middle of your square, gray kennel. There is an enclosure with towels in the back for you to sleep in private."

response_directive = "Give the response like this:\n(<adjective describing {AI_NAME}'s tone>) <what {AI_NAME} said>"

system_directive = {"role": "system", "content": "{setting} {red_character} {red_to_blue} {blue_character} What does {AI_NAME} say? {response_directive}"}

system_backup = {"role": "system", "content": "{setting} {red_character} {red_to_blue} {blue_character} What does {AI_NAME} say?"}




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
    chat = conversation.Conversation(filename=arg)

elif AUTOLOAD:
    filename = datetime.now().strftime('%Y-%m-%d')
    chat = conversation.Conversation(filename=filename)

else: # Setup a new conversation (Blue)
    chat = conversation.Conversation()
    print("You wake up in your kennel at the back of the pet store.")
    print(conversation.get_response(f"{setting} {blue_perspective} Write a couple sentences to describe the scene"))
    print("You see an old man.")

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
        if command == '!exit':
            print('Goodbye')
            break

        elif command == '!restart':
            print('Restarting conversation')
            chat.messages = [chat.messages[0]]
            chat.history = []

        elif command == '!redo' or command == '!undo':
            if len(chat.messages) > 1:
                print('Removing last 2 messages')
                chat.messages = chat.messages[:-2]
                chat.history = chat.history[:-2]
            else:
                print('No messages to remove')

        elif command == '!summary':
            if parm:
                print(f'<Summary: {chat.summarize(int(parm))}>')
            else:
                print(f'<Summary: {chat.summarize()}>')

        elif command == '!print' or command == '!messages':
            for m in chat.messages:
                print(m['role']+': '+m['content'])

        elif command == '!history':
            for m in chat.history:
                print(m['role']+': '+m['content'])

        elif command == '!save':
            if parm:
                chat.save(parm)
                print(f"Data saved to {parm}")
            else:
                chat.save(FILENAME)
                print(f"Data saved to {FILENAME}")

        elif command == '!load':
            if parm:
                chat.load(parm)
            else:
                chat.load(FILENAME)

        elif command == '!model':
            if parm:
                conversation.MODEL = parm
            print(f"<Model: {conversation.MODEL}>")

        elif command == '!translate':
            if user_input == '!translate':
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

        elif command == '!debug':
            conversation.DEBUG = not conversation.DEBUG
            print(f'<Debug={conversation.DEBUG}>')
        
    else: # Doesn't start with '!'
        answer = chat.get_dialogue(user_input)

        if AUTOSAVE:
            chat.save(FILENAME)

# except Exception as e: # Unexpected error. Dump all data
#     print(e)
#     time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#     save(f"crash_{time_string}")
