import time
from datetime import datetime

import openai
import tiktoken

from globals import *




 ######   #######  ##    ##  ######  ########    ###    ##    ## ########  ######
##    ## ##     ## ###   ## ##    ##    ##      ## ##   ###   ##    ##    ##    ##
##       ##     ## ####  ## ##          ##     ##   ##  ####  ##    ##    ##
##       ##     ## ## ## ##  ######     ##    ##     ## ## ## ##    ##     ######
##       ##     ## ##  ####       ##    ##    ######### ##  ####    ##          ##
##    ## ##     ## ##   ### ##    ##    ##    ##     ## ##   ###    ##    ##    ##
 ######   #######  ##    ##  ######     ##    ##     ## ##    ##    ##     ######

DEBUG = False
# MODEL = "gpt-3.5-turbo" # Cheaper
MODEL = "gpt-4" # Better
STREAM = True

try:
    USER_ID = os.getlogin()
except Exception:
    USER_ID = 'unknown'

openai.api_key = os.environ["OPENAI_API_KEY"]




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

def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages. Copied from OpenAI example"""
    encoding = tiktoken.encoding_for_model(MODEL)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def chunks_from_response(response):
    for chunk in response:
        yield chunk['choices'][0]['delta'].get('content', '')

def chat_complete(openai_messages, temperature=0.8, prefix='', max_tokens=None, print_result=True):
    encoding = tiktoken.encoding_for_model(MODEL)
    for _ in range(5):
        try:
            t0 = time.time()
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                user=USER_ID,
                stream=(STREAM and print_result)
            )
            full_string = ''
            if STREAM and print_result:
                if WORD_WRAP:
                    full_string = print_wrapped(chunks_from_response(response), prefix=prefix)
                else:
                    # Iterate through the stream of events
                    sys.stdout.write(prefix)
                    sys.stdout.flush()
                    for chunk in response:
                        chunk_string = chunk['choices'][0]['delta'].get('content', '')  # Extract the message
                        full_string += chunk_string
                        sys.stdout.write(chunk_string)
                        sys.stdout.flush()
                print('')

                total_time = time.time()-t0
                if DEBUG: # print debug info
                    # chunks = len(collected_chunks)
                    tokens = len(encoding.encode(full_string))
                    # finish = collected_chunks[-1]['choices'][0]['finish_reason']
                    # print(f'<chat {chunks=}, {tokens=}, {finish=}, time={total_time:.3}s>')
                    print(f'<chat {tokens=}, time={total_time:.3}s>')

            else:
                full_string = response['choices'][0]['message']['content']
                if print_result:
                    print('\n'+prefix+full_string)

                total_time = time.time()-t0

                if DEBUG: # print debug info
                    completion = response['usage']['completion_tokens']
                    prompt = response['usage']['prompt_tokens']
                    total = response['usage']['total_tokens']
                    finish = response['choices'][0]['finish_reason']
                    print(f'<chat tokens=({prompt}, {completion}, {total}), {finish=}, time={total_time:.3}s>')

            # return remove_spaces(full_string) # Remove trailing spaces and return
            return full_string

        except Exception as e:
            print(f"Error: {e} Trying again in 1 second...")
            time.sleep(1)
    raise ConnectionError("Failed to access OpenAI API after 5 attempts.")

def get_complete(system, user_request, max_tokens=None, print_result=True):
    complete_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_request }
    ]
    answer = chat_complete(complete_messages, max_tokens=max_tokens, print_result=print_result)
    return answer

def get_response(request):
    return get_complete("You are a helpful assistant", request, print_result=False)

def get_translation(request, from_lang, to_lang):
    return get_complete(
        f"Translate the {from_lang} input to {to_lang}. Preserve the meaning, tone, and formatting.",
        request, print_result=False
    )

def get_summary(request):
    return get_complete("Summarize this conversation.", request, print_result=False)




 ######   #######  ##    ## ##     ## ######## ########   ######     ###    ######## ####  #######  ##    ##
##    ## ##     ## ###   ## ##     ## ##       ##     ## ##    ##   ## ##      ##     ##  ##     ## ###   ##
##       ##     ## ####  ## ##     ## ##       ##     ## ##        ##   ##     ##     ##  ##     ## ####  ##
##       ##     ## ## ## ## ##     ## ######   ########   ######  ##     ##    ##     ##  ##     ## ## ## ##
##       ##     ## ##  ####  ##   ##  ##       ##   ##         ## #########    ##     ##  ##     ## ##  ####
##    ## ##     ## ##   ###   ## ##   ##       ##    ##  ##    ## ##     ##    ##     ##  ##     ## ##   ###
 ######   #######  ##    ##    ###    ######## ##     ##  ######  ##     ##    ##    ####  #######  ##    ##

class Conversation:
    def __init__(self, system=None, filename=None, user='Blue', ai='Red'):
        self.max_tokens = 2000
        self.token_warning = 1000

        self.summarize = True
        self.summarized = False
        self.summarize_len = 500

        self.user_name = user
        self.ai_name = ai

        # Translate requests and responses
        self.translate = False
        self.user_lang = 'English'
        self.ai_lang = 'Chinese'

        if system:
            self.messages = [{"role": "system", "content": system}]
            self.history = []

        elif filename:
            if not self.load(filename):
                self.load(FILENAME)

        else:
            self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
            self.history = []

    def prefix(self, name):
        return f'{len(self.history)+1}) {name}: '
    
    def input(self):
        return remove_spaces(input(self.prefix(self.user_name)))
    
    def message_string(self, message):
        s = message['content']
        if message['role'] == 'system':
            return f'{s}'
        elif message['role'] == 'user':
            return f'{self.user_name}: {s}'
        elif message['role'] == 'assistant':
            return f'{self.ai_name}: {s}'
    
    def messages_string(self, messages, divider=' '):
        strings = []
        for m in messages:
            strings.append(self.message_string(m))
        return divider.join(strings)
    
    def print_messages(self, messages, first=None, last=None):
        if first == None: first = 0
        elif first < 0: first += len(messages)

        if last == None: last = len(messages)
        elif last < 0: last += len(messages)

        if last == first: last += 1 #if they're the same, bump last so we return 1
        for n, m in enumerate(messages[first:last], start=first):
            print_wrapped(f"{n}) {self.message_string(m)}")
            print('')

    def get_dialogue(self, user_input):
        if self.translate:
            user_input = get_translation(user_input, self.user_lang, self.ai_lang)

        message = {"role": "user", "content": user_input}
        self.messages.append(message)
        self.history.append(message)

        total_tokens = num_tokens_from_messages(self.messages)
        # Summarize or forget if approaching token limit
        if self.summarize:
            if DEBUG:
                print(f'<{total_tokens=}>')
            if total_tokens > self.max_tokens:
                self.summarize_messages(self.summarize_len, delete=True)

        else: # If self.summarize = False, just forget oldest messages
            while num_tokens_from_messages(self.messages) > self.max_tokens:
                last = self.messages[1]['content']
                if DEBUG: print(f'<Forgetting: {last}>')
                del self.messages[1]

        answer = chat_complete(self.messages, prefix=self.prefix(self.ai_name))

        message = { "role": "assistant", "content": answer }
        self.messages.append(message)
        self.history.append(message)

        if self.translate:
            answer = get_translation(answer, self.ai_lang, self.user_lang)

        if total_tokens > self.token_warning:
            print(f'<Warning: {total_tokens} tokens. Consider restarting conversation to save money>')

        return answer

    def summarize_messages(self, n=5000, delete=False):
        """Summarize the first n tokens worth of conversation. Default n > 4096 means summarize everything"""
        i = 1
        if num_tokens_from_messages(self.messages) > n:
            while num_tokens_from_messages(self.messages[:i]) < n:
                i += 1
            string = self.messages_string(self.messages[1:i])
            if delete:
                del self.messages[1:i]
        else:
            string = self.messages_string(self.messages[1:])
            if delete:
                del self.messages[1:]

        string.strip()
        if DEBUG:
            print(f'<Summarizing: {string}>')

        summary = get_summary(string)

        if DEBUG:
            print(f'<Summary: {summary}>')

        if delete: # if messages were deleted, update summary
            if self.summarized: # Already summarized? Replace old summary system message
                self.messages[1] = {"role": "system", "content": summary}
            else: # Add a new summary system message
                self.messages.insert(1, {"role": "system", "content": summary})
                self.summarized = True

        return summary

    def load(self, filename, output=True):

        data = load_file(filename, output=output)

        self.messages, self.history = [], []

        # We loaded a file, extract the data
        if isinstance(data, dict):
            for k, v in data.items():
                if k == 'variables':
                    for x, y in v.items(): # for each variable
                        if x not in []: # Don't restore these
                            setattr(self, x, y)
                elif k == 'messages':
                    self.messages = v
                elif k == 'history':
                    self.history = v
                elif k == 'date':
                    print(f"File date: {v}")
                else:
                    print('Error: Unknown key in loaded dictionary')
        elif isinstance(data, list): # List of messages format
            self.messages = data
        else:
            print('Error: Unknown Data save format')

        self.messages[0]['content'] = self.messages[0]['content'].format(USER_NAME=self.user_name, AI_NAME=self.ai_name)

        if self.history == []: # No history? Copy from messages
            self.history = [m for m in self.messages if m['role'] != 'system']

        else: # We have history. Connect to messages so they are in sync
            n = 1 # Count the number of identical messages
            while n < len(self.history):
                n += 1
                if self.history[-n] != self.messages[-n]:
                    n -= 1
                    break
            self.history[-n:] = self.messages[-n:] # Reassign so they are the same object        

        if self.messages[-1]['role'] == 'user': # If last response is user...
            self.messages.pop() # Remove it so user can respond
            self.history.pop() # History too

        if len(self.messages) > 1:
            if self.messages[1]['role'] == 'system': # Second messages is system...
                self.summarize = True
                self.summarized = True

            # Print last messages from loaded file
            print('')
            self.print_messages(self.messages, 0, 0)
            if output:
                if len(self.messages) > 2:
                    self.print_messages(self.messages, -2)

        return filename

    def save(self, filename, output=True):
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)

        # Class instance variables except messages and history (those are listed separate)
        variables = {k: v for k, v in vars(self).items() if k not in ['messages', 'history']}
        date = datetime.now()
        # .strftime('%Y-%m-%d %H:%M:%S')
        data = {'date': date, 'variables': variables, 'messages': self.messages, 'history': self.history}

        save_file(data, filename, output=output)

        return filename
        