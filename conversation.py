import os
import sys
import json
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
PROMPT_FILE = 'help'

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

def chat_complete(openai_messages, temperature=0.8, prefix='', print_result=True):
    encoding = tiktoken.encoding_for_model(MODEL)
    for _ in range(5):
        try:
            t0 = time.time()
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=openai_messages,
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




 ######   #######  ##    ## ##     ## ######## ########   ######     ###    ######## ####  #######  ##    ##
##    ## ##     ## ###   ## ##     ## ##       ##     ## ##    ##   ## ##      ##     ##  ##     ## ###   ##
##       ##     ## ####  ## ##     ## ##       ##     ## ##        ##   ##     ##     ##  ##     ## ####  ##
##       ##     ## ## ## ## ##     ## ######   ########   ######  ##     ##    ##     ##  ##     ## ## ## ##
##       ##     ## ##  ####  ##   ##  ##       ##   ##         ## #########    ##     ##  ##     ## ##  ####
##    ## ##     ## ##   ###   ## ##   ##       ##    ##  ##    ## ##     ##    ##     ##  ##     ## ##   ###
 ######   #######  ##    ##    ###    ######## ##     ##  ######  ##     ##    ##    ####  #######  ##    ##

class Conversation:
    def __init__(self, system=None, filename=None):
        self.max_tokens = 500

        self.summarize = True
        self.summarized = False
        self.summarize_len = 200

        self.user_name = 'Blue'        
        self.ai_name = 'Red'        

        # Translate requests and responses
        self.translate = False
        self.user_lang = 'English'
        self.ai_lang = 'Chinese'

        if system:
            self.messages = [{"role": "system", "content": system}]
            self.history = []

        elif filename:
            self.load(filename)
        else:
            self.load(PROMPT_FILE)

    def prefix(self, name):
        return f'\n({len(self.history)+1}) {name}: '
    
    def input(self):
        return remove_spaces(input(self.prefix(self.user_name)))
    
    def messages_string(self, messages):
        string = ""
        for m in messages:
            s = m['content']
            if m['role'] == 'system':
                string += f' {s}'
            if m['role'] == 'user':
                string += f' {self.user_name}: {s}'
            if m['role'] == 'assistant':
                string += f' {self.ai_name}: {s}'
        return string.strip()

    def get_dialogue(self, user_input):
        if self.translate:
            user_input = self.get_translation(user_input, self.user_lang, self.ai_lang)

        message = {"role": "user", "content": user_input}
        self.messages.append(message)
        self.history.append(message)        

        # Summarize or forget if approaching token limit
        if self.summarize:
            total_tokens = num_tokens_from_messages(self.messages)
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
            answer = self.get_translation(answer, self.ai_lang, self.user_lang)

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

        summary = self.get_summary(string)

        if DEBUG:
            print(f'<Summary: {summary}>')

        if delete: # if messages were deleted, update summary
            if self.summarized: # Already summarized? Replace old summary system message
                self.messages[1] = {"role": "system", "content": summary}
            else: # Add a new summary system message
                self.messages.insert(1, {"role": "system", "content": summary})
                self.summarized = True

        return summary

    if YAML: # YAML copy for readability. Only define functions if needed
        def str_presenter(dumper, data): # Change style to | if multiple lines
            s = '|' if '\n' in data else None
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=s)

        def dict_presenter(dumper, data): # Only flow variables dictionary
            l = list(data.keys())
            f = l[0] != 'date' and l != ['role', 'content']
            return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=f)

        yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
        yaml.representer.SafeRepresenter.add_representer(dict, dict_presenter)

    def load(self, filename):
        self.messages, self.history = [], []
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
            # print(f"\n(0) {messages[0]['content']}") # Print system message
            if len(self.messages) > 2:
                l = len(self.history) # Print last question and answer
                print(f"\n({l-1}) {self.user_name}: {self.messages[-2]['content']}")
                print(f"\n({l}) {self.ai_name}: {self.messages[-1]['content']}")

        return True

    def save(self, filename):
        if not os.path.exists("conversations"):
            os.makedirs('conversations')

        # Class instance variables except messages and history (those are listed separate)
        variables = {k: v for k, v in vars(self).items() if k not in ['messages', 'history']}
        date = datetime.now()
        # .strftime('%Y-%m-%d %H:%M:%S')
        data = {'date': date, 'variables': variables, 'messages': self.messages, 'history': self.history}

        if YAML:
            yaml.safe_dump(data, open(f'conversations/{filename}.yaml', 'w', encoding='utf8'), allow_unicode=True, sort_keys=False, width=float("inf"))
        if JSON:
            json.dump(data, open(f'conversations/{filename}.json', 'w', encoding='utf8'), indent='\t', ensure_ascii=False, default=str)