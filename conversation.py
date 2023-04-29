from datetime import datetime
import os
import time

import openai
import tiktoken

import files
import console



 ######   #######  ##    ##  ######  ########    ###    ##    ## ########  ######
##    ## ##     ## ###   ## ##    ##    ##      ## ##   ###   ##    ##    ##    ##
##       ##     ## ####  ## ##          ##     ##   ##  ####  ##    ##    ##
##       ##     ## ## ## ##  ######     ##    ##     ## ## ## ##    ##     ######
##       ##     ## ##  ####       ##    ##    ######### ##  ####    ##          ##
##    ## ##     ## ##   ### ##    ##    ##    ##     ## ##   ###    ##    ##    ##
 ######   #######  ##    ##  ######     ##    ##     ## ##    ##    ##     ######

DEBUG = False

MONEY = True # Print money conversion in token warning
MODEL_COST = {
    'gpt-3.5-turbo': 0.002 / 1000,  # $0.002 / 1K tokens 4/6/23
    'gpt-4': 0.03 / 1000            # $0.03 / 1K tokens 4/6/23
}
# MODEL = "gpt-3.5-turbo" # Cheaper
MODEL = "gpt-4" # Better
STREAM = True
API_TRIES = 3

UNSAVED_VARIABLES = ['messages', 'history', 'summarized', 'translate', 'user_lang', 'ai_lang']

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

def num_tokens_from_messages(messages, model=MODEL):
    """Returns the number of tokens used by a list of messages. Copied from OpenAI example"""
    encoding = tiktoken.encoding_for_model(model)
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

def chat_complete(openai_messages, model=MODEL, temperature=0.8, max_tokens=None, print_result=True):
    for _ in range(API_TRIES): # 1, 2, ..., n
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                user=USER_ID,
                stream=(STREAM and print_result)
            )
            if STREAM and print_result:
                return console.print_stream(chunks_from_response(response))
            else:
                full_string = response['choices'][0]['message']['content']
                if print_result:
                    console.print('\n'+full_string)
                return full_string
        except Exception as e:
            console.print(f"Error: {e}\nTrying again in {1} second...\n")
            time.sleep(1)

    raise ConnectionError("Failed to access OpenAI API after {API_TRIES} attempts.")

def get_complete(system, user_request, max_tokens=None, print_result=True, model=MODEL):
    complete_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_request }
    ]
    answer = ''
    for p in chat_complete(complete_messages, max_tokens=max_tokens, print_result=print_result, model=model):
        answer += p
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
    def __init__(self, system=None, filename=None, user='Blue', ai='Red', model=MODEL):
        self.model = model
        self.user_name = user
        self.ai_name = ai

        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.history = []

        self.reset(filename=filename, system=system)

    def reset(self, filename=None, system=None, ai=None):
        self.max_tokens = 1000
        self.token_warning = 500
        self.cost = 0

        self.summarize = True
        self.summarized = False
        self.summarize_len = 400

        # Translate requests and responses
        self.translate = False
        self.user_lang = 'English'
        self.ai_lang = 'Chinese'

        if ai:
            self.ai_name = ai

        if system:
            self.messages = [{"role": "system", "content": system}]
            self.history = []
        elif filename:
            if not self.load(filename):
                console.print("Error: Couldn't load file on reset()")

    def print_vars(self):
        variables = {k: v for k, v in vars(self).items() if k not in UNSAVED_VARIABLES}
        console.print(variables)
    
    def input(self, prefix=True):
        if prefix:
            i = console.input(self.user_prefix())
        else:
            i = console.input()
        console.print('')
        return i
    
    def __num_tokens(self, messages=None):
        if messages == None:
            messages = self.messages
        return num_tokens_from_messages(messages, self.model)

    def user_prefix(self, n=None):
        if n is None:
            n = len(self.history)+1
        return f'[user_label]{n} {self.user_name}:[/] '

    def ai_prefix(self, n=None):
        if n is None:
            n = len(self.history)+1
        
        model = 'GPT-4' if self.model == 'gpt-4' else 'GPT-3.5'
        return f'[ai_label]{n} {self.ai_name}:[/] [od.red_dim]{model}[/] '

    def messages_string(self, messages, divider=' '):
        strings = []
        for m in messages:
            c = m['content']
            r = m['role']
            if r == 'system':
                s = f'{c}'
            elif r == 'user':
                s = f'{self.user_name}: {c}'
            elif r == 'assistant':
                s = f'{self.ai_name}: {c}'
            strings.append(s)
        return divider.join(strings)
    
    def __print_message(self, n, m):
        match m['role']:
            case 'system':
                p = '[bold]System:' if n == 0 else f'[bold]Summary {n}:'
            case 'user':
                p = self.user_prefix(n)
            case 'assistant':
                p = self.ai_prefix(n)
        console.print(p)
        console.print_markdown(m['content'])
        console.print('') # blank line

    def print_systems(self):
        for n, m in enumerate(self.messages):
            self.__print_message(n, m)
    
    def print_messages(self, first=None, last=None):
        l = len(self.history)

        if first == None: first = 0
        elif first < 0: first += l

        if last == None: last = l
        elif last < 0: last += l

        if last == first: last += 1 #if they're the same, bump last so we return 1
        for n, m in enumerate(self.history[first:last], start=first+1):
            self.__print_message(n, m)

    def __complete(self, messages=None, system=None, user=None, print_result=False, prefix=''):
        if not messages:
            if system and user:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            else:
                messages=self.messages
        
        if prefix:
            console.print(prefix)

        answer = chat_complete(messages, max_tokens=self.max_tokens, model=self.model, print_result=print_result)
        answer_list = [{"role": "assistant", "content": answer}]
        cost = self.__num_tokens() + self.__num_tokens(answer_list)*2 # prompt cost + 2 x response cost
        cost *= MODEL_COST[self.model]
        self.cost = round(self.cost+cost, 5) # smallest unit is $0.01 / 1000

        return answer, cost

    def get_dialogue(self, user_input):
        t0 = time.time()

        if self.translate:
            user_input = get_translation(user_input, self.user_lang, self.ai_lang)

        total_tokens = self.__num_tokens()
        # Summarize or forget if approaching token limit
        summary_cost = 0
        if self.summarize and total_tokens > self.max_tokens:
            with console.status('Consolidating memory...'):
                _, summary_cost = self.summarize_messages(self.summarize_len, delete=True)

        else: # If self.summarize = False, just forget oldest messages
            while self.__num_tokens() > self.max_tokens:
                last = self.messages[1]['content']
                if DEBUG: console.log(f'<Forgetting: {last}>')
                del self.messages[1]

        message = {"role": "user", "content": user_input}
        self.messages.append(message)
        self.history.append(message)

        answer, message_cost = self.__complete(print_result=True, prefix=self.ai_prefix())

        message = { "role": "assistant", "content": answer }
        self.messages.append(message)
        self.history.append(message)

        if self.translate:
            answer = get_translation(answer, self.ai_lang, self.user_lang)

        total_time = time.time()-t0
        if DEBUG: console.log(f'chat tokens={total_tokens}, time={total_time:.3}s')

        cost_string = f'{total_tokens} tokens'
        if MONEY:
            if summary_cost:
                cost_string += f' (+{summary_cost:.2f}+{message_cost:.2f}={self.cost:.2f})'
            else:
                cost_string += f' (+{message_cost:.2f}={self.cost:.2f})'

        if total_tokens > self.token_warning:
            console.print(f'[od.green_dim]{cost_string} [od.dim]Consider restarting conversation[/]', justify='center')
        elif MONEY:
            console.print(f'[od.green_dim]{cost_string}[/]', justify='center')

    def summarize_messages(self, n=5000, delete=False):
        """Summarize the first n tokens worth of conversation. Default n > 4096 means summarize everything"""
        i = 1
        if self.__num_tokens() > n:
            while self.__num_tokens(self.messages[:i]) < n:
                i += 1
        else:
            i = None
        string = self.messages_string(self.messages[1:i])
        if delete:
            del self.messages[1:i]

        if DEBUG: console.log(f'Summarizing: {string}')

        summary, cost = self.__complete(system='Summarize this conversation', user=string)

        if DEBUG: console.log(f'Summary (+{cost:.2f}={self.cost:.2f}): {summary}')

        if delete: # if messages were deleted, update summary
            if self.summarized: # Already summarized? Replace old summary system message
                self.messages[1] = {"role": "system", "content": summary}
            else: # Add a new summary system message
                self.messages.insert(1, {"role": "system", "content": summary})
                self.summarized = True

        return summary, cost

    def load(self, filename):
        data = files.load_file(filename)
        if not data:
            console.print(f"Error: Couldn't load file {filename}")
            return False

        self.messages, self.history = [], []

        # We loaded a file, extract the data
        if isinstance(data, dict):
            for k, v in data.items():
                if k == 'variables':
                    for x, y in v.items(): # for each variable
                        if x not in UNSAVED_VARIABLES: # Don't restore these
                            setattr(self, x, y)
                elif k == 'messages':
                    self.messages = v
                elif k == 'history':
                    self.history = v
                elif k == 'date':
                    pass
                else:
                    console.print('Error: Unknown key in loaded dictionary')
        elif isinstance(data, list): # List of messages format
            self.messages = data
        else:
            console.print('Error: Unknown Data save format')
        
        if DEBUG:
            self.print_vars()

        self.messages[0]['content'] = self.messages[0]['content'].format(
            USER_NAME=self.user_name,
            AI_NAME=self.ai_name,
            TODAY=files.TODAY
        )

        if self.history == []: # No history? Copy from messages
            self.history = [m for m in self.messages if m['role'] != 'system']

        else: # We have history. Connect to messages so they are in sync
            n = 1 # While messages are identical...
            while n < len(self.history) and self.history[-n] == self.messages[-n]:
                self.history[-n] = self.messages[-n] # ...reassign so they are the same object
                n += 1

        if self.messages[-1]['role'] == 'user': # If last response is user...
            self.messages.pop() # Remove it so user can respond
            self.history.pop() # History too

        if len(self.messages) > 1:
            if self.messages[1]['role'] == 'system': # Second messages is system...
                self.summarize = True
                self.summarized = True

        return filename

    def save(self, filename):
        # Class instance variables except messages and history (those are listed separate)
        variables = {k: v for k, v in vars(self).items() if k not in UNSAVED_VARIABLES}
        date = datetime.now()
        data = {'date': date, 'variables': variables, 'messages': self.messages, 'history': self.history}

        files.save_file(data, filename)

        return filename
        