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

MONEY = True # Print money conversion in token warning
MODEL_COST = {
    'gpt-3.5-turbo': 0.002 / 1000,  # $0.002 / 1K tokens 4/6/23
    'gpt-4': 0.03 / 1000            # $0.03 / 1K tokens 4/6/23
}
# MODEL = "gpt-3.5-turbo" # Cheaper
MODEL = "gpt-4" # Better
STREAM = True

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
    for _ in range(5):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                user=USER_ID,
                stream=(STREAM and print_result)
            )
            full_string = ''
            if STREAM and print_result:
                full_string = print_stream(chunks_from_response(response))
            else:
                full_string = response['choices'][0]['message']['content']
                if print_result:
                    console.print('\n'+full_string)

            return full_string

        except Exception as e:
            console.print(f"Error: {e}\nTrying again in 1 second...\n")
            time.sleep(1)
    raise ConnectionError("Failed to access OpenAI API after 5 attempts.")

def get_complete(system, user_request, max_tokens=None, print_result=True, model=MODEL):
    complete_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_request }
    ]
    answer = chat_complete(complete_messages, max_tokens=max_tokens, print_result=print_result, model=model)
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

    def reset(self, filename=None, system=None):
        self.max_tokens = 800
        self.token_warning = 400
        self.cost = 0

        self.summarize = True
        self.summarized = False
        self.summarize_len = 400

        # Translate requests and responses
        self.translate = False
        self.user_lang = 'English'
        self.ai_lang = 'Chinese'

        if system:
            self.messages = [{"role": "system", "content": system}]
            self.history = []
        elif filename:
            if not self.load(filename, output=True):
                self.load(FILENAME, output=True)
    
    def input(self):
        return console.input(self.user_prefix())
    
    def num_tokens(self, messages=None):
        if messages == None:
            messages = self.messages
        return num_tokens_from_messages(messages, self.model)
    
    def prefix(self, n, name, color):
        if not n:
            n = len(self.history)+1
        return f'\n[bold {color}]{n} {name}:[/] '

    def user_prefix(self, n=None):
        return self.prefix(n, self.user_name, 'blue')

    def ai_prefix(self, n=None):
        return self.prefix(n, self.ai_name, 'red')         
    
    def message_prefix(self, message, n):
        if message['role'] == 'system':
            if n == 0:
                return f'\n[bold]System:[/] '
            else:
                return f'\n[bold]Summary {n}:[/] '
        elif message['role'] == 'user':
            return self.user_prefix(n)
        elif message['role'] == 'assistant':
            return self.ai_prefix(n)

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
    
    def print_messages(self, messages, first=None, last=None):
        if first == None: first = 0
        elif first < 0: first += len(messages)

        if last == None: last = len(messages)
        elif last < 0: last += len(messages)

        if last == first: last += 1 #if they're the same, bump last so we return 1
        for n, m in enumerate(messages[first:last], start=first):
            console.print(self.message_prefix(m, n))
            print_markdown(m['content'])
            # console.print(Padding(Markdown(), (0, 1)))

    def complete(self, messages=None, system=None, user=None, print_result=False, prefix=''):
        if not messages:
            if system and user:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            else:
                messages=self.messages
        
        console.print(prefix)
        answer = chat_complete(messages, max_tokens=self.max_tokens, model=self.model, print_result=print_result)
        answer_list = [{"role": "assistant", "content": answer}]
        cost = self.num_tokens() + self.num_tokens(answer_list)*2 # prompt cost + 2 x response cost
        cost *= MODEL_COST[self.model]
        self.cost = round(self.cost+cost, 5) # smallest unit is $0.01 / 1000

        return answer, cost

    def get_dialogue(self, user_input):
        t0 = time.time()

        if self.translate:
            user_input = get_translation(user_input, self.user_lang, self.ai_lang)

        message = {"role": "user", "content": user_input}
        self.messages.append(message)
        self.history.append(message)

        total_tokens = self.num_tokens()
        # Summarize or forget if approaching token limit
        summary_cost = 0
        if self.summarize:
            if total_tokens > self.max_tokens:
                _, summary_cost = self.summarize_messages(self.summarize_len, delete=True)

        else: # If self.summarize = False, just forget oldest messages
            while self.num_tokens() > self.max_tokens:
                last = self.messages[1]['content']
                if DEBUG: console.log(f'<Forgetting: {last}>')
                del self.messages[1]

        answer, message_cost = self.complete(print_result=True, prefix=self.ai_prefix())

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
            print_markdown(f'#### {cost_string} Consider restarting conversation to save money')
            # console.print(f'[grey30]<{cost_string} Consider restarting conversation to save money>[/]')
        elif MONEY:
            print_markdown(f'#### {cost_string}')
            # console.print(f'[grey30]<{cost_string}>[/]')

        return answer

    def summarize_messages(self, n=5000, delete=False):
        """Summarize the first n tokens worth of conversation. Default n > 4096 means summarize everything"""
        i = 1
        if self.num_tokens() > n:
            while self.num_tokens(self.messages[:i]) < n:
                i += 1
            string = self.messages_string(self.messages[1:i])
            if delete:
                del self.messages[1:i]
        else:
            string = self.messages_string(self.messages[1:])
            if delete:
                del self.messages[1:]

        if DEBUG: console.log(f'Summarizing: {string}')

        summary, cost = self.complete(system='Summarize this conversation', user=string)

        if DEBUG: console.log(f'Summary (+{cost:.2f}={self.cost:.2f}): {summary}')

        if delete: # if messages were deleted, update summary
            if self.summarized: # Already summarized? Replace old summary system message
                self.messages[1] = {"role": "system", "content": summary}
            else: # Add a new summary system message
                self.messages.insert(1, {"role": "system", "content": summary})
                self.summarized = True

        return summary, cost

    def load(self, filename, output=True):

        data = load_file(filename, output=False)
        if output:
            # print_markdown('----')
            print_markdown(f'----\n### {filename.capitalize()}')

        self.messages, self.history = [], []

        variables = []
        # We loaded a file, extract the data
        if isinstance(data, dict):
            for k, v in data.items():
                if k == 'variables':
                    for x, y in v.items(): # for each variable
                        if x not in UNSAVED_VARIABLES: # Don't restore these
                            setattr(self, x, y)
                            variables += [f'{x}: {y}']
                elif k == 'messages':
                    self.messages = v
                elif k == 'history':
                    self.history = v
                elif k == 'date':
                    variables.append(f"timestamp: {v}")
                else:
                    console.print('Error: Unknown key in loaded dictionary')
        elif isinstance(data, list): # List of messages format
            self.messages = data
        else:
            console.print('Error: Unknown Data save format')
        if output and variables:
            console.print(' ' + ', '.join(variables))

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
            if output:
                self.print_messages(self.messages, 0, 0)
                if len(self.messages) > 2:
                    self.print_messages(self.messages, -2)

        return filename

    def save(self, filename, output=True):
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)

        # Class instance variables except messages and history (those are listed separate)
        variables = {k: v for k, v in vars(self).items() if k not in UNSAVED_VARIABLES}
        date = datetime.now()
        data = {'date': date, 'variables': variables, 'messages': self.messages, 'history': self.history}

        save_file(data, filename, output=output)

        return filename
        