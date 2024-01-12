from datetime import datetime
import os
import time
import re
import base64

import openai
import tiktoken
import google.generativeai as genai
from PIL import Image

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
    k:
        (a/250, b/250)
        if k.startswith('models/') # per character
        else (a/1000, b/1000) # per token
    for k, a, b in [
        # https://openai.com/pricing
        ('gpt-3.5-turbo', 0.001, 0.002),
        ('gpt-4', 0.03, 0.06),
        ('gpt-4-32k', 0.06, 0.12),
        ('gpt-4-1106-preview', 0.01, 0.03),
        ('gpt-4-vision-preview', 0.01, 0.03),
        ('gpt-4-turbo', 0.01, 0.03),
        # https://cloud.google.com/vertex-ai/docs/generative-ai/pricing
        ('models/chat-bison-001', 0.00025, 0.0005),
        ('models/chat-unicorn-001', 0.0025, 0.0075),
        ('models/gemini-pro', 0.00025, 0.0005),
        ('models/gemini-pro-vision', 0.00025, 0.0005),
        ('unknown', 0.001, 0.002)
    ]
}
# MODEL = "gpt-3.5-turbo" # Cheaper
# MODEL = "gpt-4" # Better
MODEL = "gpt-4-1106-preview" # Cheaper, Faster (?) GPT-4
# MODEL = 'models/chat-bison-001' # PaLM 2
STREAM = True
API_TRIES = 3

UNSAVED_VARIABLES = ['messages', 'history', 'summarized', 'translate', 'user_lang', 'ai_lang']

try:
    USER_ID = os.getlogin()
except Exception:
    USER_ID = 'unknown'

openai.api_key = os.environ["OPENAI_API_KEY"]

if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
else:
    genai.configure(api_key=os.environ["PALM_API_KEY"])




######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######

def num_tokens_gemini(messages, model=MODEL):
    """Returns the number of tokens used by a list of messages."""
    return sum(len(m['content']) // 4 for m in messages)
    # llm_model = genai.GenerativeModel(model)
    # return llm_model.count_tokens(gemini_messages(messages))

def num_tokens_palm(messages, model=MODEL):
    p_m = palm_messages(messages)
    if len(p_m) == 0:
        return 0
    return genai.count_message_tokens(
        model=model,
        context=messages[0]['content'],
        messages=p_m
    )['token_count']

def num_tokens_openai(messages, model=MODEL):
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

def chunks_from_gemini(response):
    for chunk in response:
        yield chunk.text if chunk.parts.pb else ''
        
def chunks_from_openai(response):
    for chunk in response:
        content = chunk.choices[0].delta.content
        yield content if content else ''

def gemini_messages(messages):
    new_roles = {
        'system': 'user',
        'assistant': 'model',
        'user': 'user'
    }

    formatted_messages = []
    for m in messages:
        parts = []
        img_match = re.match(r'^image: (\S*)\s*(.*)', m['content'])
        if img_match:
            url = img_match.group(1)
            if not url.startswith('http'): # It's a local file,
                img = Image.open(url)
                # llm_messages[-1]['parts'] = [img_match.group(2), img]
                parts = [img_match.group(2), img]
                # llm_messages = llm_messages[-1:] # Only one message supported by vision
        else:
            parts = [ m['content'] ]
        new = {
            'role': new_roles[ m['role'] ],
            'parts': parts
        }
        formatted_messages.append(new)
    # return formatted_messages
    return formatted_messages[-1:] # Multiturn (chat) not yet supported by Gemini Vision

def palm_messages(messages):
    return [
        {'author': m['role'], 'content': m['content']}
        for m in messages if m['role'] != 'system'
    ]

def openai_messages(messages):
    formatted_messages = []
    for m in messages:
        img_match = re.match(r'^image: (\S*)\s*(.*)', m['content'])
        if img_match:
            url = img_match.group(1)
            if not url.startswith('http'): # It's a local file,
                with open(url, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                url = f"data:image/jpeg;base64,{base64_image}"
            
            new = {
                'role': m['role'],
                'content': [
                    {"type": "text", "text": img_match.group(2)},
                    {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
                ]
            }
            formatted_messages.append(new)
        else:
            formatted_messages.append(m)
    return formatted_messages

def chat_gemini(messages, model, temperature, max_tokens, stream):
    # user_text = messages[-1]['content']
    llm_model = genai.GenerativeModel(model)
    config = genai.types.GenerationConfig(temperature=temperature)
    safety = [ # Dangerous mode
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        }
    ]
    if max_tokens:
        config.max_output_tokens = max_tokens
    llm_messages = gemini_messages(messages)

    with console.status('Getting response from Google...'):
        response = llm_model.generate_content(
            llm_messages,
            generation_config=config,
            safety_settings=safety,
            stream=stream
        )
    if stream:
        return console.print_stream(chunks_from_gemini(response)), True
    else:
        return response.text, False

def chat_palm(messages, model, temperature):
    with console.status('Getting response from Google...'):
        response = genai.chat(
            model=model,
            context=messages[0]['content'],
            messages=palm_messages(messages),
            temperature=temperature
        )
    return response.last, False

def chat_openai(messages, model, temperature, max_tokens, stream):
    llm_messages = openai_messages(messages)
    kwargs = {
        'model': model,
        'messages': llm_messages,
        'temperature': temperature,
        'user': USER_ID,
        'stream': stream
    }
    if max_tokens: kwargs['max_tokens'] = max_tokens
    with console.status('Getting response from OpenAI...'):
        # response = openai.ChatCompletion.create(**kwargs)
        response = openai.chat.completions.create(**kwargs) # New syntax
    if stream:
        return console.print_stream(chunks_from_openai(response)), True
    else:
        return response.choices[0].message.content, False

def chat_complete(messages, model=MODEL, temperature=0.8, max_tokens=None, print_result=True):
    for _ in range(API_TRIES): # 1, 2, ..., n
        try:
            stream = (STREAM and print_result)
            if model.startswith('models/gemini'): # Gemini
                response, printed = chat_gemini(messages, model, temperature, max_tokens, stream)
            elif model.startswith('models/'): # PaLM
                response, printed = chat_palm(messages, model, temperature)
            else: # OpenAI
                response, printed = chat_openai(messages, model, temperature, max_tokens, stream)

            if not printed:
                console.print_markdown(response)
            return response
        except Exception as e:
            console.print(f"Error: {e}\nTrying again in {1} second...\n")
            time.sleep(1)

    raise ConnectionError(f"Failed to access LLM API after {API_TRIES} attempts.")

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
        self.max_tokens = None

        self.max_chat_tokens = 8000
        self.token_warning = 4000
        self.cost = 0

        self.summarize = True
        self.summarized = False
        self.summarize_len = 4000

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
        if self.model.startswith('models/gemini'):
            return num_tokens_gemini(messages, self.model)
        elif self.model.startswith('models/'):
            return num_tokens_palm(messages, self.model)
        else:
            return num_tokens_openai(messages, self.model)

    def user_prefix(self, n=None):
        if n is None:
            n = len(self.history)+1
        return f'[user_label]{n} {self.user_name}:[/] '

    def ai_prefix(self, n=None):
        if n is None:
            n = len(self.history)+1
        labels = {
            'gpt-4': 'GPT-4',
            'gpt-4-32k': 'GPT-4 32k',
            'gpt-4-1106-preview': 'GPT-4 Turbo',
            'gpt-4-vision-preview': 'GPT-4 Vision',
            'gpt-3.5-turbo': 'GPT-3',
            'models/chat-bison-001': 'Bison',
            'models/chat-unicorn-001': 'Unicorn',
            'models/gemini-pro': 'Gemini',
            'models/gemini-pro-vision': 'Gemini Vision'
        }
        if self.model in labels:
            return f'[ai_label]{n} {self.ai_name}:[/] [od.red_dim]{labels[self.model]}[/] '
        else:
            return f'[ai_label]{n} {self.ai_name}:[/] [od.red_dim]{self.model}[/] '


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
            if m['role'] == 'system':
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

        answer = chat_complete(messages, model=self.model, max_tokens=self.max_tokens, print_result=print_result)
        answer_list = [{"role": "assistant", "content": answer}]

        if self.model in MODEL_COST:
            in_cost, out_cost = MODEL_COST[self.model]
        else:
            in_cost, out_cost = MODEL_COST['unknown'] # Guess cost for unknown models
        cost = self.__num_tokens()*in_cost + self.__num_tokens(answer_list)*out_cost # prompt cost + response cost
        self.cost = round(self.cost+cost, 5) # smallest unit is $0.01 / 1000

        return answer, cost

    def get_dialogue(self, user_input):
        t0 = time.time()

        if self.translate:
            user_input = get_translation(user_input, self.user_lang, self.ai_lang)

        total_tokens = self.__num_tokens()
        # Summarize or forget if approaching token limit
        summary_cost = 0
        if self.summarize and total_tokens > self.max_chat_tokens:
            with console.status('Consolidating memory...'):
                _, summary_cost = self.summarize_messages(self.summarize_len, delete=True)

        else: # If self.summarize = False, just forget oldest messages
            while self.__num_tokens() > self.max_chat_tokens:
                last = self.messages[1]['content']
                if DEBUG: console.log(f'<Forgetting: {last}>')
                del self.messages[1]

        message = {"role": "user", "content": user_input}
        self.messages.append(message)
        self.history.append(message)

        try:
            answer, message_cost = self.__complete(print_result=True, prefix=self.ai_prefix())
        except ConnectionError as e:
            console.print(str(e)+'\n')
            self.messages.pop()
            self.history.pop()
            return False

        message = { "role": "assistant", "content": answer }
        self.messages.append(message)
        self.history.append(message)

        if self.translate:
            answer = get_translation(answer, self.ai_lang, self.user_lang)

        total_time = time.time()-t0
        min, sec = divmod(total_time, 60)
        if min: time_string = f'[od.cyan_dim]{min:.0f}:{sec:05.2f}s[/]'
        else: time_string = f'[od.cyan_dim]{sec:05.2f}s[/]'

        cost_string = f'[od.dim][od.cyan_dim]{total_tokens}[/] tokens'
        if MONEY:
            fmt = lambda c: f'[od.green_dim]{c:.2f}[/]'
            summary_string = '+' + fmt(summary_cost) if summary_cost else ''
            cost_string += f' ({summary_string}+{fmt(message_cost)}={fmt(self.cost)})'

        if total_tokens > self.token_warning:
            console.print(f'{cost_string} {time_string} - Consider restarting conversation\n', justify='center')
        elif MONEY:
            console.print(f'{cost_string} {time_string}\n', justify='center')

        return True

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
        