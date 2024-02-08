from datetime import datetime
import os
import time
import re
import base64
import json
from math import ceil
import random

import openai
import tiktoken
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image

import files
import console
import functions



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
    k: (a/250, b/250)
    if k.startswith('models/') # per character
    else (a/1000, b/1000) # per token
    for k, a, b in [
        # https://openai.com/pricing
        ('gpt-3.5-turbo', 0.0005, 0.0015),
        ('gpt-4', 0.03, 0.06),
        ('gpt-4-32k', 0.06, 0.12),
        ('gpt-4-turbo-preview', 0.01, 0.03),
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
MODEL_LABELS = {
    'gpt-4': 'GPT-4',
    'gpt-4-32k': 'GPT-4 32k',
    'gpt-4-turbo-preview': 'GPT-4 Turbo',
    'gpt-4-vision-preview': 'GPT-4 Vision',
    'gpt-3.5-turbo': 'GPT-3',
    'models/chat-bison-001': 'Bison',
    'models/chat-unicorn-001': 'Unicorn',
    'models/gemini-pro': 'Gemini',
    'models/gemini-pro-vision': 'Gemini Vision'
}
# MODEL = "gpt-3.5-turbo" # Cheaper
# MODEL = "gpt-4" # Better
MODEL = "gpt-4-turbo-preview" # Cheaper, Faster (?) GPT-4
# MODEL = 'models/chat-bison-001' # PaLM 2
USE_TOOLS = True
CONFIRM = True
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

GEMINI_SAFETY = [
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




######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######

def num_tokens_gemini(messages, model=MODEL):
    """Returns the number of tokens used by a list of messages."""
    return sum(len(m['content']) // 4 for m in messages if 'content' in m)
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

def image_tokens_openai(w, h, detail='auto'):
    if detail == 'low' or not w or not h:
        return 85
    if max(w, h) > 2048: # Fit within a 2048 x 2048 square
        r = 2048.0 / max(w, h)
        w, h = w * r, h * r
    if min(w, h) > 768: # Max of 768px on the shortest side
        r = 768.0 / min(w, h)
        w, h = w * r, h * r
    return 85 + ( 170 * ceil(w / 512) * ceil(h / 512) )

image_pattern = re.compile(r'^image: (\S+) ?\(?(\d*)?x?(\d*)?\)?\s*(.*)')

def num_tokens_openai(messages, model=MODEL):
    """Returns the number of tokens used by a list of messages. Copied from OpenAI example"""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        if ('content' in message) and (match := image_pattern.match(message['content'])):
            num_tokens += image_tokens_openai(int(match.group(2)), int(match.group(3)))
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            if key != 'tool_calls':
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def call_functions(messages, calls):
    for c in calls:
        for k, v in c['arguments'].items():
            if '\n' not in v:
                if '\\\\n' in v:
                    c['arguments'][k] = v.replace('\\\\n', '\n')
                elif '\\n' in v:
                    c['arguments'][k] = v.replace('\\n', '\n')

    messages.append({
        "role": "assistant",
        "tool_calls": calls
    })

    for c in calls:
        console.print_function(c)
        
        name, args, id = c['name'], c['arguments'], c['id']
        if name in functions.TOOLS:
            # arg_dict = json.loads(args)
            for k, v in args.items():
                args[k] = v

            if CONFIRM and functions.TOOLS[name]['confirm']:
                console.input('Press enter to confirm...')
            else:
                console.print('')
            
            response = functions.TOOLS[name]['function'](**args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": id,
                "name": name,
                "content": response,
            })
            console.print('')

def chunks_from_gemini(response):
    for chunk in response:
        yield chunk.text if chunk.parts.pb else ''

def chunks_from_openai(response):
    for chunk in response:
        content = chunk.choices[0].delta.content
        yield content if content else ''

def gemini_messages(messages):
    new_roles = {
        'tool': 'tool',
        'assistant': 'model',
        'user': 'user'
    }

    formatted_messages = []
    system_messages = []
    for m in messages:
        r = m['role']
        c = m['content'] if 'content' in m else ''
        parts = [c]
        if r == 'system':
            system_messages.append(c)
            continue
        elif r == 'user':
            parts = ['\n\n'.join(system_messages + [c])]
            system_messages = []
        elif r == 'tool':
            parts = [
                glm.Part(function_response = glm.FunctionResponse(
                    name=m['name'],
                    response={'result': c}
                ))
            ]
        elif 'tool_calls' in m:
            parts = [
                glm.Part(function_call = glm.FunctionCall(
                    name=t['name'],
                    args=t['arguments']
                ))
                for t in m['tool_calls']
            ]

        if match := image_pattern.match(c):
            file, _, _, text = match.groups()
            try: # It's a local file,
                img = Image.open(file)
                if text:
                    parts = [text, img]
                else:
                    parts = [img]

            except OSError as e:
                console.print(e)
                parts = [c]
        
        if r in new_roles:
            formatted_messages.append({
                'role': new_roles[r],
                'parts': parts
            })
    return formatted_messages

def palm_messages(messages):
    return [
        {'author': m['role'], 'content': m['content']}
        for m in messages if m['role'] != 'system'
    ]

def openai_messages(messages):
    formatted_messages = []
    for m in messages:
        if 'tool_calls' in m:
            formatted_messages.append({
                'role': m['role'],
                'tool_calls': [
                    {
                        'id': t['id'],
                        'type': 'function',
                        'function': { 'name': t['name'], 'arguments': json.dumps(t['arguments']) }
                    }
                    for t in m['tool_calls']
                ]
            })
        elif match := image_pattern.match(m['content']):
            file, _, _, text = match.groups()
            if not file.startswith('http'): # local
                try:
                    with open(file, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        file = f"data:image/{file.split('.')[1]};base64,{base64_image}"
                except OSError as e:
                    console.print(e)
                    formatted_messages.append(m)
                    continue
            new = m.copy()
            new['content'] = [{"type": "image_url", "image_url": {"url": file}}]
            if text:
                new['content'].insert(0, {"type": "text", "text": text})
            formatted_messages.append(new)
        else:
            formatted_messages.append(m)
    return formatted_messages

def chat_gemini(messages, model, temperature, max_tokens, stream, print_result, use_tools):
    llm_messages = gemini_messages(messages)
    if DEBUG: console.print(llm_messages)
    if model == 'models/gemini-pro-vision':
        llm_messages = [llm_messages[-1]] # Multiturn not yet supported by gemini-vision

    if use_tools: llm_model = genai.GenerativeModel(model,tools=functions.tools_gemini())
    else: llm_model = genai.GenerativeModel(model)

    config = genai.types.GenerationConfig()
    if temperature: config.max_output_tokens = temperature
    if max_tokens: config.max_output_tokens = max_tokens
    kwargs = {
        'generation_config': config,
        'safety_settings': GEMINI_SAFETY,
        'stream': (stream and not use_tools)
    }

    if print_result:
        with console.status('Getting response from Google...'):
            response = llm_model.generate_content(llm_messages, **kwargs)
    else:
        response = llm_model.generate_content(llm_messages, **kwargs)

    if use_tools:
        tool_calls = response.candidates[0].content.parts
        # if hasattr(response.parts[0], 'function_call'):
        if response.parts[0].function_call.name:
        # while parts[0].function_call.name:
            alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            call_functions(messages, [
                {
                    'id': 'call_'+''.join(random.choices(alpha, k=4)),
                    'name': t.function_call.name,
                    'arguments': dict(t.function_call.args)
                }
                for t in tool_calls
            ])

            llm_messages = gemini_messages(messages)
            if DEBUG: console.print(llm_messages)
            kwargs['stream'] = stream # False?
            response = llm_model.generate_content(llm_messages, **kwargs)
        else:
            return response.text, False
    
    if stream:
        return console.print_stream(chunks_from_gemini(response)), True
    else:
        return response.text, False

def chat_palm(messages, model, temperature, print_result):
    llm_messages = palm_messages(messages)
    if DEBUG: console.print(llm_messages)
    with console.status('Getting response from Google...'):
        response = genai.chat(
            model=model,
            context=messages[0]['content'],
            messages=llm_messages,
            temperature=temperature
        )
    return response.last, False

def chat_openai(messages, model, temperature, max_tokens, stream, seed, print_result, use_tools):
    llm_messages = openai_messages(messages)
    if DEBUG: console.print(llm_messages)
    kwargs = {
        'model': model,
        'messages': llm_messages,
        'temperature': temperature,
        'user': USER_ID,
        'stream': stream,
        'seed': seed
    }
    if max_tokens: kwargs['max_tokens'] = max_tokens
    # if temperature: kwargs['temperature'] = temperature
    if use_tools:
        kwargs['tools'] = functions.tools_openai()
        kwargs['stream'] = False
    if print_result:
        with console.status('Getting response from OpenAI...'):
            # response = openai.ChatCompletion.create(**kwargs)
            response = openai.chat.completions.create(**kwargs) # New syntax
    else:
        response = openai.chat.completions.create(**kwargs) # New syntax

    if use_tools:
        if tool_calls := response.choices[0].message.tool_calls:
            call_functions(messages, [
                {
                    'id': t.id,
                    'name': t.function.name,
                    'arguments': functions.get_args(t.function.name, t.function.arguments)
                }
                for t in tool_calls
            ])

            kwargs['stream'] = stream
            kwargs['messages'] = openai_messages(messages)
            if DEBUG: console.print(kwargs['messages'])
            response = openai.chat.completions.create(**kwargs)
        else:
            return response.choices[0].message.content, False
    if stream:
        return console.print_stream(chunks_from_openai(response)), True
    else:
        return response.choices[0].message.content, False

def chat_complete(messages, model=MODEL, temperature=None, max_tokens=None, print_result=True, seed=None, use_tools=False):
    for n in range(API_TRIES): # 0, 1, 2, ..., n
        try:
            stream = (STREAM and print_result)
            if model.startswith('models/gemini'): # Gemini
                response, printed = chat_gemini(messages, model, temperature, max_tokens, stream, print_result, use_tools)
            elif model.startswith('models/'): # PaLM
                response, printed = chat_palm(messages, model, temperature, print_result)
            else: # OpenAI
                if model == 'gpt-4-vision-preview' and max_tokens is None:
                    max_tokens = 2000 # Override low default for vision
                response, printed = chat_openai(messages, model, temperature, max_tokens, stream, seed, print_result, use_tools)
            if not printed:
                console.print_markdown(response)
            return response
        except Exception as e:
            if n == 0: # Only fully print exception on first try
                console.print_exception(e)
            else:
                console.print(e)
            
            if n < API_TRIES-1: # No last try
                console.print(f"\nTrying again...")
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
    def __init__(self, system=None, filename=None, user='Blue', ai='Red', model=MODEL, seed=None, use_tools=USE_TOOLS, confirm=True):
        self.model = model
        self.use_tools = use_tools
        self.confirm = confirm
        self.user_name = user
        self.ai_name = ai
        self.seed = seed

        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.history = []

        self.reset(filename=filename, system=system)

    def reset(self, filename=None, system=None, ai=None):
        self.max_tokens = None

        self.max_chat_tokens = 8000
        self.token_warning = 4000
        self.total_tokens = 0
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
        if self.model in MODEL_LABELS:
            return f'[ai_label]{n} {self.ai_name}:[/] [od.blue_dim]{MODEL_LABELS[self.model]}[/] '
        else:
            return f'[ai_label]{n} {self.ai_name}:[/] [od.blue_dim]{self.model}[/] '

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

    def print_messages(self, first=None, last=None):
        if first == None: first = 0
        if last == None: last = len(self.messages)
        g = [
            m for m in self.messages[:first+1]
            if m['role'] in ('user', 'assistant')
            and 'content' in m
        ]
        n = len(g)
        f = False
        for m in self.messages[first:last]:
            p = None
            if m['role'] == 'system':
                p = '[bold]System:' if n == 0 else f'[bold]Summary {n}:'
            elif m['role'] == 'user':
                p = self.user_prefix(n)
            elif m['role'] == 'assistant':
                if 'tool_calls' in m:
                    console.print(self.ai_prefix(n))
                    for c in m['tool_calls']:
                        console.print_function(c)
                    n += 1
                else:
                    if f:
                        console.print_markdown(m['content'])
                        f = False
                    else:
                        p = self.ai_prefix(n)
            elif m['role'] == 'tool':
                console.print_output(m['content'])
                f = True

            if p:
                console.print(p)
                console.print_markdown(m['content'])
                n += 1
            
            console.print('') # blank line

    def __complete(self, messages=None, system=None, user=None, print_result=False, prefix='', use_tools=False):
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

        answer = chat_complete(messages, model=self.model, max_tokens=self.max_tokens, print_result=print_result, seed=self.seed, use_tools=use_tools)
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
            answer, message_cost = self.__complete(print_result=True, prefix=self.ai_prefix(), use_tools=self.use_tools)
        except ConnectionError as e:
            console.print(f'\n{e}\n')
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
        if min: time_string = f'[od.cyan_dim]{min:.0f}:{sec:.2f}s[/]'
        else: time_string = f'[od.cyan_dim]{sec:.2f}s[/]'

        self.total_tokens = self.__num_tokens()
        cost_string = f'[od.dim][od.cyan_dim]{self.total_tokens}[/] tokens'
        if MONEY:
            fmt = lambda c: f'[od.green_dim]{c:.2f}[/]'
            summary_string = '+' + fmt(summary_cost) if summary_cost else ''
            cost_string += f' ({summary_string}+{fmt(message_cost)}={fmt(self.cost)})'

        if self.total_tokens > self.token_warning:
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
        