from datetime import datetime
import os
import time
import re
import base64
import json
from math import ceil
import random
from dataclasses import dataclass, field, asdict

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

MODEL_SHORTCUTS = {}
@dataclass(frozen=True)
class Model:
    id: str
    label: str
    costs: tuple
    context: int
    llm: str
    shortcuts: tuple = ()
    tools: bool = False
    vision: bool = False

    def __post_init__(self):
        for s in (*self.shortcuts, self.id):
            if s not in MODEL_SHORTCUTS: MODEL_SHORTCUTS[s] = self.id

MODEL_LIST = [ # cost: openai/1000, google*4/1000
    # https://openai.com/pricing
    Model(id='gpt-3.5-turbo', label='GPT-3.5', costs=(0.0000005, 0.0000015),
        context=16385, llm='openai', shortcuts=('3', '3.5', 'gpt-3'), tools=True),
    Model(id='gpt-4', label='GPT-4', costs=(0.00003, 0.00006),
        context=8192, llm='openai', tools=True),
    Model(id='gpt-4-32k', label='GPT-4 32k', costs=(0.00006, 0.00012),
        context=32768, llm='openai', shortcuts=('32', '32k')),
    Model(id='gpt-4-turbo-preview', label='GPT-4 Turbo', costs=(0.00001, 0.00003),
        context=128000, llm='openai', shortcuts=('4', 'gpt-4-turbo'), tools=True, vision='gpt-4-vision-preview'),
    Model(id='gpt-4-vision-preview', label='GPT-4 Vision', costs=(0.00001, 0.00003),
        context=128000, llm='openai', vision=True),
    # https://cloud.google.com/vertex-ai/docs/generative-ai/pricing
    Model(id='models/chat-bison-001', label='Bison', costs=(0.000001, 0.000002),
        context=4096, llm='palm', shortcuts=('b', 'palm', 'bison')),
    # Model(id='models/chat-unicorn-001', label='Unicorn', costs=(0.00004, 0.00003),
    #     context=4096, llm='palm', shortcuts=('u', 'unicorn')),
    Model(id='models/gemini-pro', label='Gemini', costs=(0.000001, 0.000002),
        context=32768, llm='gemini', shortcuts=('g', 'gemini'), tools=True, vision='models/gemini-pro-vision'),
    Model(id='models/gemini-pro-vision', label='Gemini Vision', costs=(0.000001, 0.000002),
        context=32768, llm='gemini', vision=True),
    Model(id='models/gemini-ultra', label='Gemini Ultra', costs=(0.000001, 0.000002),
        context=32768, llm='gemini', shortcuts=('u',), tools=True, vision='models/gemini-ultra-vision'),
]
MODELS = {m.id: m for m in MODEL_LIST}

MODEL = "gpt-4-turbo-preview" # Cheaper, Faster (?) GPT-4
USE_TOOLS = True
CONFIRM = True
STREAM = True
API_TRIES = 3

NOSAVE_VARS = ('messages', 'history', 'summarized', 'translate', 'user_lang', 'ai_lang')
NOPRINT_VARS = ('messages', 'history')

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

@dataclass
class Call:
    name: str = ''
    id: str = ''
    arguments: dict = field(default_factory=dict)

@dataclass
class Message:
    role: str
    content: str = ''
    tool_calls: list = field(default_factory=list)
    tool_call_id: str = ''
    name: str = ''




######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######

def num_tokens_gemini(messages, model=MODEL):
    """Returns the number of tokens used by a list of messages."""
    return sum(len(m.content) // 4 for m in messages)
    # llm_model = genai.GenerativeModel(model)
    # return llm_model.count_tokens(gemini_messages(messages))

def num_tokens_palm(messages, model=MODEL):
    p_m = palm_messages(messages)
    if len(p_m) == 0:
        return 0
    return genai.count_message_tokens(
        model=model,
        context=messages[0].content,
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
        if match := image_pattern.match(message.content):
            num_tokens += image_tokens_openai(int(match.group(2)), int(match.group(3)))
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in asdict(message).items():
            if key != 'tool_calls':
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def call_functions(messages, calls):
    for c in calls:
        for k, v in c.arguments.items():
            if '\n' not in v:
                if '\\\\n' in v:
                    c.arguments[k] = v.replace('\\\\n', '\n')
                elif '\\n' in v:
                    c.arguments[k] = v.replace('\\n', '\n')

    messages.append(Message(
        'assistant', tool_calls=calls
    ))

    for c in calls:
        console.print_function(c)
        
        name, args, id = c.name, c.arguments, c.id
        if name in functions.TOOLS:
            # arg_dict = json.loads(args)
            for k, v in args.items():
                args[k] = v

            if CONFIRM and functions.TOOLS[name].confirm:
                console.print('Press enter to confirm...')
                console.input()
            else:
                console.print('')
            
            response = functions.TOOLS[name].function(**args)
            
            messages.append(Message(
                'tool', response, tool_call_id=id, name=name
            ))
            console.print('')

def gemini_messages(messages):
    new_roles = {
        'tool': 'tool',
        'assistant': 'model',
        'user': 'user'
    }

    formatted_messages = []
    system_messages = []
    for m in messages:
        r = m.role
        c = m.content
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
                    name=m.name,
                    response={'result': c}
                ))
            ]
        elif m.tool_calls:
            parts = [
                glm.Part(function_call = glm.FunctionCall(
                    name=t.name,
                    args=t.arguments
                ))
                for t in m.tool_calls
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
        {'author': m.role, 'content': m.content}
        for m in messages if m.role != 'system'
    ]

def openai_messages(messages):
    formatted_messages = []
    for m in messages:
        if m.tool_calls:
            formatted_messages.append({
                'role': m.role,
                'tool_calls': [
                    {
                        'id': t.id,
                        'type': 'function',
                        'function': { 'name': t.name, 'arguments': json.dumps(t.arguments) }
                    }
                    for t in m.tool_calls
                ]
            })
        elif match := image_pattern.match(m.content):
            file, _, _, text = match.groups()
            if not file.startswith('http'): # local
                try:
                    with open(file, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        file = f"data:image/{file.split('.')[1]};base64,{base64_image}"
                except OSError as e:
                    console.print(e)
                    formatted_messages.append({k: v for k, v in asdict(m).items() if v})
                    continue
            new = {k: v for k, v in asdict(m).items() if v}
            new['content'] = [{"type": "image_url", "image_url": {"url": file}}]
            if text:
                new['content'].insert(0, {"type": "text", "text": text})
            formatted_messages.append(new)
        else:
            d = {k: v for k, v in asdict(m).items() if v}
            if m.role == 'tool' and m.content == '': d['content'] = ''
            formatted_messages.append(d)
    return formatted_messages

def stream_gemini(llm_model, response, messages, **kwargs):
    """Stream chunks from content messages while detecting and executing function calls."""
    alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    while True:
        func_calls = []
        for chunk in response:
            if chunk.parts[0].function_call.name:
                for n, tc in enumerate(chunk.parts):
                    func = tc.function_call
                    while len(func_calls) <= n: # Extend if necessary
                        func_calls.append(Call())
                    if func:                        
                        func_calls[n].name = func.name
                        for k, v in func.args.items():
                            if k in func_calls[n].arguments:
                                func_calls[n].arguments[k] += v
                            else:
                                func_calls[n].arguments[k] = v
                        if 'id' not in func_calls[n]:
                            func_calls[n].id = 'call_'+''.join(random.choices(alpha, k=4))
            else:
                yield chunk.text if chunk.parts.pb else ''
        
        if func_calls: # Call the functions and run model again
            call_functions(messages, func_calls)
            llm_messages = gemini_messages(messages)
            response = llm_model.generate_content(llm_messages, **kwargs)
        else:
            break

def chat_gemini(messages, model, temperature, max_tokens, stream, print_result, use_tools):
    llm_messages = gemini_messages(messages)
    if DEBUG: console.print(llm_messages)
    if model == 'models/gemini-pro-vision':
        llm_messages = [llm_messages[-1]] # Multiturn not yet supported by gemini-vision

    if use_tools: llm_model = genai.GenerativeModel(model,tools=functions.tools_gemini())
    else: llm_model = genai.GenerativeModel(model)

    config = genai.types.GenerationConfig()
    if temperature: config.temperature = temperature
    if max_tokens: config.max_output_tokens = max_tokens
    kwargs = {
        'generation_config': config,
        'safety_settings': GEMINI_SAFETY,
        'stream': stream
    }

    if print_result:
        with console.status('Connecting to Google...'):
            response = llm_model.generate_content(llm_messages, **kwargs)
    else:
        response = llm_model.generate_content(llm_messages, **kwargs)

    if use_tools or stream:
        return console.print_stream(stream_gemini(llm_model, response, messages, **kwargs)), True
    else:
        return response.text, False

def chat_palm(messages, model, temperature, print_result):
    llm_messages = palm_messages(messages)
    if DEBUG: console.print(llm_messages)
    with console.status('Getting response from Google...'):
        response = genai.chat(
            model=model,
            context=messages[0].content,
            messages=llm_messages,
            temperature=temperature
        )
    return response.last, False

def stream_openai(response, messages, **kwargs):
    """Stream chunks from content messages while detecting and executing function calls."""
    while True:
        func_calls = []
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for n, tc in enumerate(delta.tool_calls):
                    while len(func_calls) <= n: # Extend if necessary
                        func_calls.append(Call(arguments=''))
                    if tc.id: func_calls[n].id = tc.id
                    if tc.function:
                        if tc.function.name: func_calls[n].name = tc.function.name
                        if tc.function.arguments: func_calls[n].arguments += tc.function.arguments
            else:
                yield delta.content if delta.content else ''

        if func_calls: # Call the functions and run model again
            for f in func_calls:
                f.arguments = functions.get_args(f.name, f.arguments)
            call_functions(messages, func_calls)
            kwargs['messages'] = openai_messages(messages)
            response = openai.chat.completions.create(**kwargs)
        else:
            break

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
    if use_tools: kwargs['tools'] = functions.tools_openai()
    if print_result:
        with console.status('Connecting to OpenAI...'):
            response = openai.chat.completions.create(**kwargs) # New syntax
    else:
        response = openai.chat.completions.create(**kwargs) # New syntax
    
    if use_tools or stream:
        del kwargs['messages']
        return console.print_stream(stream_openai(response, messages, **kwargs)), True
    else:
        return response.choices[0].message.content, False

def chat_complete(messages, model=MODELS[MODEL], temperature=None, max_tokens=None, print_result=True, seed=None, use_tools=False):
    for n in range(API_TRIES): # 0, 1, 2, ..., n
        try:
            stream = (STREAM and print_result)
            if model.llm == 'gemini':
                response, printed = chat_gemini(messages, model.id, temperature, max_tokens, stream, print_result, use_tools)
            elif model.llm == 'palm':
                response, printed = chat_palm(messages, model.id, temperature, print_result)
            else: # OpenAI
                if model.id == 'gpt-4-vision-preview' and max_tokens is None:
                    max_tokens = 2000 # Override low default for vision
                response, printed = chat_openai(messages, model.id, temperature, max_tokens, stream, seed, print_result, use_tools)
            if not printed:
                console.print_markdown(response)
            return response
        except Exception as e:
            if n == 0: # Only fully print exception on first try
                console.print_exception(e, show_locals=True)
            else:
                console.print(e)
            
            if n < API_TRIES-1: # No last try
                console.print(f"\nTrying again...")
                time.sleep(1)

    raise ConnectionError(f"Failed to access LLM API after {API_TRIES} attempts.")

def get_complete(system, user_request, max_tokens=None, print_result=True, model=MODELS[MODEL]):
    complete_messages = [
        Message('system', system),
        Message('user', user_request)
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
        self.model = MODELS[model]
        self.use_tools = use_tools
        self.confirm = confirm
        self.user_name = user
        self.ai_name = ai
        self.seed = seed

        self.messages = [Message('system', 'You are a helpful assistant.')]
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
            self.messages = [Message('system', system)]
            self.history = []
        elif filename:
            if not self.load(filename):
                console.print("Error: Couldn't load file on reset()")

    def print_vars(self):
        variables = {k: v for k, v in vars(self).items() if k not in NOPRINT_VARS}
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
        if self.model.llm == 'gemini':
            return num_tokens_gemini(messages, self.model.id)
        elif self.model.llm == 'palm':
            return num_tokens_palm(messages, self.model.id)
        else:
            return num_tokens_openai(messages, self.model.id)

    def user_prefix(self, n=None):
        if n is None:
            n = len(self.history)+1
        return f'[user_label]{n} {self.user_name}:[/] '

    def ai_prefix(self, n=None):
        if n is None:
            n = len(self.history)+1
        return f'[ai_label]{n} {self.ai_name}:[/] [od.blue_dim]{self.model.label}[/] '

    def messages_string(self, messages, divider='\n'):
        strings = []
        for m in messages:
            r = m.role
            if r == 'system':
                s = m.content
            elif r == 'user':
                s = f'{self.user_name}: {m.content}'
            elif r == 'assistant':
                if m.content:
                    s = f'{self.ai_name}: {m.content}'
                elif m.tool_calls:
                    s = f'{self.ai_name}: '
                    for t in m.tool_calls:
                        s += t.name
                        for k, v in t.arguments.items():
                            s += f'\n{k}: {v}'
            elif r == 'tool':
                s = f"{m.name}: {m.content}".rstrip('\n')
            strings.append(s)
        return divider.join(strings)

    def print_messages(self, first=None, last=None):
        if first == None: first = 0
        if last == None: last = len(self.messages)
        g = [
            m for m in self.messages[:first+1]
            if m.role in ('user', 'assistant')
            and m.content
        ]
        n = len(g)
        f = False
        for m in self.messages[first:last]:
            p = None
            if m.role == 'system':
                p = '[bold]System:' if n == 0 else f'[bold]Summary {n}:'
            elif m.role == 'user':
                p = self.user_prefix(n)
            elif m.role == 'assistant':
                if m.tool_calls:
                    console.print(self.ai_prefix(n))
                    for c in m.tool_calls:
                        console.print_function(c)
                    n += 1
                else:
                    if f:
                        console.print_markdown(m.content)
                        f = False
                    else:
                        p = self.ai_prefix(n)
            elif m.role == 'tool':
                console.print_output(m.content)
                f = True

            if p:
                console.print(p)
                console.print_markdown(m.content)
                n += 1
            
            console.print('') # blank line

    def __complete(self, messages=None, system=None, user=None, print_result=False, prefix='', use_tools=False):
        if not messages:
            if system and user:
                messages = [
                    Message('system', system),
                    Message('user', user)
                ]
            else:
                messages=self.messages
        
        if prefix:
            console.print(prefix)

        answer = chat_complete(messages, model=self.model, max_tokens=self.max_tokens, print_result=print_result, seed=self.seed, use_tools=use_tools)
        answer_list = [Message('assistant', answer)]

        in_cost, out_cost = self.model.costs
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
                last = self.messages[1].content
                if DEBUG: console.log(f'<Forgetting: {last}>')
                del self.messages[1]

        message = Message('user', user_input)
        self.messages.append(message)
        self.history.append(message)

        try:
            answer, message_cost = self.__complete(print_result=True, prefix=self.ai_prefix(), use_tools=self.use_tools)
        except ConnectionError as e:
            console.print(f'\n{e}\n')
            self.messages.pop()
            self.history.pop()
            return False

        message = Message('assistant', answer)
        self.messages.append(message)
        self.history.append(message)

        if self.translate:
            answer = get_translation(answer, self.ai_lang, self.user_lang)

        total_time = time.time()-t0
        min, sec = divmod(total_time, 60)
        if min: time_string = f'[od.cyan_dim]{min:.0f}:{sec:05.2f}[/]'
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
                self.messages[1] = Message('system', summary)
            else: # Add a new summary system message
                self.messages.insert(1, Message('system', summary))
                self.summarized = True

        return summary, cost

    def load(self, filename):
        data = files.load_file(filename)
        if not data:
            console.print(f"Error: Couldn't load file {filename}")
            return False
        
        def to_messages(dicts):
            ret = []
            for d in dicts:
                m = Message(**d)
                if m.tool_calls:
                    m.tool_calls = [Call(**t) for t in m.tool_calls]
                ret.append(m)
            return ret

        self.messages, self.history = [], []

        # We loaded a file, extract the data
        if isinstance(data, dict):
            for k, v in data.items():
                if k == 'variables':
                    for x, y in v.items(): # for each variable
                        if x == 'model':
                            self.model = MODELS[y]
                        elif x not in NOSAVE_VARS: # Don't restore these
                            setattr(self, x, y)
                elif k == 'messages':
                    self.messages = to_messages(v)
                elif k == 'history':
                    self.history = to_messages(v)
                elif k == 'date':
                    pass
                else:
                    console.print('Error: Unknown key in loaded dictionary')
        elif isinstance(data, list): # List of messages format
            self.messages = to_messages(data)
        else:
            console.print('Error: Unknown Data save format')
        
        if DEBUG:
            self.print_vars()

        self.messages[0].content = self.messages[0].content.format(
            USER_NAME=self.user_name,
            AI_NAME=self.ai_name,
            TODAY=files.TODAY
        )

        if self.history == []: # No history? Copy from messages
            self.history = [m for m in self.messages if m.role != 'system']

        else: # We have history. Connect to messages so they are in sync
            i = len(self.messages)
            for h in reversed(self.history):
                while i > 0:
                    i -= 1
                    if h == self.messages[i]:
                        self.messages[i] = h
                        break

        if self.messages[-1].role == 'user': # If last response is user...
            self.messages.pop() # Remove it so user can respond
            self.history.pop() # History too

        if len(self.messages) > 1:
            if self.messages[1].role == 'system': # Second messages is system...
                self.summarize = True
                self.summarized = True

        return filename

    def save(self, filename):
        def to_dicts(messages):
            ret = []
            for m in messages:
                d = {k: v for k, v in asdict(m).items() if v}
                ret.append(d)
            return ret

        # Class instance variables except messages and history (those are listed separate)
        variables = {k: v for k, v in vars(self).items() if k not in NOSAVE_VARS}
        variables['model'] = self.model.id
        date = datetime.now()
        data = {'date': date, 'variables': variables,
            'messages': to_dicts(self.messages),
            'history': to_dicts(self.history)
        }

        files.save_file(data, filename)

        return filename
        