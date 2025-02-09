import io
import sys
import traceback
import json
import subprocess
from dataclasses import dataclass
from typing import Callable

import pyperclip

from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import wikipediaapi

wikipedia = wikipediaapi.Wikipedia('BlueChat (github.com/DevinBerchtold)', 'en')

import google.ai.generativelanguage as glm

import console




######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######

# Define a context manager to redirect stdout
class CaptureOutput:
    def __enter__(self):
        self._old_stdout = sys.stdout # Save the current stdout
        self._str_io = io.StringIO()
        sys.stdout = self._str_io
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.output = self._str_io.getvalue()
        sys.stdout = self._old_stdout
        self._str_io.close()

def execute(python_code, stream=True):
    filename = 'temp.py'
    with open(filename, 'w') as f:
        f.write(python_code)
    ret = ''
    if stream: # Stream from another process
        process = subprocess.Popen(['python', '-u', 'temp.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True: # Read the output line by line as it is produced
            output_line = process.stdout.readline()
            if not output_line and process.poll() is not None:
                break
            ret += output_line
            console.print(output_line, end='')

        # Check for and print any standard error output
        stderr_output = process.communicate()[1]
        if stderr_output:
            console.print(stderr_output)
            ret += stderr_output
    else: # Not streaming, just use exec()
        try:
            with CaptureOutput() as capt:
                compiled_code = compile(python_code, filename, 'exec')
                exec(compiled_code, {})
            ret = capt.output
            console.print_output(ret)
        except Exception as e: # Simulate traceback in return string
            exc_lines = traceback.format_exc().splitlines()
            exc_lines = [exc_lines[0]] + exc_lines[3:] # Remove the frame of this module
            ret = capt.output + '\n'.join(exc_lines)
            # ret = ret + '\n'.join(exc_lines)
            # console.print_output(capt.output)
            console.print_exception(e, suppress=['functions'], show_locals=True)
    return ret

driver = None

def get_page(url):
    global driver
    if not driver:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--log-level=3')
        # options.add_argument('--disable-gpu') # Sometimes needed if running on Windows
        options.add_experimental_option('excludeSwitches',['enable-logging'])
    
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    
    driver.get(url)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    page_text = soup.get_text(separator='\n', strip=True)

    # driver.quit()
    return page_text

def browse(url):
    if not url.startswith('http'):
        return None

    page_text = get_page(url)
    console.print_output(page_text)
    return page_text

def clipboard(*args):
    user_input = str(pyperclip.paste())
    console.print_output(user_input)
    return user_input

def wiki(search):
    w = wikipedia.page(search)
    text = f'{w.title}\n\n{w.text}'
    console.print(text)
    return text

def get_args(name, string):
    args = {}
    try:
        args = json.loads(string)
    except:
        args = {TOOLS[name].parameters[0]['name']: string}
    return args




########  ######## ######## #### ##    ## #### ######## ####  #######  ##    ##  ######
##     ## ##       ##        ##  ###   ##  ##     ##     ##  ##     ## ###   ## ##    ##
##     ## ##       ##        ##  ####  ##  ##     ##     ##  ##     ## ####  ## ##
##     ## ######   ######    ##  ## ## ##  ##     ##     ##  ##     ## ## ## ##  ######
##     ## ##       ##        ##  ##  ####  ##     ##     ##  ##     ## ##  ####       ##
##     ## ##       ##        ##  ##   ###  ##     ##     ##  ##     ## ##   ### ##    ##
########  ######## ##       #### ##    ## ####    ##    ####  #######  ##    ##  ######

@dataclass
class Tool:
    name: str
    description: str
    icon: str
    function: Callable[..., str]
    parameters: tuple = ()
    confirm: bool = False
    enabled: bool = True

    def __repr__(self):
        return f"Tool(name='{self.name}')"

TOOL_LIST = [
    Tool(
        name='python', function=execute, confirm=True, enabled=True, icon='🐍',
        description="Runs the Python source code with exec() and returns the standard output that was printed with print(). Code can and use packages as necessary",
        parameters=[{
            'name': 'python_code',
            'description': "The python source code to be executed",
            'type': str,
            'required': True
        }]
    ),
    Tool(
        name='browse', function=browse, confirm=False, enabled=True, icon='🔗',
        description="Browses to the given url and extracts the text from the website using BeautifulSoup",
        parameters=[{
            'name': 'url',
            'description': "The url of the website to be retrieved",
            'type': str,
            'required': True
        }]
    ),
    Tool(
        name='wikipedia', function=wiki, confirm=False, enabled=True, icon='🌐',
        description="Searches Wikipedia and returns the full text of most relevant page.",
        parameters=[{
            'name': 'search',
            'description': "A string to search on Wikipedia. The most relevant page will be returned.",
            'type': str,
            'required': True
        }]
    ),
    Tool(
        name='clipboard', function=clipboard, confirm=True, enabled=False, icon='📋',
        description="Lets the user copy something to the clipboard and sends it back to you",
    ),
]
TOOLS = {t.name: t for t in TOOL_LIST}

def tools_gemini():
    return [
        glm.Tool(
            function_declarations=[
                glm.FunctionDeclaration(
                    name=n,
                    description=f.description,
                    parameters=glm.Schema(
                        type=glm.Type.OBJECT,
                        properties={
                            p['name']: glm.Schema(type=glm.Type.STRING)
                            for p in f.parameters
                        },
                        required=[
                            p['name']
                            for p in f.parameters
                            if p['required']
                        ]
                    )
                )
                if f.parameters else
                glm.FunctionDeclaration(
                    name=n,
                    description=f.description
                )
                for n, f in TOOLS.items() if f.enabled
            ]
        )
    ]

def tools_openai():
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        p['name']: {"type": "string", "description": p['description']}
                        for p in f.parameters
                    },
                    "required": [
                        p['name']
                        for p in f.parameters
                        if p['required']
                    ]
                }
                if f.parameters else {}
            }
        }
        for n, f in TOOLS.items() if f.enabled
    ]

def tools_anthropic():
    return [
        {
            "name": n,
            "description": f.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    p['name']: {"type": "string", "description": p['description']}
                    for p in f.parameters
                },
                "required": [
                    p['name']
                    for p in f.parameters
                    if p['required']
                ]
            }
            if f.parameters else {}
        }
        for n, f in TOOLS.items() if f.enabled
    ]