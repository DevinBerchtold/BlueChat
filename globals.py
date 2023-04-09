import sys
import os
import json
import shutil

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding

console = Console()




 ######   #######  ##    ##  ######  ########    ###    ##    ## ########  ######
##    ## ##     ## ###   ## ##    ##    ##      ## ##   ###   ##    ##    ##    ##
##       ##     ## ####  ## ##          ##     ##   ##  ####  ##    ##    ##
##       ##     ## ## ## ##  ######     ##    ##     ## ## ## ##    ##     ######
##       ##     ## ##  ####       ##    ##    ######### ##  ####    ##          ##
##    ## ##     ## ##   ### ##    ##    ##    ##     ## ##   ###    ##    ##    ##
 ######   #######  ##    ##  ######     ##    ##     ## ##    ##    ##     ######

DEBUG = False

YAML = True
if YAML:
    import yaml
JSON = False

FILENAME = 'help'
FOLDER = 'conversations'

FORMAT = True




######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######

def clear_screen():
    if os.name == 'nt':   # For Windows
        os.system('cls')
    elif os.name == 'posix':  # For Linux and macOS
        os.system('clear')

def generate_words(chunks):
    """Generates words grouped such that each is a full word, a space, or a newline"""
    buffer = ''
    for chunk in chunks:
        for char in chunk:
            if char in [' ', '\n']:
                if buffer:
                    yield buffer
                buffer = ''
                yield char
            else:
                buffer += char
    if buffer: # Yield the last word after the loop
        yield buffer

def print_markdown(markdown):
    console.print(Padding(Markdown(markdown), (0, 1)))

def print_stream(chunks):
    full_string = ''
    if FORMAT:
        live = Live(console=console, refresh_per_second=16)
        with live:
            # for word in generate_words(chunks):
            for w in generate_words(chunks):
                full_string += w
                if not w.isspace():
                    markdown = Padding(Markdown(full_string), (0, 1))
                    live.update(markdown)
            markdown = Padding(Markdown(full_string), (0, 1))
            live.update(markdown)
    else:
        for c in chunks:
            sys.stdout.write(c)
            full_string += c
        sys.stdout.flush()

    return full_string

if YAML: # Only define YAML functions if needed
    def str_presenter(dumper, data): # Change style to | if multiple lines
        s = '|' if '\n' in data else None
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=s)

    def dict_presenter(dumper, data): # Only flow variables dictionary
        l = list(data.keys())
        # All these != data will be shown on multiple lines in output
        f = (l[0] != 'date') and (l != ['role', 'content']) and (l[0] != 'variables') and (l[0] not in ['fact', 'help'])
        return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=f)

    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(dict, dict_presenter)

def load_file(filename, output=False):
    split = filename.split('.')
    if len(split) == 2: # if filetype specified
        filename = split[0]
        filetype = split[1]
    else:
        if os.path.isfile(f'conversations/{filename}.json'):
            filetype = 'json'
        elif YAML:
            if os.path.isfile(f'conversations/{filename}.yaml'):
                filetype = 'yaml'
            else:
                return False
        else:
            return False
    
    if filetype == 'json':
        if output: print(f"Loading {filename}.json")
        with open(f'conversations/{filename}.json', encoding='utf8') as f:
            data = json.load(f)
            return data
    elif filetype == 'yaml':
        if output: print(f"Loading {filename}.yaml")
        with open(f'conversations/{filename}.yaml', 'r', encoding='utf8') as f:
            data = yaml.safe_load(f)
            return data
    return False

def save_file(data, filename, output=False):
    if YAML:
        with open(f'{FOLDER}/{filename}.yaml', 'w', encoding='utf8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=float("inf"))
        if output:
            console.print(f"Data saved to {filename}.yaml")
        elif DEBUG:
            console.log(f"Data saved to {filename}.yaml")
    if JSON:
        with open(f'{FOLDER}/{filename}.json', 'w', encoding='utf8') as f:
            json.dump(data, f, indent='\t', ensure_ascii=False, default=str)
        if output:
            console.print(f"Data saved to {filename}.json")
        elif DEBUG:
            console.log(f"Data saved to {filename}.json")
            