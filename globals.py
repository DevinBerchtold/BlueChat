import sys
import os
import json
import shutil



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

WORD_WRAP = True
WRAP_WIDTH = 80




######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######

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

def print_wrapped(chunks, prefix=''):
    full_string = ''
    if WORD_WRAP:
        if isinstance(chunks, str):
            chunks = [chunks] # We expect a list of strings
        
        max_line_width = shutil.get_terminal_size().columns
        sys.stdout.write(prefix)
        sys.stdout.flush()
        current_line = prefix # Prefix minus last space
        for word in generate_words(chunks):
            full_string += word
            if word == '\n':
                sys.stdout.write('\n ')
                current_line = ' '
            elif word == ' ':
                if len(current_line) == max_line_width:
                    sys.stdout.write(' \n ')
                    current_line = ' '
                else:
                    sys.stdout.write(' ')
                    current_line += ' '
            elif len(current_line + word) > max_line_width-1:
                sys.stdout.write('\n '+word)
                sys.stdout.flush()
                current_line = ' '+word
            else: # word not line wrap
                sys.stdout.write(word)
                sys.stdout.flush()
                current_line += word
        sys.stdout.write('\n')
        sys.stdout.flush()
    else:
        for chunk in chunks:
            sys.stdout.write(chunk)
            full_string += chunk
        sys.stdout.write('\n')
        sys.stdout.flush()

    return full_string

if YAML: # Only define YAML functions if needed
    def str_presenter(dumper, data): # Change style to | if multiple lines
        s = '|' if '\n' in data else None
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=s)

    def dict_presenter(dumper, data): # Only flow variables dictionary
        l = list(data.keys())
        # All these != data will be shown on multiple lines in output
        f = l[0] != 'date' and l != ['role', 'content'] and l[0] != 'variables' and l[0] != 'facts'
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
        if output or DEBUG:
            print(f"Data saved to {filename}.yaml")
    if JSON:
        with open(f'{FOLDER}/{filename}.json', 'w', encoding='utf8') as f:
            json.dump(data, f, indent='\t', ensure_ascii=False, default=str)
        if output or DEBUG:
            print(f"Data saved to {filename}.json")
            