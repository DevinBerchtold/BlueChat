import sys
import os
import json
from datetime import datetime
import re

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.theme import Theme
from rich.status import Status
from rich.align import Align
from pygments.styles.onedark import OneDarkStyle




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

REFRESH = 8

class InlineStyle(OneDarkStyle):
    background_color = '#0C0C0C'
code_block_style = 'one-dark'

# one dark colors
od_red = '#EF596F bold'
od_blue = '#61AFEF bold'
od_yellow = '#E5C07B bold'
od_green = '#89CA78 bold'
od_cyan = '#2BBAC5 bold'
od_orange = '#D19A66 bold'

theme_dict = {
    "user_label": od_blue,
    "ai_label": od_red,
    "markdown.item.number": od_yellow,
    "markdown.item.bullet": od_yellow,
    "markdown.hr": od_yellow,
    "markdown.code": od_cyan,
    "markdown.link": od_cyan,
    "markdown.link_url": od_cyan,
    "repr.number": od_cyan,
    "repr.string": od_green,
    "repr.attrib_name": od_yellow,
    # "repr.attrib_equal": od_cyan,
    "repr.attrib_value": od_cyan,
    "repr.bool_true": od_green,
    "repr.bool_false": od_red,
}

console = Console(theme=Theme(theme_dict))

FORMAT = True

TODAY = datetime.now().strftime('%Y-%m-%d')




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

def status_spinner(text='Loading...'):
    interval = 100.0 # interval = 100.0 for 'arc', 80.0 for 'dots'
    # 1000.0/refresh = interval*speed
    speed = 500.0/(REFRESH*interval) # About 2 refreshes per frame
    return Live(
        Align(
            Status('[dim]'+text, spinner='arc', spinner_style='markdown.hr', speed=speed),
            align='center'
        ),
        console=console, refresh_per_second=REFRESH, transient=True
    )

def break_spaces(chunks):
    buffer = ''
    for chunk in chunks:
        for char in chunk:
            if char.isspace():
                if buffer.isspace():
                    buffer += char
                else: # Buffer has data and we're begging new chunk
                    if buffer:
                        yield buffer
                    buffer = char
            else:
                buffer += char
    if buffer:
        yield buffer

def generate_words(chunks):
    """Generates words grouped such that each is a full word. '\\n\\n' outside code blocks are grouped to signify paragraph breaks."""
    code_block = False
    for chunk in break_spaces(chunks):
        toggle_code_block = len(chunk) >= 3 and '```' in chunk

        if chunk.startswith('\n\n'):
            if code_block or re.match(r'\n\n(-|\*|[0-9]+(.|\)))', chunk):
                yield '\n'
                yield '\n'
                yield chunk[2:]
            else:
                yield '\n\n'
                yield chunk[2:]
        else:
            yield chunk

        if toggle_code_block:
            code_block = not code_block

def markdown(string):
    markdown = Markdown(string, code_theme=code_block_style, inline_code_lexer='python', inline_code_theme=InlineStyle)
    return Padding(markdown, (0, 1))

def print_markdown(string):
    console.print(markdown(string))

def print_stream(chunks):
    full_string = ''
    if FORMAT:
        live = None
        full_paragraph = ''
        for w in generate_words(chunks):
            full_string += w
            full_paragraph += w
            if not live:
                live = Live(console=console, refresh_per_second=REFRESH, transient=True)
                live.start()

            if w == '\n\n': # new paragraph
                live.stop()
                print_markdown(full_paragraph)

                # console.print('[dim]- - - -[/]', justify='center')
                console.print('')

                full_paragraph = ''
                live = Live(console=console, refresh_per_second=REFRESH, transient=True)
                live.start()
                
            else:
                if not w.isspace():
                    live.update(markdown(full_paragraph))
        live.stop()
        print_markdown(full_paragraph)
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
        l = list(data)
        # All these != data will be shown on multiple lines in output
        f = (l[0] != 'date') and (l != ['role', 'content']) and (l[0] != 'variables') and (l[0] not in ['fact', 'help'])
        return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=f)

    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(dict, dict_presenter)

def load_file(filename, output=False):
    if '.' in filename: # if filetype specified
        filename, filetype = filename.split('.')
    else:
        if os.path.isfile(f'{filename}.json'):
            filetype = 'json'
        elif YAML:
            if os.path.isfile(f'{filename}.yaml'):
                filetype = 'yaml'
            else:
                return False
        else:
            return False
    
    if filetype == 'json':
        if output: print(f"Loading {filename}.json")
        with open(f'{filename}.json', encoding='utf8') as f:
            data = json.load(f)
            return data
    elif filetype == 'yaml':
        if output: print(f"Loading {filename}.yaml")
        with open(f'{filename}.yaml', 'r', encoding='utf8') as f:
            data = yaml.safe_load(f)
            return data
    return False

def save_file(data, filename, output=False):
    if '/' in filename: # make folder if it's not there
        folder, filename = filename.split('/')
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    if '.' in filename: # if filetype specified
        filename, filetype = filename.split('.')
    else:
        filetype = None

    if filetype == 'yaml' or (YAML and filetype is None):
        file = f'{folder}/{filename}.yaml'
        with open(file, 'w', encoding='utf8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=float("inf"))
        if output:
            console.print("Data saved to "+file)
        elif DEBUG:
            console.log("Data saved to "+file)
            
    if filetype == 'json' or (JSON and filetype is None):
        file = f'{folder}/{filename}.json'
        with open(file, 'w', encoding='utf8') as f:
            json.dump(data, f, indent='\t', ensure_ascii=False, default=str)
        if output:
            console.print("Data saved to "+file)
        elif DEBUG:
            console.log("Data saved to "+file)
            