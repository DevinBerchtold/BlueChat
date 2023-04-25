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
from rich.rule import Rule
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
    # background_color = '#0C0C0C'
    background_color = '#181818'
code_block_style = 'one-dark'

# one dark colors
od_white = '#ABB2BF'
od_red = '#EF596F'
od_blue = '#61AFEF'
od_yellow = '#E5C07B'
od_green = '#89CA78'
od_cyan = '#2BBAC5'
od_orange = '#D19A66'
od_dim = '#7E848E'
od_red_dim = '#AB4859'
od_blue_dim = '#4D82AE'
od_yellow_dim = '#A58D61'
od_green_dim = '#67945F'
od_cyan_dim = '#298992'
od_orange_dim = '#977453'

theme_dict = {
    # one dark colors
    "od.white": od_white,
    "od.red": od_red,
    "od.blue": od_blue,
    "od.yellow": od_yellow,
    "od.green": od_green,
    "od.cyan": od_cyan,
    "od.orange": od_orange,
    "od.dim": od_dim,
    "od.red_dim": od_red_dim,
    "od.blue_dim": od_blue_dim,
    "od.yellow_dim": od_yellow_dim,
    "od.green_dim": od_green_dim,
    "od.cyan_dim": od_cyan_dim,
    "od.orange_dim": od_orange_dim,
    # others
    "rule.line": od_yellow,
    "user_label": od_blue+' bold',
    "ai_label": od_red+' bold',
    "markdown.item.number": od_yellow+' bold',
    "markdown.item.bullet": od_yellow+' bold',
    "markdown.hr": od_yellow,
    "markdown.code": od_cyan,
    "markdown.link": od_cyan,
    "markdown.link_url": od_cyan,
    "repr.number": od_cyan,
    "repr.str": od_green,
    "repr.attrib_name": od_yellow,
    # "repr.attrib_equal": od_cyan,
    "repr.attrib_value": od_cyan,
    "repr.bool_true": od_green+' bold',
    "repr.bool_false": od_red+' bold',
    "repr.ipv4": od_cyan,
    "repr.ipv6": od_cyan,
    "repr.eui48": od_cyan,
    "repr.eui64": od_cyan,
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
    name = os.name
    if name == 'nt': # For Windows
        os.system('cls')
    elif name == 'posix': # For Linux and macOS
        os.system('clear')

def status_spinner(text='Loading...'):
    interval = 100.0 # interval = 100.0 for 'arc', 80.0 for 'dots'
    # 1000.0/refresh = interval*speed
    speed = 500.0/(REFRESH*interval) # About 2 refreshes per frame
    return Live(
        Align(
            Status('[od.dim]'+text, spinner='arc', spinner_style='markdown.hr', speed=speed),
            align='center'
        ),
        console=console, refresh_per_second=REFRESH, transient=True
    )

def generate_words(chunks):
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

def generate_paragraphs(chunks):
    """Generates words grouped such that each is a full word. '\\n\\n' outside code blocks are grouped to signify paragraph breaks."""
    code_block = False
    list_block = False
    for chunk in generate_words(chunks):
        toggle_code_block = len(chunk) >= 3 and '```' in chunk

        if chunk.startswith('\n\n'):
            list_match = re.match(r'\n\n(-|\*|[0-9]+(.|\)))', chunk)
            if not list_match:
                list_block = False
            
            if code_block or list_block:
                yield '\n' # No breaks...
                yield '\n' # ...just new lines
                yield chunk[2:]
            elif list_match: # first element of a list
                yield '\n\n' # Paragraph break
                yield chunk[2:]
            else:
                yield '\n\n\n' # Paragraph break + new line
                yield chunk[2:]

            if list_match: # switch after action so 
                list_block = True
        else:
            yield chunk

        if toggle_code_block:
            code_block = not code_block

def markdown(string):
    markdown = Markdown(string, code_theme=code_block_style, inline_code_lexer='python', inline_code_theme=InlineStyle)
    return Padding(markdown, (0, 1))

def print_markdown(string):
    console.print(markdown(string))

def print_rule(string):
    console.print(Padding(Rule(string), (0, 1)))

def print_stream(chunks):
    if FORMAT:
        live = Live(console=console, refresh_per_second=REFRESH, transient=True)
        live.start()
        full_paragraph = ''
        for w in generate_paragraphs(chunks):
            if w in ['\n\n', '\n\n\n']: # new paragraph
                full_paragraph += '\n\n'

                live.stop()
                print_markdown(full_paragraph)
                yield full_paragraph

                if w == '\n\n\n': # Paragraph break plus new line
                    # console.print('[od.dim]- - -[/]', justify='center')
                    console.print('')
                elif w == '\n\n': # This signals that we don't need the new line
                    # console.print('[od.red]_ _ _[/]', justify='center')
                    pass

                full_paragraph = ''
                live = Live(console=console, refresh_per_second=REFRESH, transient=True)
                live.start()
            else: # add to paragraph
                full_paragraph += w
                if not w.isspace(): # update if visible characters
                    live.update(markdown(full_paragraph))
        live.stop()
        print_markdown(full_paragraph)
        if full_paragraph:
            yield full_paragraph
    else:
        full_string = ''
        for c in chunks:
            sys.stdout.write(c)
            full_string += c
        sys.stdout.flush()
        yield full_string

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

def load_file(filename):
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
        if DEBUG: console.log(f"Loading {filename}.json")
        with open(f'{filename}.json', encoding='utf8') as f:
            return json.load(f)
        
    elif filetype == 'yaml':
        if DEBUG: console.log(f"Loading {filename}.yaml")
        with open(f'{filename}.yaml', 'r', encoding='utf8') as f:
            return yaml.safe_load(f)
        
    return False

def save_file(data, filename):
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
        if DEBUG:
            console.log("Data saved to "+file)
            
    if filetype == 'json' or (JSON and filetype is None):
        file = f'{folder}/{filename}.json'
        with open(file, 'w', encoding='utf8') as f:
            json.dump(data, f, indent='\t', ensure_ascii=False, default=str)
        if DEBUG:
            console.log("Data saved to "+file)
            