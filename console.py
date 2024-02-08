import os
import sys
import re

RICH = True
if RICH:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.padding import Padding
    from rich.theme import Theme
    from rich.status import Status
    from rich.align import Align
    from rich.rule import Rule
    from rich.columns import Columns
    from pygments.styles.onedark import OneDarkStyle




 ######   #######  ##    ##  ######  ########    ###    ##    ## ########  ######
##    ## ##     ## ###   ## ##    ##    ##      ## ##   ###   ##    ##    ##    ##
##       ##     ## ####  ## ##          ##     ##   ##  ####  ##    ##    ##
##       ##     ## ## ## ##  ######     ##    ##     ## ## ## ##    ##     ######
##       ##     ## ##  ####       ##    ##    ######### ##  ####    ##          ##
##    ## ##     ## ##   ### ##    ##    ##    ##     ## ##   ###    ##    ##    ##
 ######   #######  ##    ##  ######     ##    ##     ## ##    ##    ##     ######

DEBUG = False

REFRESH = 8
# Paragraphs not in the Live object that have not been returned yet
paragraphs = None
resets = 0

if RICH:
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
        "user_label": od_red+' bold',
        "ai_label": od_blue+' bold',
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

    rich_console = Console(theme=Theme(theme_dict))




######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######

std_print = print
std_input = input

def remove_tags(s):
    return re.sub(r'\[.*?\]', '', repr(s))

def print(*args, **kwargs):
    if RICH:
        return rich_console.print(*args, **kwargs)
    else:
        exclude = ['justify'] # ignore these parms
        print_kwargs = {k: v for k, v in kwargs.items() if k not in exclude}
        return std_print(*(remove_tags(a) for a in args), **print_kwargs)

def log(s, **kwargs):
    if RICH:
        return rich_console.log(s, **kwargs)
    else:
        return std_print(s, **kwargs)

def input(s=None, **kwargs):
    if RICH:
        return rich_console.input(s, **kwargs)
    else:
        return std_input(remove_tags(s), **kwargs)

class BasicStatus:
    def __init__(self, string):
        self.string = string

    def __enter__(self):
        std_print(self.string, end='')

    def __exit__(self, exc_type, exc_val, exc_tb):
        std_print('\r'+(' '*len(self.string)), end='\r')

def status(text='Loading...'):
    if RICH:
        interval = 100.0 # interval = 100.0 for 'arc', 80.0 for 'dots'
        # 1000.0/refresh = interval*speed
        speed = 500.0/(REFRESH*interval) # About 2 refreshes per frame
        return Live(
            Align(
                Status('[od.dim]'+text, spinner='arc', spinner_style='markdown.hr', speed=speed),
                align='center'
            ),
            console=rich_console, refresh_per_second=REFRESH, transient=True
        )
    else:
        return(BasicStatus(text))

def clear():
    name = os.name
    if name == 'nt': # For Windows
        os.system('cls')
    elif name == 'posix': # For Linux and macOS
        os.system('clear')

def reset(chat, filename, print_start):
    global resets
    resets += 1
    clear()
    print_filename(filename)
    chat.print_messages(print_start, None)
    if paragraphs is not None:
        print(chat.ai_prefix())
        if paragraphs != '':
            print_markdown(paragraphs)
            print('')
    else:
        print('\r'+chat.user_prefix(), end='')

def generate_words(chunks):
    """Generates words where each element starts with whitespace and ends with a word"""
    buffer = ''
    for chunk in chunks:
        for char in chunk:
            if char.isspace():
                if buffer.isspace():
                    buffer += char
                else: # Buffer has data and we're beginning new chunk
                    if buffer:
                        yield buffer
                    buffer = char
            else:
                buffer += char
    if buffer:
        yield buffer

def generate_paragraphs(chunks):
    """Print and yield complete paragraphs, one by one. Keep code blocks and lists as one paragraph"""
    code_block = False
    list_block = False
    live = Live(console=rich_console, refresh_per_second=REFRESH)
    live.start()
    buffer = ''
    for chunk in generate_words(chunks):
        if chunk.startswith('\n\n'):
            list_match = re.match(r'\n\n(-|\*|[0-9]+(.|\)))', chunk)
            if code_block or (list_block and list_match):
                buffer += chunk
            else: 
                live.stop()
                yield buffer
                buffer = chunk[2:]
                if list_match: # first element of a list (list_match = True, list_block = False)
                    # console.print('[od.dim]_ _ _[/]', justify='center')
                    pass
                else: # (list_match = False, list_block = ?)
                    # console.print('[od.dim]- - -[/]', justify='center')
                    rich_console.print('')
                live = Live(console=rich_console, refresh_per_second=REFRESH)
                live.start()
            list_block = list_match
        else:
            buffer += chunk
        if len(chunk) >= 3 and '```' in chunk:
            code_block = not code_block
        if not chunk.isspace(): # update if visible characters
                live.update(markdown(buffer))
    live.stop()
    if buffer:
        yield buffer

def markdown(string):
    markdown = Markdown(string, code_theme=code_block_style, inline_code_lexer='python', inline_code_theme=InlineStyle)
    return Padding(markdown, (0, 1))

def print_markdown(string):
    if RICH:
        rich_console.print(markdown(string))
    else:
        std_print(string)

def print_rule(string):
    if RICH:
        rich_console.print(Padding(Rule(string), (0, 1)))
    else:
        std_print(f'==== {string} ====')

def print_filename(filename):
    folder, file = filename.split('/')
    match file.capitalize().split('_'):
        case [f, d, n]:
            s = f'[od.dim]{folder} /[/] [bold]{f}[/] [od.dim]-[/] [od.white]{d}[/] [od.dim]-[/] [bold]{n}'
        case [f, n]:
            s = f'[od.dim]{folder} /[/] [bold]{f}[/] [od.dim]-[/] [bold]{n}'
        case [f]:
            s = f'[od.dim]{folder} /[/] [bold]{f}'
    print_rule(s)

def print_function(func):
    print_markdown(f"**{func['name']}:**")
    for k, v in func['arguments'].items():
        if k == 'python_code':
            print_markdown(f'```python\n{v}\n```')
        else:
            print_markdown(f'*{k}: {v}*')

def print_output(c):
    lines = c.splitlines()
    if not c:
        c = '*No output*'
        print_markdown(c)
    elif len(lines) > 30 or len(c) > 1000:
        c = f'*{len(c)/1024.:.2f} kB, {len(lines)} lines*'
        print_markdown(c)
    else:
        for l in lines:
            print(l)

def print_exception(e, **kwargs):
    if RICH:
        rich_console.print_exception(
            theme = code_block_style,
            width = os.get_terminal_size().columns,
            **kwargs
        )
    else:
        std_print(e)

def plen(string): # print length
    return len(remove_tags(string))

def split_string_list(strings, w):
    total_chars = float(sum(plen(s) for s in strings))
    total_chars += w*(len(strings)-1)*1.5 # estimate blank lines
    half_chars = total_chars / 2.0

    first_part = [strings[0]]
    second_part = []
    running_sum = float(plen(strings[0]))
    for s in strings[1:]:
        l = (w*1.5)+plen(s)
        if running_sum+(l*0.65) < half_chars:
            first_part.append(s)
            running_sum += l
        else:
            second_part.append(s)

    return first_part, second_part

def print_columns(cells):
    w = os.get_terminal_size().columns/2.0
    l, r = split_string_list(cells, w-3.0)
    cols = Columns(('\n\n'.join(l), '\n\n'.join(r)), padding=(0, 3), width=int(w-6.0))
    print(cols, justify='center')

def print_stream(chunks):
    global paragraphs
    if RICH:
        paragraphs = ''
        for i, p in enumerate(generate_paragraphs(chunks)):
            if i == 0:
                paragraphs += p
            else:
                paragraphs += '\n\n'+p
        paragraphs, result = None, paragraphs
        return result # Return and then reset paragraphs
    else:
        full_string = ''
        for c in chunks:
            sys.stdout.write(c)
            sys.stdout.flush()
            full_string += c
        sys.stdout.write('\n')
        sys.stdout.flush()
        return full_string
