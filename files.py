import os
import json
from datetime import datetime

import console




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

TODAY = datetime.now().strftime('%Y-%m-%d')




######## ##     ## ##    ##  ######  ######## ####  #######  ##    ##  ######
##       ##     ## ###   ## ##    ##    ##     ##  ##     ## ###   ## ##    ##
##       ##     ## ####  ## ##          ##     ##  ##     ## ####  ## ##
######   ##     ## ## ## ## ##          ##     ##  ##     ## ## ## ##  ######
##       ##     ## ##  #### ##          ##     ##  ##     ## ##  ####       ##
##       ##     ## ##   ### ##    ##    ##     ##  ##     ## ##   ### ##    ##
##        #######  ##    ##  ######     ##    ####  #######  ##    ##  ######

if YAML: # Only define YAML functions if needed
    def str_presenter(dumper, data): # Change style to | if multiple lines
        s = '|' if '\n' in data else None
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=s)

    def dict_presenter(dumper, data): # Only flow variables dictionary
        l, s = list(data), set(data)
        # All these will be shown on multiple lines in output
        f = (l[0] not in ('date', 'role', 'variables', 'blue', 'python_code')
            and s != {'id', 'name', 'arguments'})
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
            