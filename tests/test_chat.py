import pytest
import conftest
from chat import *

@pytest.mark.parametrize('input,expected', [
    ("I need some help", 'help'),
    ("Square root of 67", 'math'),
    ("How much should I feed my puppy?", 'dog'),
    ("Define werk", 'dictionary'),
    ("List comprehension example", 'code'),
    ("Words like comprehensive", 'thesaurus'),
    ("Joe Biden and Donald Trump", 'politics'),
    ("Tell me about Detroit, MI", 'fact'),
    ("Help me brainstorm ideas for a poem", 'creative'),
])
def test_switch_context(input, expected):
    print(f'  {input=}')
    print(f'{expected=}')
    result = switch_context(input)
    print(f'  {result=}')
    assert result == expected

@pytest.mark.parametrize('input,filename,date,number', [
    ("help", 'help', None, 0),
    ("help_2023-04-19_1", 'help', '2023-04-19', 1),
    ("help_2023-04-19_10", 'help', '2023-04-19', 10),
    ("conversations/help_2023-04-19_10", 'help', '2023-04-19', 10),
])
def test_filename_vars(input, filename, date, number):
    f, d, n = filename_vars(input)
    assert f == filename
    if date:
        assert d == date
    assert n == number

if __name__ == '__main__':
    pytest.main(['tests/test_chat.py'])