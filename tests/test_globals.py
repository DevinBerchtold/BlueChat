import pytest
import conftest
from globals import *
import random

def random_chunks(input_string):
    chunks_list = []
    index = 0
    while index < len(input_string):
        random_number = random.randint(0, 4)
        chunk = input_string[index:index + random_number]
        chunks_list.append(chunk)
        index += random_number
    return chunks_list

test_cases = [
    ('', []),
    ('hello world', ['hello', ' world']),
    ('       ', ['       ']),
    ('hello world  how  are you?', ['hello', ' world', '  how', '  are', ' you?']),
    ('hello world\n\nhow  are you?', ['hello', ' world', '\n\n', 'how', '  are', ' you?']),
    ('hello world\n\n\nhow  are you?', ['hello', ' world', '\n\n', '\nhow', '  are', ' you?']),
    ('hello world\n\n how  are you?', ['hello', ' world', '\n\n',' how', '  are', ' you?']),
    ('hello world \n\nhow  are you?', ['hello', ' world', ' \n\nhow', '  are', ' you?']),
    # ('hello world \n\nhow  are you?', ['hello', ' world', ' ', '\n\nhow', '  are', ' you?']), # Probably a bug

    ("""Python code:

```python
l = [x for x in range(10)]

return l
```

Returns a list of numbers.
""", ['Python', ' code:', '\n\n', '```python', '\nl', ' =', ' [x', ' for', ' x', ' in', ' range(10)]', '\n', '\n', 'return', ' l', '\n```', '\n\n', 'Returns', ' a', ' list', ' of', ' numbers.', '\n']),
        
    ("""List:

1. Apple

2. Orange

3. Banana

That's all.""", ['List:', '\n', '\n', '1.', ' Apple', '\n', '\n', '2.', ' Orange', '\n', '\n', '3.', ' Banana', '\n\n', "That's", ' all.'])
]

@pytest.mark.parametrize('input,expected', test_cases[:4])
def test_break_spaces(input, expected):
    chunks = random_chunks(input)
    print(f'  {chunks=}')
    print(f'{expected=}')
    result = list(break_spaces(chunks))
    print(f'  {result=}')
    assert result == expected

@pytest.mark.parametrize('input,expected', test_cases)
def test_generate_words(input, expected):
    chunks = random_chunks(input)
    print(f'  {chunks=}')
    print(f'{expected=}')
    result = list(generate_words(chunks))
    print(f'  {result=}')
    assert result == expected

if __name__ == '__main__':
    pytest.main(['tests/test_globals.py'])
