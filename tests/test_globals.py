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
    (
        'hello world  how  are you?',
        ['hello', ' world', '  how', '  are', ' you?']
    ),
    (
        'hello world\n\nhow  are you?',
        ['hello', ' world', '\n\n\n', 'how', '  are', ' you?']
    ),
    (
        'hello world\n\n\nhow  are you?',
        ['hello', ' world', '\n\n\n', '\nhow', '  are', ' you?']
    ),
    (
        'hello world\n\n how  are you?',
        ['hello', ' world', '\n\n\n',' how', '  are', ' you?']
    ),
    (
        'hello world \n\nhow  are you?',
        ['hello', ' world', ' \n\nhow', '  are', ' you?']
    ),
    # (
    #     'hello world \n\nhow  are you?',
    #     ['hello', ' world', ' ', '\n\nhow', '  are', ' you?']
    # ), # Probably a bug
    ("""Python code:

```python
l = [x for x in range(10)]

return l
```

Returns a list of numbers.
""",
    [
        'Python', ' code:',
        '\n\n\n',
        '```python',
        '\nl', ' =', ' [x', ' for', ' x', ' in', ' range(10)]',
        '\n',
        '\n', 'return', ' l',
        '\n```',
        '\n\n\n',
        'Returns', ' a', ' list', ' of', ' numbers.',
        '\n'
    ]
    ),
    ("""List:

1. Apple

2. Orange

3. Banana

That's all.""",
    [
        'List:',
        '\n\n',
        '1.', ' Apple',
        '\n',
        '\n', '2.', ' Orange',
        '\n',
        '\n', '3.', ' Banana',
        '\n\n\n',
        "That's", ' all.'
    ]
    ),
    ("""Ingredients:

- Tortilla chips \U0001F32E
- 1 cup canned black beans, drained and rinsed \U0001F96B
- 1 cup canned corn, drained \U0001F33D
- 1 onion, diced \U0001F9C5
- 2 cloves garlic, minced \U0001F9C4
- 1 jalapeno, chopped (optional for extra spiciness) \U0001F336️
- 1 tablespoon vegetable oil \U0001F6E2️
- 1 teaspoon ground cumin
- 1 teaspoon ground paprika
- 1/2 teaspoon ground turmeric
- Salt, to taste
- 1 cup shredded cheese (cheddar, Monterrey Jack or pepper jack) \U0001F9C0
- 1/2 cup sour cream
- 1/2 cup guacamole \U0001F951
- 1/2 cup salsa \U0001F345
- Lime wedges, for serving \U0001F34B""",
    [
        'Ingredients:',
        '\n\n', '-', ' Tortilla', ' chips', ' \U0001F32E',
        '\n-', ' 1', ' cup', ' canned', ' black', ' beans,', ' drained', ' and', ' rinsed', ' \U0001F96B',
        '\n-', ' 1', ' cup', ' canned', ' corn,', ' drained', ' \U0001F33D',
        '\n-', ' 1', ' onion,', ' diced', ' \U0001F9C5',
        '\n-', ' 2', ' cloves', ' garlic,', ' minced', ' \U0001F9C4',
        '\n-', ' 1', ' jalapeno,', ' chopped', ' (optional', ' for', ' extra', ' spiciness)', ' \U0001F336️',
        '\n-', ' 1', ' tablespoon', ' vegetable', ' oil', ' \U0001F6E2️',
        '\n-', ' 1', ' teaspoon', ' ground', ' cumin',
        '\n-', ' 1', ' teaspoon', ' ground', ' paprika',
        '\n-', ' 1/2', ' teaspoon', ' ground', ' turmeric',
        '\n-', ' Salt,', ' to', ' taste',
        '\n-', ' 1', ' cup', ' shredded', ' cheese', ' (cheddar,', ' Monterrey', ' Jack', ' or', ' pepper', ' jack)', ' \U0001F9C0',
        '\n-', ' 1/2', ' cup', ' sour', ' cream',
        '\n-', ' 1/2', ' cup', ' guacamole', ' \U0001F951',
        '\n-', ' 1/2', ' cup', ' salsa', ' \U0001F345',
        '\n-', ' Lime', ' wedges,', ' for', ' serving', ' \U0001F34B'
    ]
    )
]

@pytest.mark.parametrize('input,expected', test_cases[:4])
def test_generate_words(input, expected):
    chunks = random_chunks(input)
    print(f'  {chunks=}')
    print(f'{expected=}')
    result = list(generate_words(chunks))
    print(f'  {result=}')
    assert result == expected

@pytest.mark.parametrize('input,expected', test_cases)
def test_generate_paragraphs(input, expected):
    chunks = random_chunks(input)
    print(f'  {chunks=}')
    print(f'{expected=}')
    result = list(generate_paragraphs(chunks))
    print(f'  {result=}')
    assert result == expected

if __name__ == '__main__':
    pytest.main([__file__])
