import pytest
import conftest
from conversation import *

@pytest.mark.parametrize('filename, ai_name, model, seed, test_dialogue', [
    ('tests/test1', 'CodeBot', 'gpt-3.5-turbo', 42, ['Ok', "Great! If you have any more questions, feel free to ask. I'm here to help!"]),
    ('tests/test2', 'Red', 'gpt-4-1106-preview', 42, ['Hi', 'Hello! How can I assist you today?']),
    ('tests/test3', 'ChefBot', 'gpt-4', 42, []),
])
def test_conversation(filename, ai_name, model, seed, test_dialogue):
    chat = Conversation(filename=filename, seed=seed)
    assert chat.ai_name == ai_name
    assert chat.model == model

    while test_dialogue: # Alternates between user input and ai validation
        assert chat.get_dialogue(test_dialogue.pop(0)) == True
        assert chat.messages[-1]['content'] == test_dialogue.pop(0)

    chat.summarize_messages(500)
    chat.save(filename+'_out.yaml')
    chat.save(filename+'_out.json')

    yaml = Conversation(filename=filename+'_out.yaml')
    json = Conversation(filename=filename+'_out.json')

    assert yaml.model == json.model
    assert yaml.user_name == json.user_name
    assert yaml.ai_name == json.ai_name

    assert yaml.messages == json.messages
    assert yaml.history == json.history

if __name__ == '__main__':
    pytest.main([__file__])