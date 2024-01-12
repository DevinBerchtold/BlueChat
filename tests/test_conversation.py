import pytest
import conftest
from conversation import *

@pytest.mark.parametrize('filename,ai_name,model', [
    ('tests/test1', 'CodeBot', 'gpt-3.5-turbo'),
    ('tests/test2', 'Red', 'gpt-4'),
    ('tests/test3', 'ChefBot', 'gpt-4'),
])
def test_conversation(filename, ai_name, model):
    chat = Conversation(filename=filename)
    assert chat.ai_name == ai_name
    # assert chat.model == model # Removed because gpt-4-1106-preview is default, but it's temporary
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