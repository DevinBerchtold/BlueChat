import pytest
import conftest
import console
import conversation

console.RICH = False
conversation.CONFIRM = False

@pytest.mark.parametrize('filename, ai_name, model_id, seed, test_dialogue', [
    ('tests/test4', 'Red', 'gpt-4-turbo-preview', 42, ['What is 2.45/2.44?', "The result of dividing 2.45 by 2.44 is approximately 1.0041."]),
    ('tests/test5', 'Red', 'gpt-4-turbo-preview', 42, ['56th fibonacci number', 'The 56th Fibonacci number is 225,851,433,717.']),
])
def test_functions(filename, ai_name, model_id, seed, test_dialogue):
    chat = conversation.Conversation(filename=filename, seed=seed, use_tools=True, confirm=False)
    assert chat.ai_name == ai_name
    assert chat.model.id == model_id

    while test_dialogue: # Alternates between user input and ai validation
        assert chat.get_dialogue(test_dialogue.pop(0)) == True
        assert chat.messages[-1].content == test_dialogue.pop(0)

    # chat.summarize_messages(500)
    chat.save(filename+'_out.yaml')
    chat.save(filename+'_out.json')

    yaml = conversation.Conversation(filename=filename+'_out.yaml')
    json = conversation.Conversation(filename=filename+'_out.json')

    assert yaml.model == json.model
    assert yaml.user_name == json.user_name
    assert yaml.ai_name == json.ai_name

    assert yaml.messages == json.messages
    assert yaml.history == json.history

if __name__ == '__main__':
    pytest.main([__file__])