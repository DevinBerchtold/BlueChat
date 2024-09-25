# BlueChat

A conversational engine for experimenting with OpenAI GPT Chat API. Enables a basic 2-way conversation between the user and an AI, or between two AIs.

* Create custom prompts to chat with any AI and seamlessly switch between them (e.g., story characters, assistants, teachers)
* Automatically save chats and easily continue past conversations
* Dynamically summarize conversations to enhance memory and reduce token costs
* More reliable and likely cheaper than a ChatGPT subscription (depending on usage)
* Chat with your AIs via the terminal like you're in a sci-fi movie

![Example Usage](image/example.gif)


---

## Installation

1. Download the project from [GitHub](https://github.com/DevinBerchtold/BlueChat)
2. Install the required Python libraries:
    * [OpenAI](https://platform.openai.com/docs/api-reference/introduction?lang=python)
    * [tiktoken](https://github.com/openai/tiktoken)
    * [Rich](https://pypi.org/project/rich/)
    * [Pyperclip](https://pypi.org/project/pyperclip/)
    * [PyYAML](https://pypi.org/project/PyYAML/) (Recommended)
3. Add your OpenAI API key:

    ```
    setx OPENAI_API_KEY "your key"
    ```

    [Best Practices for API Key Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)

---

## Usage

1. Run it:

    ```
    py chat.py
    ```

2. Talk as though you are a cute puppy or ask questions to one of the many assistants.

3. Create new AIs to chat with by adding a YAML file with their system message.

---

## Commands

Commands begin with `!`. Everything else will be interpreted as conversation.

`<!help|!h>`: Print the help screen.

`<!exit|!e>`: Exit the application.

`<!restart|!r>`: Restart the chat by clearing memory and creating a new save file.

`<!undo|!u>`: Undo and remove the last set of messages in the chat.

`<!save|!s> [filename]`: Save the chat to `filename`. Automatically adds file extension and names the file if `filename` is not specified.

`<!load|!l> [filename]`: Load the chat from `filename`. Loads the latest file if `filename` is a bot name or the exact file if it's a filename.

`<!summary|!su> [tokens]`: Summarize the last `tokens` of the conversation. Summarizes the entire conversation if `tokens` is not specified.

`<!history|!hi>`: Print the conversation system messages and full chat history.

`<!model|!m> [reset] [model]`: Set the LLM model to be used in the chat if `model` is specified, or prints the current model otherwise. Valid values for `model` are 'gpt-3.5', 'gpt-4', 'gemini', 'claude', or 'llama'. If `reset` is 'redo', the last message is regenerated with the new model.

`<!tools|!t> [tool]`: Toggle tool usage for `tool` if specified. Otherwise, toggle tool usage for all tools. Valid values for `tool` are 'python' and 'browse'.

`<!translate|!tr> [user] [ai]`: Toggle translate mode if no parameters are specified. If specified, `ai` is the AI language and `user` is the users language.

`<!debug|!d>`: Toggle debug mode for all modules.

`<!auto|!a> [text]`: Toggle auto-switch context mode. If `text` is specified, context is checked and `text` is sent as a message.

`<!variable|!v> [parm]`: Set or print variables for global or chat settings. If `parm` is specified in the format `variable=value`, set the variable. If not specified, print all variables.

`<!copy|!c> [parm]`: Copy the last `parm` chat messages to the clipboard if `parm` is a number or 'all'. Copy code blocks from the last message if `parm` is 'code'. Copies 2 messages by default.

`<!paste|!p> [text]`: Paste the clipboard content and send as a message. If `text` is specified, `text` is prepended to the message before sending.

`<!image|!i> [text]`: Attach an image or image URL from the clipboard. If `text` is specified, `text` is included with the message.

`<!theme|!th> [theme]`: Set the syntax highlighting theme to the pygments theme `theme`.

---

## Customization

Add new AIs by adding YAML files to the `conversations` folder. As an example lets make a new bot, `emoji.yaml`

```yaml 
- role: system
  content: You are a helpful assistant that uses lots of emojis.
```

Add a description in `bots.yaml` to enable automatic context switching based on the user's input. This allows you to automatically chat with the most relevant bot for your task/question.

```yaml
# Variables: User, default bot, last bot, context switching toggle
variables: {USER_NAME: Devin, DEFAULT: help, FILENAME: code, SWITCH: true}
bots:
  fact: Gives encyclopedia entries for things and ideas.
  # ...
  # Key is the filename for your bot. Value is the description used for context switching
  emoji: General help. Responds when user sends emojis.
```

Now, EmojiBot will automatically respond when the criteria in `bots.yaml` is met.

![EmojiBot](image/emoji.gif)

---

## Cost

Costs increase linearly on the length of the conversation. Previous messages are included for the character's memory. We can facilitate 'infinite' memory by recursively summarizing old messages when the token limit is reached. After the conversations reach a certain length (about 50 messages at max memory), the cost would average 2048-4096 tokens per response. The memory can be limited to reduce costs.

---

## Credits

This project uses [OpenAI API](https://platform.openai.com/) for AI functionality

Dedicated to Blue Berchtold

---

## License

This project is released under the MIT license:

[MIT License](https://choosealicense.com/licenses/mit/)