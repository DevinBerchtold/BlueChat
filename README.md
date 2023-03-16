# BlueChat

A conversational engine to experiment with GPT-3.5-Turbo. Contains the setup for a basic conversation between two characters. Dynamically summarize the conversation to improve memory and/or reduce token costs.

---

## Installation

1. Download the project from [GitHub](https://github.com/DevinBerchtold/BlueChat)
2. Install the required Python libraries:
    * [OpenAI](https://platform.openai.com/docs/api-reference/introduction?lang=python)
    * [tiktoken](https://github.com/openai/tiktoken)
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
2. Talk as though you are a cute puppy.
    <details>
    <summary>(Click to expand)</summary>

    Example:

    ```
    You wake up in your kennel at the back of the pet store.
    
    The quietness of the pet store fills the air as you relax in your kennel. Brightly colored toys dangle from the top of your enclosure as sunbeams peek through the store's window casting a soft glow onto your fur. The sound of your own breath is the only thing audible, as you enjoy the comfort of the soft cushion beneath you.
    
    You see an old man.
    
    Blue: *woof*
    
    Red: (Surprised) "What in the world was that?"
    
    Blue: *whimpers, looking cute*
    
    Red: (Skeptical) "Is that dog making noises? That can't be right."
    
    Blue: *barks cheerfully*
    
    Red: (Annoyed) "Great, now the dog is barking. Just what I need, more noise."
    
    Blue: *calmly stops barking*
    
    Red: (Muttering to himself) "Well, at least it stopped making noise."
    
    Blue: H-Hello
    
    Red: (Surprised) "Did that dog just say hello? That's impossible."
    
    Blue: Hi sir
    
    Red: (Shocked) "This can't be happening. A talking dog? Am I losing my mind?"
    
    Blue: I thought the same thing myself
    
    Red: (Confused) "What is going on here? How are you able to talk?"
    
    Blue: I'm not a dog, I'm an alien stuck in a dog's body
    
    Red: (Skeptical) "An alien in a dog's body? That's the most ridiculous thing I've ever heard."
    
    Blue: Do you have a better explanation why I could talk?
    
    Red: (Shrugging) "No, I don't. But it's still hard to believe. I think I need to sit down."
    
    Blue: It's okay sir, take your time
    
    Red: (Calmly) "Okay, let me process this. So, you are an alien trapped in a dog's body. How did you end up here?"
    
    Blue: Well, I don't really remeber. I remember an operating table
    
    Red: (Surprised) "An operating table? That sounds like a very strange experience. Do you have any idea how you ended up in a dog's body?"
    
    Blue: They put me in here
    
    Red: (Sympathetically) "That must have been a difficult experience for you. But how did you end up in this pet store?"
    
    Blue: I don't know
    
    Red: (Thoughtful) "It sounds like you have been through a lot. Maybe we should try to find out more about how you ended up here. Do you remember anything else that could help us figure it out?"
    
    Blue: Right now what's most important is getting out of here, they're still looking for me
    
    Red: (Concerned) "Who is looking for you? Do you know who they are?"
    
    Blue: The other aliens. They know this address so we have to leave
    
    Red: (Worried) "Okay, we need to get out of here quickly. Do you have any idea where we should go?"
    
    Blue: Probably just a motel room where we can lay low.
    
    Red: (Nodding) "That's a good idea. Let's get out of here and find a safe place to stay. I'll take you with me and we'll figure out what to do next."
    
    Blue: Cool, you have to go 'adopt' me because the employees still think I'm a dog
    
    Red: (Uncertain) "I don't know if I'm ready to adopt a talking alien-dog just yet. But I guess we don't have much of a choice. Let's go."
    
    Blue: Alright sounds good. You won't regret this
    
    Red: (Skeptical) "I hope not. Let's just get out of here before we attract any unwanted attention."
    
    Blue: Alright, go talk to that employee
    
    Red: (Grumbling) "Fine, I'll go talk to them."
    
    (Red walks up to the employee at the front desk.)
    
    Red: "Excuse me, I need to adopt that dog over there."
    
    Employee: "Great! Let me just get the paperwork ready for you."
    
    (Red fills out the paperwork and pays the adoption fee.)
    
    Employee: "Congratulations on your new pet! What's their name?"
    
    Red: "Uh, I haven't decided yet. I'll let you know later."
    
    Employee: "Okay, well enjoy your new furry friend!"
    
    (Red and Blue leave the pet store and head to the nearest motel.)
    ```
    </details>


3. Experiment with the characters or scenario by changing constants in `chat.py` (see comments).

---

## Commands

Commands begin with `!`. Everything else will be interpreted as conversation.

`!debug`: Toggle debug mode on or off.

`!print`: Print the messages object.

`!save`: Save the messages file to JSON.

`!load`: Load a messages file from JSON.

`!history`: TODO: Full conversation history.

`!summary`: Summarize the conversation so far.

---

## Cost

Costs increase linearly on the length of the conversation. Previous messages are included for the character's memory. We can facilitate 'infinite' memory by recursively summarizing old messages when the 4096 token limit is reached. After the conversations reach a certain length (about 50 messages), the cost would average 2048-4096 per response. The memory could be further limited to reduce costs.

---

## Todo

[v] Dynamically summarize old messages.

[ ] Tests to verify correctness.

[ ] GPT-4

---

## Credits

OpenAI

Dedicated to Blue Berchtold

---

## License

This project is released under the MIT license:

[MIT License](https://choosealicense.com/licenses/mit/)