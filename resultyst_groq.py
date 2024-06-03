import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()

def chat_with_groq():

    messages = [
        {
            "role": "user",
            "content": "What is machine learning?"
        },
        {
            "role": "assistant",
            "content": "Machine learning is like teaching a smart robot to do tasks by showing it lots of examples. For instance, if you show the robot many pictures of cats and dogs and tell it which is which, it learns to recognize them on its own. This way, the robot uses what it has learned to make decisions about new pictures it hasn't seen before."
        },
        {
            "role": "user",
            "content": "What is a neural network?"
        },
        {
            "role": "assistant",
            "content": "A neural network is a computer system inspired by the way the human brain works. It consists of layers of interconnected nodes (like brain cells) that process information and learn to recognize patterns from data. By adjusting the connections between nodes, the network gets better at tasks like recognizing images or understanding speech.\n\n\n\n\n\n"
        },
        {
            "role": "user",
            "content": "What are the differences between supervised and unsupervised learning?"
        },
        {
            "role": "assistant",
            "content": "In supervised learning, the model is trained using labeled data, which means each example has an input and a known correct output (like teaching a robot with flashcards that have both the picture and the name of the object). In unsupervised learning, the model is given data without labels and must find patterns or groupings on its own (like giving the robot a pile of toys and asking it to sort them without telling it what each toy is). Essentially, supervised learning involves learning from examples with answers, while unsupervised learning involves discovering hidden patterns without specific guidance."
        },
        {
            "role": "user",
            "content": "Explain the backpropagation algorithm."
        },
        {
            "role": "assistant",
            "content": "Backpropagation is like learning from mistakes: when a computer makes a wrong guess, it looks at what it did and adjusts its thinking to do better next time. It does this by tweaking the connections between different parts of its \"brain,\" called neurons, based on how much they contributed to the mistake. Over time, this helps the computer learn to make more accurate guesses, like practicing riding a bike and getting better each time you fall."
        },
        {
            "role": "user",
            "content": "Explain feature scaling and normalization"
        },
        {
            "role": "assistant",
            "content": "Feature scaling and normalization are like making sure all the ingredients in a recipe are in the right proportions. Just like you wouldn't want too much salt or too little sugar in a cake, feature scaling makes sure all the different parts of the data are in a similar range so one doesn't overpower the others. This helps the computer learn better and make fair comparisons, like making sure all the ingredients in your cake are just right for the tastiest result!"
        },
        {
            "role": "user",
            "content": "Explain the concept of reinforcement learning."
        },
        {
            "role": "assistant",
            "content": "Reinforcement learning is like teaching a dog new tricks using treats as rewards. In this method, a computer (or \"agent\") learns to make decisions by trying different actions and receiving feedback, just like when the dog gets a treat for doing a trick right. The computer learns which actions lead to the best rewards over time, allowing it to improve its decision-making skills without being explicitly programmed."
        },
        {
            "role": "user",
            "content": "Explain transfer learning."
        },
        {
            "role": "assistant",
            "content": "Transfer learning is like learning to play a new game by using the skills you already have from playing a similar game. Instead of starting from scratch, you can apply what you've learned in one situation to another situation, which can help you learn faster. Similarly, in computer science, transfer learning involves using knowledge gained from solving one problem to help solve a related but different problem, making learning more efficient and effective."
        }
    ]
    
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending the chat. Goodbye!")
            break
        
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        
        print(f"Assistant: {response}")
        

        messages.append({
            "role": "assistant",
            "content": response
        })

if __name__ == "__main__":
    chat_with_groq()
