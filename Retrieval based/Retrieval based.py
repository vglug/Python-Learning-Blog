import re
import random
patterns = {
    r"hi|hello|hey": ["Hello!", "Hi there!", "Hey! How can I help you?"],
    r"how are you": ["I'm a bot, but I'm doing great!", "All good! How about you?"],
    r"your name": ["I am ChatBot.", "You can call me ChatBot."],
    r"what can you do": ["I can chat with you and answer simple questions.", "I can respond to greetings and basic queries."],
    r"bye|exit": ["Goodbye!", "See you later!", "It was nice talking to you!"]
}
def chatbot_response(user_input):
    user_input = user_input.lower()
    for pattern, responses in patterns.items():
        if re.search(pattern, user_input):
            return random.choice(responses)
    return "Sorry, I don't understand that."
print(" ChatBot: Hello! Type 'bye' or 'exit' to end the chat.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "exit"]:
        print("ChatBot:", chatbot_response(user_input))
        break
    response = chatbot_response(user_input)
    print(" ChatBot:", response)
