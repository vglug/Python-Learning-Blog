# Prompt the user to enter a sentence
text = input("Enter a sentence: ")

# Split the sentence into individual words
words = text.split()

# Create an empty dictionary to store word counts
count = {}

# Loop through each word in the list
for w in words:
    # Increase the count of the word by 1 if it already exists, otherwise set it to 1
    count[w] = count.get(w, 0) + 1

# Print the dictionary containing each word and its frequency
print(count)
