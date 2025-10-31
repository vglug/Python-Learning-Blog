# String Slicing and Indexing Example

# Input a string from the user
text = input("Enter a string: ")

# Displaying first and last character
print("First character:", text[0])
print("Last character:", text[-1])

# Display a substring (from index 2 to 5)
print("Substring (index 2 to 5):", text[2:6])

# Reverse the string
print("Reversed string:", text[::-1])

# Print every second character
print("Every 2nd character:", text[::2])

# Length of the string
print("Length of the string:", len(text))
