# Function to reverse a string using recursion
def reverse_string(s):
    # If string is empty, return it
    if len(s) == 0:
        return s
    else:
        # Call function again for the rest of the string
        return reverse_string(s[1:]) + s[0]

# Get input from user
text = input("Enter a string: ")

# Call the function
result = reverse_string(text)

# Print the reversed string
print("Reversed string:", result)
