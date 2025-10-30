# Character Type Checker

ch = input("Enter any character: ")

if ch.isalpha():  # Check if it's an alphabet
    if ch.lower() in ['a', 'e', 'i', 'o', 'u']:
        print(f"{ch} is a vowel.")
    else:
        print(f"{ch} is a consonant.")
elif ch.isdigit():  # Check if it's a number
    print(f"{ch} is a digit.")
else:  # Otherwise, it's a special character
    print(f"{ch} is a special character.")
