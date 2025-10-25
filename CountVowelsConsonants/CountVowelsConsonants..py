"""
Count Vowels and Consonants
---------------------------
A simple Python program to count the number of vowels and consonants in a given string.

Features:
- Converts the input string to lowercase to ensure uniform counting.
- Ignores non-alphabetic characters.
- Returns both vowel and consonant counts.

Folder: CountVowelsConsonants
File: CountVowelsConsonants.py
Author: Priyadharshni G
"""

# Function to count vowels and consonants in a given text
def count_vowels_consonants(text):
    vowels = "aeiou"           # Define the vowels
    v_count = 0                # Initialize vowel counter
    c_count = 0                # Initialize consonant counter
    
    text = text.lower()        # Convert text to lowercase for uniform comparison
    
    for char in text:
        if char.isalpha():     # Process only alphabetic characters
            if char in vowels: 
                v_count += 1    # Increment vowel count
            else:
                c_count += 1    # Increment consonant count

    return v_count, c_count     # Return both counts


# Main function to handle user input and display results
def main():
    text = input("Enter a string: ")                     # Take input from the user
    vowels, consonants = count_vowels_consonants(text)   # Get the counts
    
    print(f"Vowels: {vowels}")                           # Display vowel count
    print(f"Consonants: {consonants}")                   # Display consonant count


# Entry point of the program
if __name__ == "__main__":
    main()
