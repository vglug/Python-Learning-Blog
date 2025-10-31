#!/usr/bin/env python3
"""
Simple GCD and Palindrome Programs
Educational programs for basic algorithms
"""

# ============================================================================
# GCD (Greatest Common Divisor) Programs
# ============================================================================

def gcd_euclidean(a, b):
    """
    Calculate GCD using Euclidean algorithm.
    
    Algorithm:
    - GCD(a, b) = GCD(b, a % b)
    - Base case: GCD(a, 0) = a
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Greatest Common Divisor of a and b
    """
    while b != 0:
        a, b = b, a % b
    return a


def gcd_recursive(a, b):
    """
    Calculate GCD using recursive approach.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Greatest Common Divisor of a and b
    """
    if b == 0:
        return a
    return gcd_recursive(b, a % b)


def gcd_subtraction(a, b):
    """
    Calculate GCD using subtraction method.
    
    Algorithm:
    - Keep subtracting smaller number from larger
    - Continue until both become equal
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Greatest Common Divisor of a and b
    """
    while a != b:
        if a > b:
            a = a - b
        else:
            b = b - a
    return a


def lcm(a, b):
    """
    Calculate LCM (Least Common Multiple) using GCD.
    
    Formula: LCM(a, b) = (a * b) / GCD(a, b)
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        Least Common Multiple of a and b
    """
    return abs(a * b) // gcd_euclidean(a, b)


# ============================================================================
# Palindrome Programs
# ============================================================================

def is_palindrome_string(text):
    """
    Check if a string is a palindrome.
    
    A palindrome reads the same forward and backward.
    
    Args:
        text: String to check
    
    Returns:
        True if palindrome, False otherwise
    """
    # Remove spaces and convert to lowercase
    text = text.replace(" ", "").lower()
    
    # Compare with reversed string
    return text == text[::-1]


def is_palindrome_number(num):
    """
    Check if a number is a palindrome.
    
    Args:
        num: Number to check
    
    Returns:
        True if palindrome, False otherwise
    """
    # Convert to string and check
    return str(num) == str(num)[::-1]


def is_palindrome_manual(text):
    """
    Check palindrome using two-pointer approach.
    
    Args:
        text: String to check
    
    Returns:
        True if palindrome, False otherwise
    """
    text = text.replace(" ", "").lower()
    
    left = 0
    right = len(text) - 1
    
    while left < right:
        if text[left] != text[right]:
            return False
        left += 1
        right -= 1
    
    return True


def reverse_number(num):
    """
    Reverse a number.
    
    Args:
        num: Number to reverse
    
    Returns:
        Reversed number
    """
    reversed_num = 0
    temp = abs(num)
    
    while temp > 0:
        digit = temp % 10
        reversed_num = reversed_num * 10 + digit
        temp = temp // 10
    
    return reversed_num if num >= 0 else -reversed_num


def is_palindrome_number_manual(num):
    """
    Check if number is palindrome without string conversion.
    
    Args:
        num: Number to check
    
    Returns:
        True if palindrome, False otherwise
    """
    if num < 0:
        return False
    
    return num == reverse_number(num)


def find_palindromes_in_range(start, end):
    """
    Find all palindrome numbers in a range.
    
    Args:
        start: Start of range
        end: End of range
    
    Returns:
        List of palindrome numbers
    """
    palindromes = []
    
    for num in range(start, end + 1):
        if is_palindrome_number(num):
            palindromes.append(num)
    
    return palindromes


def make_palindrome(text):
    """
    Create a palindrome by appending reverse.
    
    Args:
        text: Input string
    
    Returns:
        Palindrome string
    """
    return text + text[::-1]


# ============================================================================
# Interactive Demo Functions
# ============================================================================

def demo_gcd():
    """Demonstrate GCD calculations."""
    print("\n" + "="*60)
    print("GCD (Greatest Common Divisor) Demo")
    print("="*60)
    
    test_cases = [
        (48, 18),
        (100, 50),
        (17, 19),
        (270, 192),
        (1071, 462)
    ]
    
    for a, b in test_cases:
        print(f"\nNumbers: {a} and {b}")
        print(f"  GCD (Euclidean):    {gcd_euclidean(a, b)}")
        print(f"  GCD (Recursive):    {gcd_recursive(a, b)}")
        print(f"  GCD (Subtraction):  {gcd_subtraction(a, b)}")
        print(f"  LCM:                {lcm(a, b)}")


def demo_palindrome():
    """Demonstrate palindrome checking."""
    print("\n" + "="*60)
    print("Palindrome Demo")
    print("="*60)
    
    # String palindromes
    print("\n--- String Palindromes ---")
    strings = ["racecar", "hello", "madam", "A man a plan a canal Panama", "python"]
    
    for text in strings:
        result = is_palindrome_string(text)
        status = "✓ Palindrome" if result else "✗ Not palindrome"
        print(f"'{text}' -> {status}")
    
    # Number palindromes
    print("\n--- Number Palindromes ---")
    numbers = [121, 123, 1221, 12321, 12345, 999]
    
    for num in numbers:
        result = is_palindrome_number(num)
        status = "✓ Palindrome" if result else "✗ Not palindrome"
        print(f"{num} -> {status}")
    
    # Palindromes in range
    print("\n--- Palindromes in Range 100-200 ---")
    palindromes = find_palindromes_in_range(100, 200)
    print(f"Found {len(palindromes)} palindromes:")
    print(palindromes)


def interactive_gcd():
    """Interactive GCD calculator."""
    print("\n" + "="*60)
    print("Interactive GCD Calculator")
    print("="*60)
    
    try:
        a = int(input("\nEnter first number: "))
        b = int(input("Enter second number: "))
        
        gcd = gcd_euclidean(abs(a), abs(b))
        lcm_val = lcm(abs(a), abs(b))
        
        print(f"\n--- Results ---")
        print(f"GCD of {a} and {b} = {gcd}")
        print(f"LCM of {a} and {b} = {lcm_val}")
        
    except ValueError:
        print("Error: Please enter valid integers!")


def interactive_palindrome():
    """Interactive palindrome checker."""
    print("\n" + "="*60)
    print("Interactive Palindrome Checker")
    print("="*60)
    
    text = input("\nEnter text or number to check: ")
    
    # Try as number first
    try:
        num = int(text)
        is_pal = is_palindrome_number(num)
        reversed_num = reverse_number(num)
        
        print(f"\n--- Number Analysis ---")
        print(f"Original: {num}")
        print(f"Reversed: {reversed_num}")
        print(f"Result: {'✓ PALINDROME' if is_pal else '✗ NOT A PALINDROME'}")
        
    except ValueError:
        # Treat as string
        is_pal = is_palindrome_string(text)
        reversed_text = text[::-1]
        
        print(f"\n--- String Analysis ---")
        print(f"Original: {text}")
        print(f"Reversed: {reversed_text}")
        print(f"Result: {'✓ PALINDROME' if is_pal else '✗ NOT A PALINDROME'}")


# ============================================================================
# Main Program
# ============================================================================

def main():
    """Main program with menu."""
    while True:
        print("\n" + "="*60)
        print("GCD and Palindrome Programs")
        print("="*60)
        print("\n1. GCD Demo")
        print("2. Palindrome Demo")
        print("3. Interactive GCD Calculator")
        print("4. Interactive Palindrome Checker")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            demo_gcd()
        elif choice == '2':
            demo_palindrome()
        elif choice == '3':
            interactive_gcd()
        elif choice == '4':
            interactive_palindrome()
        elif choice == '5':
            print("\nThank you! Goodbye!")
            break
        else:
            print("\nInvalid choice! Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
