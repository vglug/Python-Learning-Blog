# Take two numbers as input from the user
a = int(input("a: "))
b = int(input("b: "))

# Perform basic arithmetic operations
print(f"Sum of {a} and {b} is {a + b}")              # Addition
print(f"Subtraction of {a} and {b} is {a - b}")      # Subtraction
print(f"Multiplication of {a} and {b} is {a * b}")   # Multiplication

# Try to perform division and handle division by zero error
try:
    print(f"Division of {a} and {b} is {a / b}")     # Division
except Exception as e:
    print("Division by 0 is impossible")              # Error message if b = 0
