# Ask the user to enter the first number and convert it to a float (decimal value)
a = float(input("Enter first number: "))

# Ask the user to enter the second number and convert it to a float
b = float(input("Enter second number: "))

# Perform and display basic arithmetic operations
print("Addition:", a + b)          # Adds the two numbers
print("Subtraction:", a - b)       # Subtracts second number from the first
print("Multiplication:", a * b)    # Multiplies the two numbers

# Check if the second number is not zero before dividing
if b != 0:
    print("Division:", a / b)      # Divides first number by the second
else:
    print("Not divisible by zero") # Display message if user entered 0 for b
