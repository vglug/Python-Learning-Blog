


num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

print("\nSelect operation:")
print("1. Addition (+)")
print("2. Subtraction (-)")
print("3. Multiplication (*)")
print("4. Division (/)")

choice = input("Enter your choice (1/2/3/4): ")


if choice == '1':
    print(f"The result of addition: {num1 + num2}")
elif choice == '2':
    print(f"The result of subtraction: {num1 - num2}")
elif choice == '3':
    print(f"The result of multiplication: {num1 * num2}")
elif choice == '4':
    if num2 != 0:
        print(f"The result of division: {num1 / num2}")
    else:
        print("Error: Division by zero is not allowed.")
else:
    print("Invalid choice! Please select a valid operation.")
