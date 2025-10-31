total = 0
number = int(input("Enter a number (0 to stop): "))

while number != 0:
    total += number
    number = int(input("Enter another number (0 to stop): "))

print("The total sum is:", total)
