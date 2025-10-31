# Find 2nd largest element using sort()
numbers = [10, 20, 4, 45, 99, 6]

# Remove duplicates
numbers = list(set(numbers))

# Sort in ascending order
numbers.sort()

# Second largest element
second_largest = numbers[-2]

print("The second largest element is:", second_largest)
