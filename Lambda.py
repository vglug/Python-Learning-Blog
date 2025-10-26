add = lambda x, y: x + y
print("1. Add two numbers:", add(5, 3))

square = lambda x: x ** 2
print("   Square a number:", square(5))

numbers = [1, 2, 3, 4, 5]

squared = list(map(lambda x: x ** 2, numbers))
print("\n2. Squared numbers:", squared)

strings = list(map(lambda x: f"Number: {x}", numbers))
print("   As strings:", strings)

evens = list(filter(lambda x: x % 2 == 0, numbers))
print("\n3. Even numbers:", evens)

greater_than_3 = list(filter(lambda x: x > 3, numbers))
print("   Greater than 3:", greater_than_3)

students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78}
]

sorted_students = sorted(students, key=lambda x: x["grade"], reverse=True)
print("\n4. Students sorted by grade:")
for student in sorted_students:
    print(f"   {student['name']}: {student['grade']}")

from functools import reduce

total = reduce(lambda x, y: x + y, numbers)
print("\n5. Sum of numbers:", total)

maximum = reduce(lambda x, y: x if x > y else y, numbers)
print("   Maximum value:", maximum)

celsius_temps = [0, 10, 20, 30]
fahrenheit = list(map(lambda c: (c * 9/5) + 32, celsius_temps))
print("\n6. Celsius to Fahrenheit:", fahrenheit)

data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

flattened = reduce(lambda x, y: x + y, data)
even_flattened = list(filter(lambda x: x % 2 == 0, flattened))
print("\n7. Flattened even numbers:", even_flattened)

words = ["hello", "world", "python", "lambda"]

uppercase = list(map(lambda w: w.upper(), words))
print("\n8. Uppercase words:", uppercase)

long_words = list(filter(lambda w: len(w) > 5, words))
print("   Words with length > 5:", long_words)

classification = list(map(
    lambda x: "even" if x % 2 == 0 else "odd",
    numbers
))
print("\n9. Number classification:", classification)

prices = [100, 250, 50, 300]
discounted = list(map(
    lambda p: p * 0.8 if p > 100 else p,
    prices
))
print("   Discounted prices:", discounted)

data_list = ["user1@gmail.com", "john_doe", "admin@example.org", "invalid"]
emails = list(filter(lambda x: "@" in x and "." in x, data_list))
print("\n10. Valid emails:", emails)

items = [{"name": "apple", "price": 1.5}, {"name": "banana", "price": 0.8}]
total_with_tax = sum(map(lambda item: item["price"] * 1.1, items))
print("    Total with 10% tax:", round(total_with_tax, 2))


def apply_operation(x, y, operation):
    return operation(x, y)

result1 = apply_operation(10, 5, lambda a, b: a + b)
result2 = apply_operation(10, 5, lambda a, b: a * b)
result3 = apply_operation(10, 5, lambda a, b: a ** b)

print("\n11. Custom operations:")
print(f"    10 + 5 = {result1}")
print(f"    10 * 5 = {result2}")
print(f"    10 ^ 5 = {result3}")

large_dataset = range(1, 1001)
result = len(list(filter(lambda x: x % 2 == 0 and x > 500, large_dataset)))
print(f"\n12. Count of even numbers > 500: {result}")

print("\n✓ Lambda functions are perfect for small, quick operations!")



