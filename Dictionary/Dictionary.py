student = {
    "name": "Alice",
    "age": 21,
    "course": "Computer Science"
}

print("Name:", student["name"])
print("Age:", student["age"])

student["grade"] = "A"
student["age"] = 22
del student["course"]

print("\nUpdated Student Dictionary:")
print(student)
