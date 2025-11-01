a = input("Enter the value: ")
b = input("Enter the type of given datatype (int, str, float, list, tuple): ")
c = input("Which datatype do you want to convert to (int, str, float, list, tuple): ")

if b == "int":
    a = int(a)
elif b == "float":
    a = float(a)
elif b == "list" or b == "tuple":
    a = eval(a)   
elif b == "str":
    a = str(a)
else:
    print("Invalid original type")

if c == "int":
    a = int(a)
elif c == "float":
    a = float(a)
elif c == "str":
    a = str(a)
elif c == "list":
    a = list(a)
elif c == "tuple":
    a = tuple(a)
else:
    print("Invalid target type")

print("Converted value:", a)
print("New datatype:", type(a))
