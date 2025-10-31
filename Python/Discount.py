x = int(input("value1: "))
y = int(input("value2: "))
z = int(input("value3: "))
total = x + y + z
print("total amount:", total)

if x == y:
    a = (x + y) * 0.1
    print("bill:", total - a)
elif y == z:
    b = (y + z) * 0.1
    print("bill:", total - b)
elif x == z:
    c = (x + z) * 0.1
    print("bill:", total - c)
else:
    print("not discount bill:", total)
