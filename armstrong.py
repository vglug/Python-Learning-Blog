num = int(input("Enter a value: "))
sum = 0
temp = num

while temp > 0:
    digit = temp % 10
    sum += digit ** 3
    temp //= 10

if sum == num:
    print(num, "is an Armstrong number")
else:
    print(num, "is not an Armstrong number")
