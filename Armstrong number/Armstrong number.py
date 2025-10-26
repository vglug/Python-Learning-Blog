num = int(input("Enter the number: "))
num1=num
sum1= 0
power = len(str(num))
while num!=0:
                    val=num % 10
                    sum1+=(val**power)
                    num //= 10
if num1==sum1:
                    print(" armstrong number ",num1)
else:
                    print("  not armstrong number ",num1)

    


