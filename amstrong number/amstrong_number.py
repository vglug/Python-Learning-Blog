num = int(input("Enter the number: "))#enter the number
num1=num
sum1= 0
power = len(str(num))#count the lenght of the number
while num!=0:# it check the condition untill the number is 0
                    val=num % 10  # remainder in val
                    sum1+=(val**power) 
                    num //= 10
if num1==sum1:#check the condition
                    print(" armstrong number ",num1)
else:
                    print("  not armstrong number ",num1)
