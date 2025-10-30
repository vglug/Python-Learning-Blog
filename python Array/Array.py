print("SUM OF ELEMENTS IN ARRAY")
arr=[] 
sum=0
n=int(input("Enter the Array size:"))
print("Enter the Array Elements:")
for i in range(0,n):
    element=int(input())
    arr.append(element)
print("The entered Array Elements are:",arr)
for i in range(n):
    sum =sum + arr[i]
print("The sum of Array Elements are:",sum)
