a=int(input("a:"))    #get value 1 from user
b=int(input("b:"))    #get value 2 from user
print(f"sum of {a} and {b} is {a+b}")    #add value 1 and 2 and print it
print(f"subraction of {a} and {b} is {a-b}")    #subract the value 1 and 2 and print it
print(f"multiplication  of {a} and {b} is {a*b}")    #multiply the value 1 and 2 and print it
try:    #this may occuerd error so use exception handling
    print(f"division of {a} and {b} is {a/b}")    #divide the value 1 and 2 and print it if valied input like 10/2
except Exception as e:
    print("division by 0 is impossible")    #return the error as message if not valied input like 10/0
