
print("Combining 'break' and 'continue'")
for i in range(1, 11):
    if i == 5:
        print("Skipping 5")
        continue  
    if i == 8:
        print("Stopping at 8")
        break  
    print("Number:", i)
