import csv

total_marks = {}

with open("students.csv", "r") as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        name = row["Name"]
        marks = int(row["Marks"])
        
        
        if name in total_marks:
            total_marks[name] += marks
        else:
            total_marks[name] = marks

#print output
for name, marks in total_marks.items():
    print(name, "=", marks)
