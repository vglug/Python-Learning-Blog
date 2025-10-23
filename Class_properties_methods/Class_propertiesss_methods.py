class Student:
        def __init__(self, name, roll_no, marks):
        self.name = name
        self.roll_no = roll_no
        self.marks = marks


    def display_details(self):
        print(f"Name: {self.name}")
        print(f"Roll No: {self.roll_no}")
        print(f"Marks: {self.marks}")


    def calculate_grade(self):
        if self.marks >= 90:
            grade = "A+"
        elif self.marks >= 80:
            grade = "A"
        elif self.marks >= 70:
            grade = "B"
        elif self.marks >= 60:
            grade = "C"
        elif self.marks >= 35:
            grade = "D"
        else:
            grade = "Fail"
        return grade

    def show_result(self):
        grade = self.calculate_grade()
        print(f"Student {self.name} has secured Grade: {grade}\n")

student1 = Student("Ranjith", 101, 92)
student2 = Student("Rajarajan", 102, 76)
student3 = Student("Rajesh", 103, 58)


student1.display_details()
student1.show_result()

student2.display_details()
student2.show_result()

student3.display_details()
student3.show_result()
