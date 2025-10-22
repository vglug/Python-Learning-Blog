class assignment():    #create a base class for overriding
    def __init__(s,name,n):    #it is assign the value
        s.name=name
        s.n=n
        s.submit()
    def submit(s):    #overriding function
         pass
    def mark(s):    #overriding function
        pass
class dbmsAssignment(assignment):    #child class of assignment
    def submit(s):    #overriding function
        s.Ano=[]    #list for give marks
        for i in range(s.n):
            s.Ano.append(i)    #append the assignmet no in list 
        print(f"{s.name} is submited {type(s).__name__[:4]}  assignment {s.n}")    #print the student is submitted the assignment
        s.mark()    # call the abstract method as overriding function
    def mark(s):    #overriding function and it assign the mark depend submitted assignment
        if len(s.Ano)==1:
            print("mark:12")
        elif len(s.Ano)==2:
            print("mark:18")
        elif len(s.Ano)==3:
            print("mark:20")
        elif len(s.Ano)==4:
            print("mark:22")
        elif len(s.Ano)==5:
            print("mark:24")
class osAssignment(assignment):    #child class of assignment
    def submit(s):     #overriding function
        s.Ano=[]
        for i in range(s.n):
            s.Ano.append(i)    #append the assignmet no in list
        print(f"{s.name} is submited {type(s).__name__[:2]}  assignment {s.n}")    #print the student is submitted the assignment
        s.mark()    # call the abstract method as overriding function
    def mark(s):    #overriding function and it assign the mark depend submitted assignment
        if len(s.Ano)==1:
            print("mark:12")
        elif len(s.Ano)==2:
            print("mark:18")
        elif len(s.Ano)==3:
            print("mark:20")
        elif len(s.Ano)==4:
            print("mark:22")
        elif len(s.Ano)==5:
            print("mark:24")

#object for assignments
s1=dbmsAssignment("praveenkanth",1)
s1=osAssignment("praveenkanth",1)
s1=dbmsAssignment("praveenkanth",2)
s1=dbmsAssignment("praveenkanth",3)
s1=osAssignment("praveenkanth",2)
s1=dbmsAssignment("praveenkanth",4)
s1=osAssignment("praveenkanth",3)
s1=osAssignment("praveenkanth",4)
s1=dbmsAssignment("praveenkanth",5)
s1=osAssignment("praveenkanth",5)
