class assignment():
    def __init__(s,name,n):
        s.name=name
        s.n=n
        s.submit()
    def submit(s):
         pass
    def mark(s):
        pass
class dbmsAssignment(assignment):
    def submit(s):
        s.Ano=[]
        for i in range(s.n):
            s.Ano.append(i)
        print(f"{s.name} is submited {type(s).__name__[:4]}  assignment {s.n}")
        s.mark()
    def mark(s):
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
class osAssignment(assignment):
    def submit(s):
        s.Ano=[]
        for i in range(s.n):
            s.Ano.append(i)
        print(f"{s.name} is submited {type(s).__name__[:2]}  assignment {s.n}")
        s.mark()
    def mark(s):
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
        else:
             ()
            
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
