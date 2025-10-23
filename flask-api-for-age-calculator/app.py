from flask import Flask, render_template, request, flash, redirect,url_for
from datetime import date

app = Flask(__name__)
app.secret_key='my_super_key1728'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    dob = request.form.get('dob')
    print(dob)
    #2025-11-22
    #2005-05-28
    year=dob[:4]
    month=dob[5:7]
    date=dob[-2:]
    return redirect(url_for('calc', by=year,bm=month,bd=date))

@app.route('/calc')
def calc():
    by=request.args.get('by')
    bm=request.args.get('bm')  
    bd=request.args.get('bd')
    
    today = date.today()
    td=today.day
    tm=today.month
    ty=today.year
    
    if ty<int(by):
        msg='please give valied year'
        return render_template('index.html',msg=msg)
    yob=ty-int(by)
    
    if tm<int(bm):
        yob-=1
        tm+=12
    mob=tm-int(bm)
    
    if td<int(bd):
        mob-=1
        td+=30
    dob=td-int(bd)
    
   
    result=f"you age {yob} years, {mob} months, {dob} days."
    return render_template('index.html',result=result)

app.run(debug=True)
