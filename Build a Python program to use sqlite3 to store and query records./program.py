import sqlite3
import pandas as pd
import os

DB_PATH = "employees.db" 

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    salary REAL NOT NULL,
    hired DATE DEFAULT (DATE('now'))
)
""")

employees = [
    ("Asha", "Engineer", 45000.0),
    ("Ravi", "Data Analyst", 38000.0),
    ("Maya", "Designer", 32000.0),
    ("Karan", "Engineer", 47000.0),
    ("Leela", "HR", 30000.0)
]
cur.executemany("INSERT INTO employees (name, role, salary) VALUES (?, ?, ?)", employees)
conn.commit()

print(pd.read_sql_query("SELECT * FROM employees", conn))

threshold = 35000.0
print(pd.read_sql_query("SELECT id, name, role, salary FROM employees WHERE salary > ?", conn, params=(threshold,)))

cur.execute("UPDATE employees SET salary = salary * 1.10 WHERE name = ?", ("Maya",))
conn.commit()
print(pd.read_sql_query("SELECT * FROM employees WHERE name = 'Maya'", conn))

cur.execute("DELETE FROM employees WHERE name = ?", ("Leela",))
conn.commit()
print(pd.read_sql_query("SELECT * FROM employees", conn))

try:
    conn.execute("BEGIN")
    conn.execute("INSERT INTO employees (name, role, salary) VALUES (?, ?, ?)", ("Sam", "Intern", 15000))
    conn.execute("INSERT INTO nonexistent (x) VALUES (1)")
    conn.commit()
except Exception as e:
    conn.rollback()
    print("Transaction failed and rolled back. Exception:", e)

print(pd.read_sql_query("SELECT * FROM employees", conn))
conn.close()
