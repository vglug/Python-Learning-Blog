import sqlite3


conn = sqlite3.connect("students.db")


cursor = conn.cursor()


cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        grade TEXT
    )
''')


cursor.execute("INSERT INTO students (name, age, grade) VALUES (?, ?, ?)", ("Alice", 20, "A"))
cursor.execute("INSERT INTO students (name, age, grade) VALUES (?, ?, ?)", ("Bob", 22, "B"))
cursor.execute("INSERT INTO students (name, age, grade) VALUES (?, ?, ?)", ("Charlie", 19, "A"))


conn.commit()


cursor.execute("SELECT * FROM students")
rows = cursor.fetchall()


print("Student Records:")
for row in rows:
    print(f"ID: {row[0]}, Name: {row[1]}, Age: {row[2]}, Grade: {row[3]}")


conn.close()
