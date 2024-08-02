import sqlite3

# connect to sqlite
connection = sqlite3.connect("student.db")

# Create a cursor object to insert record, create table
cursor = connection.cursor()

# create the table
table_info = """
create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25), 
SECTION VARCHAR(25), MARKS INT)
"""

cursor.execute(table_info)

# INSERT some more records
cursor.execute('''INSERT INTO STUDENT values('krish', 'Data Science', 'A', 90)''')
cursor.execute('''INSERT INTO STUDENT values('John', 'Data Science', 'B', 100)''')
cursor.execute('''INSERT INTO STUDENT values('Mukesh', 'Data Science', 'A', 86)''')
cursor.execute('''INSERT INTO STUDENT values('Jacob', 'DevOps', 'A', 50)''')
cursor.execute('''INSERT INTO STUDENT values('Dipesh', 'DevOps', 'A', 35)''')

# Display all the records
print("The inserted records are")
data = cursor.execute("""select * from STUDENT """)
for row in data:
    print(row)

# Commit your changes in the database
connection.commit()
connection.close()
