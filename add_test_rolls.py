from database import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute('''
    INSERT OR IGNORE INTO students 
    (name, roll_number, enrollment_number, college, branch, semester, div)
    VALUES (?, ?, ?, ?, ?, ?, ?)
''', ('Test Student 70', '070', '2402031870001', 'ASOIT', 'CSE', 4, 'A'))

cursor.execute('''
    INSERT OR IGNORE INTO students 
    (name, roll_number, enrollment_number, college, branch, semester, div)
    VALUES (?, ?, ?, ?, ?, ?, ?)
''', ('Test Student 110', '110', '2402031870002', 'ASOIT', 'CSE', 4, 'A'))

conn.commit()
conn.close()
print("Done! Roll 70 and 110 added to ASOIT CSE Sem4 DivA")
