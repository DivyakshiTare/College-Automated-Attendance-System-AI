#setup_database.py

import psycopg2

# Database connection parameters
dbname = "attendance_system"    #your database name
user = "postgres"
password = "postgresql09"    #your own password
host = "localhost"

# Connect to the PostgreSQL server
conn = psycopg2.connect(
    dbname=dbname,
    user=user,
    password=password,
    host=host
)

cur = conn.cursor()

# Create the Student table
cur.execute('''
CREATE TABLE IF NOT EXISTS Student (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);
''')

# Create the Faculty table
cur.execute('''
CREATE TABLE IF NOT EXISTS Faculty (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);
''')

# Create the Subject table
cur.execute('''
CREATE TABLE IF NOT EXISTS Subject (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    from_time TIME,
    to_time TIME,
    faculty_id INT,
    FOREIGN KEY (faculty_id) REFERENCES Faculty (id)
);
''')

# Create the Attendance table
cur.execute('''
CREATE TABLE IF NOT EXISTS Attendance (
    id SERIAL PRIMARY KEY,
    date DATE,
    subject_id INT,
    student_id INT,
    image VARCHAR(255),
    FOREIGN KEY (subject_id) REFERENCES Subject (id),
    FOREIGN KEY (student_id) REFERENCES Student (id)
);
''')

# Commit changes and close the connection
conn.commit()
cur.close()
conn.close()

print("Tables created successfully")
