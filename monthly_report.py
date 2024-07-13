# monthly_report.py

import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import smtplib

# Database connection parameters
dbname = "attendance_system"
user = "postgres"
password = "postgresql09"
host = "localhost"

# Email credentials
email = 'support@aptpath.in'
email_password = 'btpdcnfkgjyzdndh'
smtp_host = 'smtp.office365.com'
smtp_port = 587

# Connect to PostgreSQL
def connect_db():
    print(f"Connecting to database: dbname={dbname}, user={user}, host={host}")
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
        print("Connection successful")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Connection failed: {e}")
        raise

# Generate monthly report
def generate_monthly_report():
    conn = connect_db()
    cur = conn.cursor()
    first_day = datetime.now().replace(day=1)
    last_day = (first_day + timedelta(days=32)).replace(day=1) - timedelta(days=1)

    query = '''
    SELECT stu.name AS student_name, s.name AS subject_name, COUNT(a.id) AS total_classes,
           SUM(CASE WHEN a.date IS NOT NULL THEN 1 ELSE 0 END) AS present,
           COUNT(a.id) - SUM(CASE WHEN a.date IS NOT NULL THEN 1 ELSE 0 END) AS absent,
           (SUM(CASE WHEN a.date IS NOT NULL THEN 1 ELSE 0 END) * 100.0) / COUNT(a.id) AS attendance_percentage
    FROM attendance a
    JOIN subject s ON a.subject_id = s.id
    JOIN student stu ON a.student_id = stu.id
    WHERE a.date BETWEEN %s AND %s
    GROUP BY stu.name, s.name
    '''
    cur.execute(query, (first_day, last_day))
    data = cur.fetchall()

    columns = ['student_name', 'subject_name', 'total_classes', 'present', 'absent', 'attendance_percentage']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('monthly_report.csv', index=False)

# Send email with attachment
def send_email(subject, body, recipients, attachment, filename):
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    attachment_part = MIMEApplication(attachment.to_csv(index=False), _subtype="csv")
    attachment_part.add_header('Content-Disposition', 'attachment', filename=filename)
    msg.attach(attachment_part)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(email, email_password)
        server.send_message(msg)

# Fetch faculty emails
def fetch_faculty_emails():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT email FROM faculty")
    emails = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return emails

# Main function
def main():
    generate_monthly_report()
    faculty_emails = fetch_faculty_emails()
    subject = f"Monthly Attendance Report - {datetime.now().strftime('%B %Y')}"
    body = "Please find attached the monthly attendance report."
    recipients = faculty_emails
    attachment = pd.read_csv('monthly_report.csv')
    filename = 'monthly_report.csv'
    send_email(subject, body, recipients, attachment, filename)

if __name__ == "__main__":
    main()
