# daily_report.py

import psycopg2
import pandas as pd
from datetime import datetime
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

# Generate daily report
def generate_daily_report():
    conn = connect_db()
    cur = conn.cursor()
    today = datetime.now().date()

    query = '''
    SELECT s.name AS subject_name, stu.name AS student_name, a.image AS image_path
    FROM attendance a
    JOIN subject s ON a.subject_id = s.id
    JOIN student stu ON a.student_id = stu.id
    WHERE a.date = %s
    '''
    
    cur.execute(query, (today,))
    data = cur.fetchall()

    report = {}
    for subject, student, image_path in data:
        if subject not in report:
            report[subject] = []
        report[subject].append((student, image_path))

    cur.close()
    conn.close()

    # Convert to DataFrame for CSV generation
    report_data = {k: [stu for stu, _ in v] for k, v in report.items()}
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in report_data.items()]))
    df.to_csv('daily_report.csv', index=False)

    return report

# Send email with attachment
def send_email(subject, body, recipient, attachment, filename, image_path):
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = recipient
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    attachment_part = MIMEApplication(attachment.to_csv(index=False), _subtype="csv")
    attachment_part.add_header('Content-Disposition', 'attachment', filename=filename)
    msg.attach(attachment_part)

    with open(image_path, 'rb') as img_file:
        img_part = MIMEApplication(img_file.read(), _subtype="jpeg")
        img_part.add_header('Content-Disposition', 'attachment', filename='math.jpeg')
        msg.attach(img_part)

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
    report = generate_daily_report()
    faculty_emails = fetch_faculty_emails()
    subject = f"Daily Attendance Report - {datetime.now().date()}"
    body = "Please find attached the daily attendance report and an image."

    for email in faculty_emails:
        # For simplicity, attaching the first image from the report
        # In a real scenario, you might want to customize this per faculty
        image_path = None
        for subject_name, records in report.items():
            if records:  # If there are any records for this subject
                _, image_path = records[0]
                break
        if not image_path:
            continue  # No records found, skip this email

        recipients = [email]
        attachment = pd.read_csv('daily_report.csv')
        filename = 'daily_report.csv'
        send_email(subject, body, recipients[0], attachment, filename, image_path)

if __name__ == "__main__":
    main()