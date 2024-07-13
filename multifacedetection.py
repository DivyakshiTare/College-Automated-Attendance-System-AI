# multifacedetection.py

import os
import numpy as np
import pandas as pd
from deepface import DeepFace
import ast
import cv2
from datetime import datetime
import psycopg2

# Database connection parameters
dbname = "attendance_system"
user = "postgres"
password = "postgresql09"
host = "localhost"

# Function to normalize a vector
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

# Function to compute cosine similarity
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2)

# Function to extract date and time from image path
def extract_datetime(image_path):
    # Assuming the format of the filename is 'yyyy-mm-dd-hh-mm-ss.millisec_groupX.jpeg'
    filename = os.path.basename(image_path)
    datetime_str = filename.split('_')[0]  # Extract datetime part
    try:
        datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d-%H-%M-%S.%f')  # Convert datetime string to datetime object
    except ValueError:
        print(f"Error parsing datetime from {datetime_str}. Check the format.")
        return None
    return datetime_obj

# Function to get subject based on the time period
def get_subject_by_time(time):
    try:
        # Connect to the PostgreSQL server
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host
        )
        cur = conn.cursor()
        
        # Query to get the subject based on the given time
        cur.execute('''
        SELECT name FROM Subject 
        WHERE from_time <= %s::time AND to_time >= %s::time;
        ''', (time, time))
        
        subject = cur.fetchone()
        cur.close()
        conn.close()
        
        return subject[0] if subject else None

    except Exception as e:
        print(f"Error fetching subject from database: {e}")
        return None

# Function to recognize multifaced image and store attendance information
def recognize_multi_face_and_store(image_path, embeddings_csv, attendance_list):
    try:
        # Load embeddings CSV
        df = pd.read_csv(embeddings_csv)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)  # Convert string representation of list back to list
        df['embedding'] = df['embedding'].apply(lambda x: normalize_vector(np.array(x)))  # Normalize embeddings

        # Check if the image path is valid and the file exists
        if not os.path.isfile(image_path):
            print(f"Failed to load image from {image_path}. Check the file path or integrity.")
            return None, None, None
        
        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image from {image_path}. Check the file path or integrity.")
            return None, None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces using OpenCV Haar Cascade Classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("No faces detected in the image.")
            return None, None, None

        # Extract datetime from image path
        datetime_obj = extract_datetime(image_path)
        if not datetime_obj:
            return None, None, None
        
        time = datetime_obj.time()
        date = datetime_obj.date()

        # Get subject by time period
        subject = get_subject_by_time(time)
        if not subject:
            print("Subject not found for the given time period.")
            return None, None, None

        # Recognize each detected face
        recognized_faces = []
        for (x, y, w, h) in faces:
            face = img_rgb[y:y+h, x:x+w]  # Crop face region
            face_embedding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]['embedding']
            face_embedding = normalize_vector(face_embedding)

            # Calculate similarity with stored embeddings
            df['similarity'] = df['embedding'].apply(lambda emb: cosine_similarity(face_embedding, emb))

            # Find the most similar embedding
            recognized_face = df.loc[df['similarity'].idxmax()]
            recognized_label = recognized_face['label']
            recognized_faces.append(recognized_label)

            # Append attendance information to the list
            if recognized_label:
                attendance_list.append((date, subject, recognized_label, image_path))

        return recognized_faces, subject, date

    except Exception as e:
        print(f"Error recognizing faces: {e}")
        return None, None, None

if __name__ == "__main__":
    # Initialize attendance list
    attendance_list = []

    # Path to multifaced test image
    test_image_path = "face_dataset\\Testing_dataset\\grouped_images\\2024-07-04-09-15-49.277709_group20.jpeg"
    
    # Path to embeddings CSV
    embeddings_csv = 'student_embedding.csv'

    # Recognize faces in the multifaced image and store attendance
    recognized_labels, subject, date = recognize_multi_face_and_store(test_image_path, embeddings_csv, attendance_list)

    # Check if faces are recognized
    if recognized_labels:
        print(f"The recognized persons are: {recognized_labels}")
        print(f"Subject: {subject}, Date: {date}")
    else:
        print("Failed to recognize faces in the image.")
