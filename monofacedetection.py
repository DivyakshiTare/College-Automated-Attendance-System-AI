# monofacedetection.py

import pandas as pd
import ast
import numpy as np
from deepface import DeepFace

# Function to normalize a vector
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

# Function to compute cosine similarity
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2)

# Function to recognize a single face
def recognize_mono_face(image_path, embeddings_csv):
    # Read embeddings CSV
    df = pd.read_csv(embeddings_csv)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)  # Convert string representation of list back to list

    # Normalize stored embeddings
    df['embedding'] = df['embedding'].apply(lambda x: normalize_vector(np.array(x)))

    # Get the embedding for the test image
    try:
        test_embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        test_embedding = normalize_vector(test_embedding)
    except Exception as e:
        print(f"Error in generating embedding for the test image: {e}")
        return None

    # Calculate cosine similarity
    df['similarity'] = df['embedding'].apply(lambda emb: cosine_similarity(test_embedding, emb))

    # Debug: Print the similarities
    print(df[['label', 'similarity']])

    # Find the most similar embedding
    recognized_image = df.loc[df['similarity'].idxmax()]

    return recognized_image['label']

# Path to test image
test_image_path = "face_dataset\Testing_dataset\monofaced_images\child1.jpeg"

# Recognize the face
recognized_label = recognize_mono_face(test_image_path, 'student_embedding.csv')
if recognized_label:
    print(f"The recognized person is: {recognized_label}")
else:
    print("Failed to recognize the person.")
