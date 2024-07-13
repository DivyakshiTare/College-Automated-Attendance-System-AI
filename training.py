# training.py

import os
import pandas as pd
from deepface import DeepFace

# Function to get image paths and labels
def get_image_paths(training_dir):
    image_paths = []
    labels = []
    for label in os.listdir(training_dir):
        student_dir = os.path.join(training_dir, label)
        if os.path.isdir(student_dir):
            for image_name in os.listdir(student_dir):
                if image_name.endswith('.jpeg'):  # Ensure the file is a JPEG image
                    image_path = os.path.join(student_dir, image_name)
                    if os.path.isfile(image_path):
                        image_paths.append(image_path)
                        labels.append(label)
    return image_paths, labels

# Training Directory
training_dir = r"face_dataset\Training_dataset"
image_paths, labels = get_image_paths(training_dir)

print("Total Images:", len(image_paths))

# Training the Model (Creating embeddings)
data = {'image_path': image_paths, 'label': labels}
df = pd.DataFrame(data)

# Function to get embedding for an image
def get_embedding(img_path):
    try:
        embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
    except Exception as e:
        print(f"Error extracting embedding for {img_path}: {str(e)}")
        embedding = []
    return embedding

# Apply the function to each image path
df['embedding'] = df['image_path'].apply(get_embedding)

# Save to CSV
output_csv = 'student_embedding.csv'
df.to_csv(output_csv, index=False)

print(f"Embeddings saved to {output_csv}")
