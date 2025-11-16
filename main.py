# face recognition part II
#IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# Add GUI imports
import tkinter as tk
from tkinter import filedialog

#INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

# Define name mapping
name_mapping = {
    0: 'Abhishek', # Assuming '0' is the label for Barnes, adjust if labels are 1-based
    1: 'Barnes',
    2: 'Madhav',
    3: 'Mrinmoy'
}

haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

def recognize_faces_in_image(image_path):
    img = cv.imread(image_path)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x, y, w, h in faces:
        face_img = rgb_img[y:y+h, x:x+w]
        face_img = cv.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)
        ypred = facenet.embeddings(face_img)
        face_probabilities = model.predict_proba(ypred)[0]
        max_probability = np.max(face_probabilities)
        predicted_class_index = np.argmax(face_probabilities)
        
        # Set a confidence threshold
        confidence_threshold = 0.7
        
        if max_probability > confidence_threshold:
            face_name_encoded = encoder.inverse_transform([predicted_class_index])[0]
            final_name = name_mapping.get(face_name_encoded, "Unknown")
        else:
            final_name = "Not Found"
            
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv.putText(img, final_name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow("Uploaded Image Face Recognition", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        recognize_faces_in_image(file_path)

def start_webcam():
    cap = cv.VideoCapture(0)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
            for x, y, w, h in faces:
                face_img = rgb_img[y:y+h, x:x+w]
                face_img = cv.resize(face_img, (160, 160))
                face_img = np.expand_dims(face_img, axis=0)
                ypred = facenet.embeddings(face_img)
                face_probabilities = model.predict_proba(ypred)[0]
                max_probability = np.max(face_probabilities)
                predicted_class_index = np.argmax(face_probabilities)

                # Set a confidence threshold
                confidence_threshold = 0.7

                if max_probability > confidence_threshold:
                    face_name_encoded = encoder.inverse_transform([predicted_class_index])[0]
                    final_name = name_mapping.get(str(face_name_encoded), "Unknown")
                else:
                    final_name = "Not Found"

                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
                cv.putText(frame, final_name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 3, cv.LINE_AA)

            cv.imshow("Face Recognition:", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Webcam stream interrupted by user.")
    finally:
        cap.release()
        cv.destroyAllWindows()

# GUI setup
root = tk.Tk()
root.title("Face Recognition")

btn_webcam = tk.Button(root, text="Start Webcam", command=start_webcam)
btn_webcam.pack(pady=10)

btn_upload = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload.pack(pady=10)

root.mainloop()