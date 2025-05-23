import cv2
import numpy as np
import threading
from ultralytics import YOLO
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load YOLO model
model = YOLO("best.pt")  # Replace with your trained model

# Function to classify mental disorder based on detected emotion
def map_emotion_to_disorder(emotion):
    if emotion in ['fear', 'surprise']:
        return 'Anxious'
    elif emotion in ['sad', 'angry', 'disgust']:
        return 'Depressed'
    else:
        return 'Normal'

# Initialize Tkinter
root = tk.Tk()
root.title("Emotion & Mental Disorder Detection")

# UI Elements
label_video = tk.Label(root)
label_video.pack()

label_emotion = tk.Label(root, text="Emotion: ", font=("Arial", 16))
label_emotion.pack()

label_disorder = tk.Label(root, text="Mental Disorder: ", font=("Arial", 16))
label_disorder.pack()

def process_frame(frame):
    """Detect faces, analyze emotions, and classify disorder in a frame."""
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy  # Extract bounding box coordinates

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                try:
                    # Analyze emotions using DeepFace
                    analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                    detected_emotion = analysis[0]['dominant_emotion']

                    # Map emotion to mental disorder
                    mental_disorder = map_emotion_to_disorder(detected_emotion)

                    # Update UI
                    label_emotion.config(text=f"Emotion: {detected_emotion}")
                    label_disorder.config(text=f"Mental Health: {mental_disorder}")

                    # Draw bounding box and text on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{detected_emotion}, {mental_disorder}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                except Exception as e:
                    print(f"Emotion detection error: {e}")

    return frame

def select_image():
    """Allow user to upload an image and process it."""
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = process_frame(image)

        img = Image.fromarray(processed_image)
        img_tk = ImageTk.PhotoImage(image=img)

        label_video.img_tk = img_tk
        label_video.config(image=img_tk)

def start_webcam():
    """Start real-time webcam processing in a new thread."""
    def webcam_thread():
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame)

            # Convert to Tkinter-compatible format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)

            label_video.img_tk = img_tk
            label_video.config(image=img_tk)
            root.update_idletasks()

        cap.release()
        cv2.destroyAllWindows()

    thread = threading.Thread(target=webcam_thread)
    thread.daemon = True
    thread.start()

# Buttons
btn_webcam = tk.Button(root, text="Use Webcam", command=start_webcam, font=("Arial", 14))
btn_webcam.pack()

btn_upload = tk.Button(root, text="Upload Image", command=select_image, font=("Arial", 14))
btn_upload.pack()

# Run Tkinter UI loop
root.mainloop()
