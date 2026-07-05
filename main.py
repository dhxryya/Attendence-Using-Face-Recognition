import cv2
import os
import numpy as np
import re
from datetime import datetime

def log_attendance(person_name):
    """Logs the recognized person's name into attendance.csv without duplicates."""
    file_name = "attendance.csv"
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write("Name,Date,Time\n")
            
    already_logged = False
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            if f"{person_name},{current_date}" in line:
                already_logged = True
                break
                
    if not already_logged:
        with open(file_name, "a") as f:
            f.write(f"{person_name},{current_date},{current_time}\n")
        print(f"\n[🚀 ATTENDANCE] Logged successfully for: {person_name} at {current_time}")

def register_new_face():
    """Captures 15 face samples from webcam and updates the database."""
    print("\n--- 📸 FACE REGISTRATION MODULE ---")
    username = input("Enter User / Employee Name: ").strip().capitalize()
    if not username or not re.match("^[A-Za-z0-9_]+$", username):
        print("[❌ ERROR] Invalid name! Only alphanumeric characters allowed.")
        return
        
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print("[❌ ERROR] 'haarcascade_frontalface_default.xml' missing in root directory.")
        return
        
    target_dir = "known_faces"
    os.makedirs(target_dir, exist_ok=True)
    
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    print(f"[INFO] Look directly into the camera. Capturing structures for '{username}'...")
    
    sample_count = 0
    while sample_count < 15:
        ret, frame = cam.read()
        if not ret:
            print("[❌ ERROR] Webcam stream failed.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            sample_count += 1
            img_path = os.path.join(target_dir, f"{username.lower()}{sample_count}.jpg")
            cv2.imwrite(img_path, gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
        cv2.imshow("Enrolling Face - Press 'q' to Abort", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
            
    cam.release()
    cv2.destroyAllWindows()
    
    if sample_count >= 15:
        print(f"[✅ SUCCESS] Captured 15 samples for '{username}'. Triggering training network...")
        train_face_model()
    else:
        print("[⚠️ WARNING] Registration pipeline interrupted.")

def train_face_model():
    """Trains the LBPH recognizer using saved images."""
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        known_faces_dir = "known_faces"
        
        if not os.path.exists(known_faces_dir):
            return
            
        image_files = [f for f in os.listdir(known_faces_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        if len(image_files) == 0:
            if os.path.exists('trainer.yml'): os.remove('trainer.yml')
            if os.path.exists('labels.txt'): os.remove('labels.txt')
            print("[INFO] No face data left. Model files cleared.")
            return

        face_samples, ids = [], []
        name_map, current_id = {}, 0
        
        for file_name in image_files:
            raw_name = os.path.splitext(file_name)[0]
            name = re.sub(r'\d+$', '', raw_name).strip().capitalize()
            
            if name not in name_map:
                name_map[name] = current_id
                current_id += 1
                
            gray_img = cv2.imread(os.path.join(known_faces_dir, file_name), cv2.IMREAD_GRAYSCALE)
            face_samples.append(gray_img)
            ids.append(name_map[name])
            
        recognizer.train(face_samples, np.array(ids))
        recognizer.write('trainer.yml')
        
        with open("labels.txt", "w") as f:
            for name, label_id in name_map.items():
                f.write(f"{label_id}:{name}\n")
        print("[⚙️ NETWORK] Model re-compiled and weights synchronized successfully.")
    except Exception as e:
        print(f"[❌ ERROR] Re-training failure: {str(e)}")

def delete_user_data():
    """Purges a user's images from disk and re-trains the system."""
    print("\n--- 🗑️ BIOMETRIC DATA PURGE ENGINE ---")
    username = input("Enter the EXACT name of the user to delete: ").strip().capitalize()
    if not username:
        return
        
    known_faces_dir = "known_faces"
    if not os.path.exists(known_faces_dir):
        print("[❌ ERROR] Database is empty.")
        return

    confirm = input(f"Are you absolutely sure you want to completely wipe out '{username}'? (y/n): ").strip().lower()
    if confirm != 'y':
        print("[INFO] Purge canceled.")
        return

    deleted_count = 0
    for file_name in os.listdir(known_faces_dir):
        raw_name = os.path.splitext(file_name)[0]
        cleaned_file_name = re.sub(r'\d+$', '', raw_name).strip().capitalize()
        
        if cleaned_file_name == username:
            os.remove(os.path.join(known_faces_dir, file_name))
            deleted_count += 1

    if deleted_count == 0:
        print(f"[❌ NOT FOUND] No biometric samples found matching '{username}'.")
        return

    print(f"[✅ SYSTEM PURGED] Successfully wiped {deleted_count} samples for '{username}'.")
    train_face_model()

def start_live_recognition():
    """Launches the core face matching loop."""
    print("\n--- ▶ LAUNCHING LIVE INFERENCE STREAM ---")
    cascade_path = 'haarcascade_frontalface_default.xml'
    
    if not os.path.exists(cascade_path) or not os.path.exists('trainer.yml') or not os.path.exists('labels.txt'):
        print("[❌ ERROR] Core assets ('trainer.yml', 'labels.txt', XML) are missing. Register a face first!")
        return
        
    face_cascade = cv2.CascadeClassifier(cascade_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    name_map = {}
    with open("labels.txt", "r") as f:
        for line in f:
            label_id, name = line.strip().split(":")
            name_map[int(label_id)] = name

    video_capture = cv2.VideoCapture(0)
    print("[INFO] System Active. Press 'q' on the camera window to exit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=8, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            label_id, confidence = recognizer.predict(face_roi)

            if confidence < 65: 
                name = name_map.get(label_id, "Unknown")
                accuracy_percentage = round(100 - confidence)
                confidence_text = f"Verified: {accuracy_percentage}%"
                if name != "Unknown":
                    log_attendance(name)
            else:
                name = "Unknown"
                confidence_text = "Unknown Face"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
            
            # Draw HUD Corners
            length = 20
            cv2.line(frame, (x, y), (x + length, y), color, 4)
            cv2.line(frame, (x, y), (x, y + length), color, 4)
            cv2.line(frame, (x + w, y), (x + w - length, y), color, 4)
            cv2.line(frame, (x + w, y), (x + w, y + length), color, 4)
            cv2.line(frame, (x, y + h), (x + length, y + h), color, 4)
            cv2.line(frame, (x, y + h), (x, y + h - length), color, 4)
            cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, 4)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, 4)

            cv2.putText(frame, str(name), (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence_text), (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.putText(frame, "SECURITY INFERENCE MODE: MAXIMUM STRICT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('Advanced Face Recognition System (CLI Mode)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("\n=============================================")
        print("🛡️  BIOMETRIC ACCESS CONTROL ENGINE (CLI) 🛡️")
        print("=============================================")
        print("1. 📸 Register & Capture New Face")
        print("2. ▶  Launch Live Inference/Attendance Stream")
        print("3. 🗑️  Delete User Biometric Data Block")
        print("4. 🚪 Exit System")
        print("=============================================")
        
        choice = input("Select an option (1-4): ").strip()
        if choice == '1':
            register_new_face()
        elif choice == '2':
            start_live_recognition()
        elif choice == '3':
            delete_user_data()
        elif choice == '4':
            print("\nShutting down security core. Goodbye!")
            break
        else:
            print("[❌] Invalid selection! Please enter 1, 2, 3, or 4.")