import cv2
import numpy as np
import pyautogui

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Error loading Haar cascade files")
    exit()
def move_cursor(x, y):
    screen_width, screen_height = pyautogui.size()
    x = np.clip(x, 0, screen_width)
    y = np.clip(y, 0, screen_height)
    pyautogui.moveTo(x, y)

def detect_blink(eye_region):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    white_area = cv2.countNonZero(threshold_eye)
    return white_area < 100

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error accessing webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(f"Faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        print(f"Eyes detected: {len(eyes)}")

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eye_center = (x + ex + ew//2, y + ey + eh//2)
            move_cursor(eye_center[0] * 2, eye_center[1] * 2)

            eye_region = roi_color[ey:ey+eh, ex:ex+ew]
            if detect_blink(eye_region):
                pyautogui.click()

    cv2.imshow('Eye Cursor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
