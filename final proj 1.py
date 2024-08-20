


import cv2
import numpy as np
from playsound import playsound
import pyaudio
import threading
import struct

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

# import cascade file for facial recognition
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

def play_buzzer():
  playsound("C:/Users/TUSHAR JADHAV/Downloads/incorrect-buzzer-sound-147336.mp3")  # Replace "buzzer.wav" with your buzzer sound file path

def sound_detection():
  CHUNK = 1024
  FORMAT = pyaudio.paInt16
  CHANNELS = 2
  RATE = 44100
  THRESHOLD = 1000

  p = pyaudio.PyAudio()
  stream = p.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)

  while True:
    data = stream.read(CHUNK)
    data_int = struct.unpack(str(2*CHUNK) + 'h', data)  # Modified to 2*CHUNK
    rms = np.sqrt(np.mean(np.array(data_int) ** 2))
    if rms > THRESHOLD:
      cv2.putText(img, "Don't talk!", (50, 50), font, 1, (0, 0, 255), 2)
      play_buzzer()

  stream.stop_stream()
  stream.close()
  p.terminate()

sound_thread = threading.Thread(target=sound_detection)
sound_thread.daemon = True
sound_thread.start()

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Getting corners around the face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  

    # drawing bounding box around face
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # detecting eyes
    eyes = eyeCascade.detectMultiScale(imgGray)
    # drawing bounding box for eyes
    for (ex, ey, ew, eh) in eyes:
        img = cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

    mouth_rects = mouth_cascade.detectMultiScale(imgGray, 1.7, 11)
    for (x, y, w, h) in mouth_rects:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

    if(len(faces)== 0 or len(mouth_rects)==1):
        text = "ERROR"
        coordinates = (20,40)
        color = (0,0,255)
        thickness = 2
        img = cv2.putText(img, text, coordinates, font, 1, color, thickness, cv2.LINE_AA)
        img = cv2.rectangle(img, (0, 0), (img.shape[1],img.shape[0] ), (0, 0, 255), 2)

    if len(faces) > 1:
        play_buzzer()  # Play buzzer sound
        cv2.putText(img, "Don't cheat! Exam is on!", (50, 50), font, 1, (0, 0, 255), 2)

    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow('face_detect')





