import cv2
import winsound  # Import winsound module for audio

# Replace with the path to your pre-trained cascade classifier files (ensure correct paths and permissions)
face_cascade = cv2.CascadeClassifier("C:\Users\TUSHAR JADHAV\Downloads")

# Font for the warning message
font = cv2.FONT_HERSHEY_SIMPLEX

# Function to play a short buzzer sound
def play_buzzer():
  """
  Plays a short buzzer sound using the winsound module (Windows only).
  """
  winsound.PlaySound("buzzer.wav", winsound.SND_FILENAME)  # Replace "buzzer.wav" with your sound file path

# Previous eye and mouth positions for movement tracking (currently unused)
prev_eye_center = None
prev_mouth_center = None

def face_detection_function(video_source=0):
  """
  This function captures video from the specified source (default webcam)
  and performs real-time face detection with warnings for multiple faces.
  Displays the video with detected faces marked in blue boxes.
  Plays a buzzer sound and displays a warning message if multiple faces are detected.
  """

  cap = cv2.VideoCapture(video_source)

  while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame capture was successful
    if not ret:
      print("Error: Unable to capture frame from video source.")
      break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Check for multiple faces and display warning if necessary
    if len(faces) > 1:
      play_buzzer()  # Play buzzer sound
      cv2.putText(frame, "Don't cheat! Exam is on!", (50, 50), font, 1, (0, 0, 255), 2)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue box for face

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Process keyboard input (optional)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
      break

# Run the function
face_detection_function()

# Release resources
cap.release()
cv2.destroyAllWindows()
