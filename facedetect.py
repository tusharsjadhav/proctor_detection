import cv2
from IPython.display import Video
from IPython.core.display import clear_output

# Replace with the path to your pre-trained face cascade classifier (download from OpenCV or a reliable source)
face_cascade = cv2.CascadeClassifier("C:\Users\TUSHAR JADHAV\Downloads")

def face_detection_function(video_source=0):
    """
    This function captures video from the specified source (default webcam)
    and performs real-time face detection with a pre-trained cascade classifier.
    Displays the video with detected faces within the Jupyter Notebook.
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

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue rectangle around faces

        # Display the resulting frame with detected faces within the notebook
        clear_output(wait=True)  # Clear previous output for smooth video display
        cv2.imshow('Video', frame)

        # Process keyboard input (optional)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Run the function
face_detection_function()
