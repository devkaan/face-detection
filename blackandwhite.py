import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)    
    faces = faceCascade.detectMultiScale(
        blackAndWhiteImage,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(blackAndWhiteImage, (x, y), (x+w, y+h), (0, 200, 0), 2)

    # Display the resulting blackAndWhiteImage
    flippedFrame = cv2.flip(blackAndWhiteImage,1)
    cv2.imshow('Video', flippedFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
