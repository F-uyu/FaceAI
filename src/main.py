import cv2
import pathlib

face_path = pathlib.Path(cv2.__file__).parent.absolute() / 'data' / 'haarcascade_frontalface_default.xml'
face = cv2.CascadeClassifier(str(face_path))
eye_path = pathlib.Path(cv2.__file__).parent.absolute() / 'data' / 'haarcascade_eye.xml'
eye = cv2.CascadeClassifier(str(eye_path))
camera = cv2.VideoCapture(0)

while True:
    _,frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=7,
        minSize=(40, 40),
        flags =cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye.detectMultiScale(
            roi_gray
        )
        
        for (lx, ly, lw, lh) in eyes:
            cv2.rectangle(roi_color, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 2)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()