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
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags =cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        eyes = eye.detectMultiScale(
            gray[y:y+h, x:x+w],
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags =cv2.CASCADE_SCALE_IMAGE
        )
        for (lx, ly, lw, lh) in eyes:
            cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 2)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()