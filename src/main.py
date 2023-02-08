import cv2
import pathlib

face_path = pathlib.Path(cv2.__file__).parent.absolute() / 'data' / 'haarcascade_frontalface_default.xml'
face = cv2.CascadeClassifier(str(face_path))
eye_path = pathlib.Path(cv2.__file__).parent.absolute() / 'data' / 'haarcascade_eye.xml'
eye = cv2.CascadeClassifier(str(eye_path))
mouth_path = pathlib.Path(cv2.__file__).parent.absolute() / 'data' / 'haarcascade_mcs_mouth.xml'
mouth = cv2.CascadeClassifier(str(mouth_path))
camera = cv2.VideoCapture(0)

while True:
    _,frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(
        gray,
        scaleFactor=1.03,
        minNeighbors=8,
        minSize=(40, 40),
        flags =cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.ellipse(frame, (int(x+w/2), int(y+h/2)), (int(w/5*3), int(h/5*3)), 0, 0, 360, (255, 192, 203), 2)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye.detectMultiScale(
            roi_gray,
            scaleFactor = 1.03,
            minNeighbors = 8,
        )
        
        for (lx, ly, lw, lh) in eyes:
            cv2.ellipse(frame, (int(x+lx+lw/2), int(y+ly+lh/2)), (int(lw/3), int(lh/3)), 0, 0, 360, (128, 0, 0), 2)
            #cv2.rectangle(roi_color, (lx, ly), (lx+lw, ly+lh), (128, 0, 0), 2)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()