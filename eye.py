import cv2
#load the Haar Cascade for eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
#read the image
image = cv2.imread("image1.jpg")
#convert the color
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#detect the faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#rectangle around the faces and eyes
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#display the image
cv2.imshow("Eye Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()