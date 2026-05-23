import cv2

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



#read the image

image = cv2.imread("image1.jpg")

#convert the color
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect the faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

#display the image
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
