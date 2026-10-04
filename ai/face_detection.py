import cv2

# Load the face detection model
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load image
image = cv2.imread("test.jpg")

if image is None:
    print("❌ Could not load image.")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Faces detected:", len(faces))

# Draw rectangle around each detected face
for (x, y, width, height) in faces:

    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

# Show result
cv2.imshow("Smart Event AI - Face Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()