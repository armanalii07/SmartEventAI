import cv2
import os

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load image
image = cv2.imread("test.jpg")

if image is None:
    print("❌ Could not load image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print("Faces detected:", len(faces))

# Create folder for cropped faces
os.makedirs("cropped_faces", exist_ok=True)

# Crop each face
for index, (x, y, width, height) in enumerate(faces):

    face = image[y:y + height, x:x + width]

    filename = f"cropped_faces/face_{index + 1}.jpg"

    cv2.imwrite(filename, face)

    print("Saved:", filename)

# Display original image
cv2.imshow("Original Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()