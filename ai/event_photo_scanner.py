
import cv2
import os
from insightface.app import FaceAnalysis


# ==========================================
# 1. LOAD AI MODEL
# ==========================================

print("Loading AI model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("Model loaded successfully.\n")


# ==========================================
# 2. EVENT PHOTO FOLDER
# ==========================================

event_folder = "event_photos"


# ==========================================
# 3. SUPPORTED IMAGE TYPES
# ==========================================

supported_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


# ==========================================
# 4. CHECK FOLDER
# ==========================================

if not os.path.exists(event_folder):

    print("❌ Event photo folder not found.")

    exit()


# ==========================================
# 5. GET ALL PHOTOS
# ==========================================

photos = []

for filename in os.listdir(event_folder):

    if filename.lower().endswith(supported_extensions):

        photos.append(filename)


print("Photos found:", len(photos))
print()


# ==========================================
# 6. PROCESS EVERY PHOTO
# ==========================================

total_faces = 0


for filename in photos:

    image_path = os.path.join(
        event_folder,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:

        print("❌ Could not read:", filename)

        continue


    # Detect faces
    faces = app.get(image)


    print(
        f"{filename} → "
        f"{len(faces)} face(s)"
    )


    total_faces += len(faces)


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n================================")
print("SCAN COMPLETE")
print("================================")

print("Total photos :", len(photos))
print("Total faces  :", total_faces)