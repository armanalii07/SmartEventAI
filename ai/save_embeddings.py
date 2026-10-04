import cv2
import os
import pickle
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
# 2. FOLDERS
# ==========================================

event_folder = "event_photos"

output_file = "face_database.pkl"


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
# 4. CHECK EVENT FOLDER
# ==========================================

if not os.path.exists(event_folder):

    print("❌ Event photo folder not found.")

    exit()


# ==========================================
# 5. FACE DATABASE
# ==========================================

face_database = []


# ==========================================
# 6. PROCESS PHOTOS
# ==========================================

for filename in os.listdir(event_folder):

    if not filename.lower().endswith(
        supported_extensions
    ):
        continue


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


    # ======================================
    # SAVE EACH FACE
    # ======================================

    for face_number, face in enumerate(faces):

        embedding = face.embedding


        face_record = {

            "photo": filename,

            "face_number": face_number + 1,

            "embedding": embedding

        }


        face_database.append(face_record)


# ==========================================
# 7. SAVE DATABASE
# ==========================================

with open(
    output_file,
    "wb"
) as file:

    pickle.dump(
        face_database,
        file
    )


# ==========================================
# 8. SUMMARY
# ==========================================

print("\n================================")
print("EMBEDDING DATABASE CREATED")
print("================================")

print(
    "Photos processed :",
    len(set(
        record["photo"]
        for record in face_database
    ))
)

print(
    "Faces stored     :",
    len(face_database)
)

print(
    "Database file    :",
    output_file
)
