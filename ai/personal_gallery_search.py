import cv2
import numpy as np
import pickle


# ==========================================
# 1. LOAD AI MODEL
# ==========================================

from insightface.app import FaceAnalysis

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
# 2. LOAD FACE DATABASE
# ==========================================

database_file = "face_database.pkl"

with open(database_file, "rb") as file:

    face_database = pickle.load(file)


print(
    "Face database loaded."
)

print(
    "Total stored faces:",
    len(face_database)
)


# ==========================================
# 3. COSINE SIMILARITY
# ==========================================

def cosine_similarity(
    embedding1,
    embedding2
):

    return np.dot(
        embedding1,
        embedding2
    ) / (
        np.linalg.norm(embedding1)
        *
        np.linalg.norm(embedding2)
    )


# ==========================================
# 4. GET USER FACE EMBEDDING
# ==========================================

def get_user_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:

        print(
            "❌ Could not load:",
            image_path
        )

        return None


    faces = app.get(image)


    print(
        f"\nUser photo → "
        f"{len(faces)} face(s) found"
    )


    if len(faces) == 0:

        print(
            "❌ No face found."
        )

        return None


    if len(faces) > 1:

        print(
            "⚠️ Multiple faces found."
        )

        print(
            "Using the first face."
        )


    return faces[0].embedding


# ==========================================
# 5. USER SELFIE
# ==========================================

query_image = (
    "test_same_person/arman_1.jpg"
)


user_embedding = get_user_embedding(
    query_image
)


if user_embedding is None:

    exit()


# ==========================================
# 6. MATCHING THRESHOLD
# ==========================================

# Temporary threshold for testing.
#
# We will NOT use this as the
# final production threshold.

MATCH_THRESHOLD = 0.60


# ==========================================
# 7. SEARCH DATABASE
# ==========================================

matching_photos = {}


print(
    "\nSearching event photos..."
)


for record in face_database:

    photo = record["photo"]

    face_number = record["face_number"]

    stored_embedding = record["embedding"]


    # --------------------------------------
    # Don't return the user's own query photo
    # --------------------------------------

    if photo == "arman_1.jpg":

        continue


    # --------------------------------------
    # Calculate similarity
    # --------------------------------------

    score = cosine_similarity(
        user_embedding,
        stored_embedding
    )


    # --------------------------------------
    # Check threshold
    # --------------------------------------

    if score >= MATCH_THRESHOLD:

        # If the same photo contains
        # multiple matching faces,
        # keep only the highest score.

        if (
            photo not in matching_photos
            or
            score > matching_photos[photo]["score"]
        ):

            matching_photos[photo] = {

                "score": score,

                "face_number": face_number

            }


# ==========================================
# 8. SORT MATCHING PHOTOS
# ==========================================

sorted_photos = sorted(
    matching_photos.items(),
    key=lambda item: item[1]["score"],
    reverse=True
)


# ==========================================
# 9. DISPLAY PERSONAL GALLERY
# ==========================================

print(
    "\n========================================"
)

print(
    "YOUR PERSONAL GALLERY"
)

print(
    "========================================\n"
)


if len(sorted_photos) == 0:

    print(
        "No matching photos found."
    )

else:

    for index, (
        photo,
        information
    ) in enumerate(
        sorted_photos,
        start=1
    ):

        print(
            f"{index}. {photo}"
        )

        print(
            f"   Face: "
            f"{information['face_number']}"
        )

        print(
            f"   Similarity: "
            f"{information['score']:.4f}"
        )

        print()


# ==========================================
# 10. SUMMARY
# ==========================================

print(
    "========================================"
)

print(
    "Total matching photos:",
    len(sorted_photos)
)

print(
    "========================================"
)
