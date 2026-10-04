import cv2
import numpy as np
import pickle
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
# 4. GET USER EMBEDDING
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

        print("❌ No face found.")

        return None

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
# 6. TEMPORARY THRESHOLD
# ==========================================

MATCH_THRESHOLD = 0.60


# ==========================================
# 7. SEARCH DATABASE
# ==========================================

matching_photos = {}


for record in face_database:

    photo = record["photo"]

    stored_embedding = record["embedding"]


    # Don't show the query photo itself
    if photo == "arman_1.jpg":
        continue


    score = cosine_similarity(
        user_embedding,
        stored_embedding
    )


    if score >= MATCH_THRESHOLD:

        # Keep only highest score
        # for each photo

        if (
            photo not in matching_photos
            or
            score > matching_photos[photo]
        ):

            matching_photos[photo] = score


# ==========================================
# 8. SORT PHOTOS
# ==========================================

sorted_photos = sorted(
    matching_photos.items(),
    key=lambda item: item[1],
    reverse=True
)


# ==========================================
# 9. CREATE HTML GALLERY
# ==========================================

html = """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Smart Event AI - Personal Gallery</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    margin: 0;
    padding: 30px;
}

h1 {
    text-align: center;
}

.gallery {

    display: grid;

    grid-template-columns:
        repeat(auto-fit, minmax(280px, 1fr));

    gap: 25px;

    max-width: 1200px;

    margin: 30px auto;
}

.card {

    background: white;

    border-radius: 12px;

    padding: 15px;

    box-shadow:
        0 4px 15px rgba(0,0,0,0.1);

}

.card img {

    width: 100%;

    border-radius: 8px;

}

.filename {

    margin-top: 10px;

    font-weight: bold;

}

.score {

    margin-top: 5px;

    color: #555;

}

</style>

</head>

<body>

<h1>
    📸 Your Event Photos
</h1>

<div class="gallery">
"""


# ==========================================
# 10. ADD MATCHING PHOTOS
# ==========================================

for photo, score in sorted_photos:

    image_path = os.path.join(
        "event_photos",
        photo
    )

    html += f"""
    <div class="card">

        <img
            src="{image_path}"
            alt="{photo}"
        >

        <div class="filename">
            {photo}
        </div>

        <div class="score">
            Similarity: {score:.4f}
        </div>

    </div>
    """


# ==========================================
# 11. CLOSE HTML
# ==========================================

html += """
</div>

</body>

</html>
"""


# ==========================================
# 12. SAVE HTML FILE
# ==========================================

output_file = "personal_gallery.html"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(html)


# ==========================================
# 13. FINAL RESULT
# ==========================================

print("\n========================================")

print(
    "PERSONAL GALLERY CREATED"
)

print(
    "========================================"
)

print(
    "Matching photos:",
    len(sorted_photos)
)

print(
    "Gallery file:",
    output_file
)

print(
    "\nOpen personal_gallery.html "
    "in your browser."
)
