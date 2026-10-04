import cv2
import numpy as np
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
# 2. LOAD SAVED FACE DATABASE
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
# 3. FUNCTION TO GET QUERY EMBEDDING
# ==========================================

def get_query_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:

        print(
            "❌ Could not load:",
            image_path
        )

        return None


    faces = app.get(image)


    print(
        f"\nQuery image → {len(faces)} face(s) found"
    )


    if len(faces) == 0:

        print("❌ No face found.")

        return None


    if len(faces) > 1:

        print(
            "⚠️ Multiple faces found. "
            "Using the first face."
        )


    return faces[0].embedding


# ==========================================
# 4. COSINE SIMILARITY
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
# 5. USER SELFIE
# ==========================================

query_image = (
    "test_same_person/arman_1.jpg"
)


query_embedding = get_query_embedding(
    query_image
)


if query_embedding is None:

    exit()


# ==========================================
# 6. SEARCH DATABASE
# ==========================================

results = []


print(
    "\nSearching face database..."
)


for record in face_database:

    stored_embedding = record[
        "embedding"
    ]


    score = cosine_similarity(
        query_embedding,
        stored_embedding
    )


    results.append({

        "photo": record["photo"],

        "face_number":
            record["face_number"],

        "score": score

    })


# ==========================================
# 7. SORT RESULTS
# ==========================================

results.sort(
    key=lambda item: item["score"],
    reverse=True
)


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print(
    "\n========================================"
)

print(
    "SEARCH RESULTS"
)

print(
    "========================================\n"
)


for result in results:

    print(
        f"{result['photo']} | "
        f"Face {result['face_number']} | "
        f"Similarity: "
        f"{result['score']:.4f}"
    )


# ==========================================
# 9. BEST MATCH
# ==========================================

if len(results) > 0:

    best = results[0]


    print(
        "\n========================================"
    )

    print(
        "BEST MATCH"
    )

    print(
        "========================================"
    )


    print(
        "Photo:",
        best["photo"]
    )

    print(
        "Face:",
        best["face_number"]
    )

    print(
        "Similarity:",
        f"{best['score']:.4f}"
    )
