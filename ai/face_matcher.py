
import cv2
import numpy as np
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
# 2. GET ALL FACE EMBEDDINGS FROM AN IMAGE
# ==========================================

def get_all_embeddings(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Could not load:", image_path)
        return []

    faces = app.get(image)

    print(f"{image_path} → {len(faces)} face(s) found")

    embeddings = []

    for face in faces:
        embeddings.append(face.embedding)

    return embeddings


# ==========================================
# 3. COSINE SIMILARITY
# ==========================================

def cosine_similarity(embedding1, embedding2):

    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) *
        np.linalg.norm(embedding2)
    )


# ==========================================
# 4. QUERY IMAGE
# ==========================================

query_image = "test_same_person/arman_1.jpg"

query_embeddings = get_all_embeddings(query_image)

if len(query_embeddings) == 0:

    print("\n❌ No face found in query image.")

    exit()


# ==========================================
# 5. USE FIRST FACE AS QUERY
# ==========================================

query_embedding = query_embeddings[0]

print("\nQuery face selected.")
print("Embedding length:", len(query_embedding))


# ==========================================
# 6. CANDIDATE PHOTOS
# ==========================================

candidate_images = [

    "test_same_person/arman_2.jpg",
    "test_same_person/arman_3.jpg",
    "test_same_person/arman_4.jpg",
    "test.jpg"

]


# ==========================================
# 7. SEARCH FOR MATCH
# ==========================================

all_results = []

print("\n========================================")
print("SEARCHING FOR YOUR FACE...")
print("========================================\n")


for image_path in candidate_images:

    embeddings = get_all_embeddings(image_path)

    if len(embeddings) == 0:
        continue


    # Compare query face with EVERY face
    # inside the candidate image

    for face_number, embedding in enumerate(embeddings):

        score = cosine_similarity(
            query_embedding,
            embedding
        )

        all_results.append(
            (
                image_path,
                face_number + 1,
                score
            )
        )


# ==========================================
# 8. SORT RESULTS
# ==========================================

all_results.sort(
    key=lambda item: item[2],
    reverse=True
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n========================================")
print("MATCHING RESULTS")
print("========================================\n")


for image_path, face_number, score in all_results:

    print(
        f"{image_path} | "
        f"Face {face_number} | "
        f"Similarity: {score:.4f}"
    )


# ==========================================
# 10. BEST MATCH
# ==========================================

if len(all_results) > 0:

    best_image, best_face, best_score = all_results[0]

    print("\n========================================")
    print("BEST MATCH")
    print("========================================")

    print("Image :", best_image)
    print("Face  :", best_face)
    print("Score :", f"{best_score:.4f}")

else:

    print("\n❌ No faces available for comparison.")

