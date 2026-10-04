import cv2
import numpy as np
from insightface.app import FaceAnalysis


# -----------------------------
# 1. Load AI model
# -----------------------------

print("Loading AI model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("Model loaded.\n")


# -----------------------------
# 2. Function to get embedding
# -----------------------------

def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Could not load:", image_path)
        return None

    faces = app.get(image)

    if len(faces) == 0:
        print("No face detected:", image_path)
        return None

    return faces[0].embedding


# -----------------------------
# 3. Get embeddings
# -----------------------------

print("Processing face 1...")
embedding1 = get_embedding("test_same_person/arman_4.jpg")

print("Processing face 2...")
embedding2 = get_embedding("test_same_person/arman_3.jpg")


# -----------------------------
# 4. Compare faces
# -----------------------------

if embedding1 is not None and embedding2 is not None:

    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) *
        np.linalg.norm(embedding2)
    )

    print("\nSimilarity score:", similarity)