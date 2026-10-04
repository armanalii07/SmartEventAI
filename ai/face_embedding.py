import cv2
from insightface.app import FaceAnalysis

print("Loading AI model...")

app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

print("AI model loaded successfully.")


def get_face_embeddings(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not load image.")

    faces = app.get(image)

    results = []

    for face in faces:
        results.append(face.embedding.tolist())

    return results