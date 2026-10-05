from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

import os
import ast
import numpy as np
import uuid

from backend.database import SessionLocal, Base, engine
from backend.models import Event, Photo, Face, Admin
from ai.face_embedding import get_face_embeddings

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from passlib.context import CryptContext


# =========================================================
# PASSWORD CONFIGURATION
# =========================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Smart Event AI",
    description="AI-Based Smart Event Photo Management System",
    version="1.0.0"
)

Base.metadata.create_all(bind=engine)


# =========================================================
# STORAGE
# =========================================================

UPLOAD_DIR = "storage/event_photos"
SEARCH_DIR = "storage/search"

# Create storage directories BEFORE mounting StaticFiles
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)


# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/gallery",
    StaticFiles(directory="frontend", html=True),
    name="gallery"
)

app.mount(
    "/photos",
    StaticFiles(directory=UPLOAD_DIR),
    name="photos"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# EVENT MODEL
# =========================================================

class EventCreate(BaseModel):
    event_name: str
    event_code: str


# =========================================================
# DATABASE
# =========================================================

def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Smart Event AI Backend is running!",
        "status": "success"
    }


# =========================================================
# CREATE EVENT
# =========================================================

@app.post("/events")
def create_event(
    event: EventCreate,
    db: Session = Depends(get_db)
):

    event_name = event.event_name.strip()
    event_code = event.event_code.strip().upper()

    if not event_name or not event_code:

        raise HTTPException(
            status_code=400,
            detail="Event name and event code are required."
        )

    # Check duplicate event code
    existing_event = (
        db.query(Event)
        .filter(Event.event_code == event_code)
        .first()
    )

    if existing_event:

        raise HTTPException(
            status_code=400,
            detail="Event code already exists. Please use another code."
        )

    new_event = Event(
        event_name=event_name,
        event_code=event_code
    )

    db.add(new_event)
    db.commit()
    db.refresh(new_event)

    return {
        "message": "Event created successfully",
        "event": {
            "id": new_event.id,
            "event_name": new_event.event_name,
            "event_code": new_event.event_code
        }
    }


# =========================================================
# GET ALL EVENTS
# =========================================================

@app.get("/events")
def get_events(
    db: Session = Depends(get_db)
):

    events = db.query(Event).order_by(
        Event.created_at.desc()
    ).all()

    return {
        "events": [

            {
                "id": event.id,
                "event_name": event.event_name,
                "event_code": event.event_code,
                "created_at": event.created_at
            }

            for event in events
        ]
    }


# =========================================================
# GET EVENT BY CODE
# =========================================================

@app.get("/events/{event_code}")
def get_event_by_code(
    event_code: str,
    db: Session = Depends(get_db)
):

    event = db.query(Event).filter(
        Event.event_code == event_code.upper()
    ).first()

    if not event:

        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    return {
        "id": event.id,
        "event_name": event.event_name,
        "event_code": event.event_code,
        "created_at": event.created_at
    }


# =========================================================
# SINGLE PHOTO UPLOAD
# =========================================================

@app.post("/events/{event_code}/photos")
async def upload_photo(
    event_code: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    event = db.query(Event).filter(
        Event.event_code == event_code.upper()
    ).first()

    if not event:

        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    unique_filename = (
        f"{uuid.uuid4().hex}_{file.filename}"
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        unique_filename
    )

    contents = await file.read()

    with open(file_path, "wb") as f:
        f.write(contents)

    new_photo = Photo(
        event_id=event.id,
        filename=unique_filename
    )

    db.add(new_photo)
    db.commit()
    db.refresh(new_photo)

    embeddings = get_face_embeddings(
        file_path
    )

    for index, embedding in enumerate(
        embeddings,
        start=1
    ):

        new_face = Face(
            photo_id=new_photo.id,
            face_number=index,
            embedding=str(embedding)
        )

        db.add(new_face)

    db.commit()

    return {
        "message": "Photo uploaded and processed successfully",

        "photo": {
            "id": new_photo.id,
            "event_id": new_photo.event_id,
            "filename": new_photo.filename,
            "faces_detected": len(embeddings)
        }
    }


# =========================================================
# SEARCH PHOTOS USING SELFIE
# =========================================================

@app.post("/events/{event_code}/search")
async def search_photos(
    event_code: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    event = db.query(Event).filter(
        Event.event_code == event_code.upper()
    ).first()

    if not event:

        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    search_dir = SEARCH_DIR

    selfie_filename = (
        f"{uuid.uuid4().hex}_{file.filename}"
    )

    selfie_path = os.path.join(
        search_dir,
        selfie_filename
    )

    contents = await file.read()

    with open(
        selfie_path,
        "wb"
    ) as f:

        f.write(contents)

    selfie_embeddings = get_face_embeddings(
        selfie_path
    )

    if len(selfie_embeddings) == 0:

        raise HTTPException(
            status_code=400,
            detail="No face detected in selfie"
        )

    selfie_embedding = selfie_embeddings[0]

    photos = db.query(Photo).filter(
        Photo.event_id == event.id
    ).all()

    matches = []

    for photo in photos:

        faces = db.query(Face).filter(
            Face.photo_id == photo.id
        ).all()

        best_score = 0

        for face in faces:

            stored_embedding = ast.literal_eval(
                face.embedding
            )

            score = cosine_similarity(
                selfie_embedding,
                stored_embedding
            )

            if score > best_score:
                best_score = score

        if best_score >= 0.60:

            matches.append({

                "photo_id": photo.id,

                "filename": photo.filename,

                "similarity": round(
                    float(best_score),
                    4
                ),

                "photo_url":
                    f"/photos/{photo.filename}"
            })

    matches.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return {

        "event_code": event_code.upper(),

        "matches_found": len(matches),

        "matches": matches
    }


# =========================================================
# BULK PHOTO UPLOAD
# =========================================================

@app.post("/events/{event_code}/photos/bulk")
async def upload_bulk_photos(
    event_code: str,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):

    event = db.query(Event).filter(
        Event.event_code == event_code.upper()
    ).first()

    if not event:

        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    results = []

    for file in files:

        unique_filename = (
            f"{uuid.uuid4().hex}_{file.filename}"
        )

        file_path = os.path.join(
            UPLOAD_DIR,
            unique_filename
        )

        contents = await file.read()

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(contents)

        new_photo = Photo(
            event_id=event.id,
            filename=unique_filename
        )

        db.add(new_photo)
        db.commit()
        db.refresh(new_photo)

        embeddings = get_face_embeddings(
            file_path
        )

        for index, embedding in enumerate(
            embeddings,
            start=1
        ):

            new_face = Face(
                photo_id=new_photo.id,
                face_number=index,
                embedding=str(embedding)
            )

            db.add(new_face)

        db.commit()

        results.append({

            "photo_id": new_photo.id,

            "filename": unique_filename,

            "faces_detected": len(embeddings)
        })

    return {

        "message":
            "Bulk photos uploaded and processed successfully",

        "event_code": event_code.upper(),

        "total_photos": len(results),

        "photos": results
    }


# =========================================================
# ADMIN LOGIN
# =========================================================

@app.post("/admin/login")
def admin_login(
    username: str,
    password: str,
    db: Session = Depends(get_db)
):

    print("ADMIN LOGIN CALLED")

    admin = db.query(Admin).filter(
        Admin.username == username
    ).first()

    if not admin:

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    if not pwd_context.verify(
        password,
        admin.password_hash
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    return {

        "message": "Login successful",

        "username": admin.username
    }


# =========================================================
# ADMIN STATS
# =========================================================

@app.get("/admin/stats")
def admin_stats(
    db: Session = Depends(get_db)
):

    total_events = db.query(Event).count()

    total_photos = db.query(Photo).count()

    total_faces = db.query(Face).count()

    return {

        "total_events": total_events,

        "total_photos": total_photos,

        "total_faces": total_faces
    }


# =========================================================
# ADMIN EVENTS
# =========================================================

@app.get("/admin/events")
def admin_events(
    db: Session = Depends(get_db)
):

    events = db.query(Event).order_by(
        Event.created_at.desc()
    ).all()

    result = []

    for event in events:

        photos = db.query(Photo).filter(
            Photo.event_id == event.id
        ).all()

        photo_count = len(photos)

        face_count = 0

        for photo in photos:

            face_count += db.query(Face).filter(
                Face.photo_id == photo.id
            ).count()

        result.append({

            "id": event.id,

            "event_name": event.event_name,

            "event_code": event.event_code,

            "created_at": event.created_at,

            "total_photos": photo_count,

            "total_faces": face_count
        })

    return {
        "events": result
    }


# =========================================================
# GET EVENT PHOTOS
# =========================================================

@app.get("/admin/events/{event_code}/photos")
def get_event_photos(
    event_code: str,
    db: Session = Depends(get_db)
):

    event = db.query(Event).filter(
        Event.event_code == event_code.upper()
    ).first()

    if not event:

        raise HTTPException(
            status_code=404,
            detail="Event not found"
        )

    photos = db.query(Photo).filter(
        Photo.event_id == event.id
    ).all()

    return {

        "event_name": event.event_name,

        "event_code": event.event_code,

        "total_photos": len(photos),

        "photos": [

            {
                "id": photo.id,

                "filename": photo.filename,

                "photo_url":
                    f"/photos/{photo.filename}"
            }

            for photo in photos
        ]
    }