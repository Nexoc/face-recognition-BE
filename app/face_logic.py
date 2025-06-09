import numpy as np
import face_recognition
from fastapi import UploadFile
from qdrant_client.models import PointStruct, VectorParams
from pathlib import Path
import io
import logging

# https://face-recognition.readthedocs.io/en/latest/readme.html
# https://pypi.org/project/face-recognition/

logger = logging.getLogger(__name__)

# Константы
COLLECTION_NAME = "faces"
SCORE_THRESHOLD = 0.9
MAX_PHOTOS = 5
ID_COUNTER_FILE = Path("id_counter.txt")
START_ID = 100_000_000

def get_next_id() -> int:
    """Безопасно получить следующий ID"""
    if not ID_COUNTER_FILE.exists() or ID_COUNTER_FILE.read_text().strip() == "":
        ID_COUNTER_FILE.write_text(str(START_ID))
        return START_ID

    try:
        current = int(ID_COUNTER_FILE.read_text().strip())
    except ValueError:
        logger.warning("Пустой или некорректный id_counter.txt — сбрасываем.")
        current = START_ID

    next_id = current + 1
    ID_COUNTER_FILE.write_text(str(next_id))
    return next_id


async def process_faces(photos: list[UploadFile], qdrant_client):
    try:
        if len(photos) > MAX_PHOTOS:
            logger.warning(f"Too many photos uploaded ({len(photos)}). Trimming to {MAX_PHOTOS}")
            photos = photos[:MAX_PHOTOS]

        logger.info(f"Processing {len(photos)} file(s)")
        encodings = []

        for photo in photos:
            logger.info(f"Processing: {photo.filename}")
            image_bytes = await photo.read()
            image = face_recognition.load_image_file(io.BytesIO(image_bytes))
            detected = face_recognition.face_encodings(image)

            if detected and len(detected[0]) == 128:
                encodings.append(detected[0])
                logger.info(f"Face found in {photo.filename}")
            else:
                logger.warning(f"No face found in {photo.filename}")

        if not encodings:
            return {"error": "Kein Gesicht erkannt"}

        face_vector = np.mean(encodings, axis=0)

        if not qdrant_client.collection_exists(COLLECTION_NAME):
            logger.info(f"Creating collection '{COLLECTION_NAME}' in Qdrant")
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=128,
                    distance="Cosine"
                )
            )

        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=face_vector.tolist(),
            limit=1
        )

        if search_result:
            logger.info(f"Top score: {search_result[0].score}")

        if search_result and search_result[0].score > SCORE_THRESHOLD:
            matched_id = search_result[0].id
            logger.info(f"Face recognized as ID {matched_id}")
            return {"id": matched_id, "status": "recognized"}
        else:
            new_id = get_next_id()
            logger.info(f"➕ Registering new face with ID {new_id}")
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(id=new_id, vector=face_vector.tolist(), payload={})
                ]
            )
            return {"id": new_id, "status": "registered"}

    except Exception as e:
        logger.exception("Unexpected error during face processing")
        return {"error": "Interner Fehler beim Verarbeiten der Bilder."}
