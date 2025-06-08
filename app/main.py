from fastapi import FastAPI, UploadFile, File
from .face_logic import process_faces
from qdrant_client import QdrantClient
from .logging_config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("🚀 FastAPI server started")

# Подключение к Qdrant
qdrant = QdrantClient(host="qdrant", port=6333)

@app.post(
    "/recognize",
    summary="Recognize or register a face",
    description="Accepts **1 to 5 photos** of a face. Extracts face embeddings from all images, "
                "averages them, and compares the result against known faces in the database. "
                "If the face is already known, returns the matching ID. Otherwise, registers a new face and returns a new ID."
)
async def recognize(photos: list[UploadFile] = File(...)):
    result = await process_faces(photos, qdrant)
    return result
