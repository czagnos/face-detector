"""Real-time face recognition and metadata server.

This module starts a background camera worker that captures frames from the
configured webcam, detects faces, extracts embeddings with DeepFace, and stores
metadata for every unique visitor. A FastAPI application exposes an endpoint to
query the most recent face that is currently visible to the camera.

Usage (from the repository root)::

    python main.py --host 0.0.0.0 --port 8000

The server will start streaming from the first webcam (index 0). Use CTRL+C to
stop both the FastAPI server and the camera worker.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

try:
    # DeepFace is used only for embedding extraction; detection is performed by OpenCV.
    from deepface import DeepFace
except ImportError as exc:  # pragma: no cover - import guard for documentation only
    raise SystemExit(
        "DeepFace is required for this application. Install it with 'pip install deepface'."
    ) from exc


LOGGER = logging.getLogger("face_monitor")
DEFAULT_DATA_DIR = Path("data")
EMBEDDING_THRESHOLD = 0.6  # Facenet512 recommended threshold for L2 distance
VISIT_GAP_SECONDS = 5.0


@dataclass
class UserProfile:
    """Representation of a known user's face embedding and metadata."""

    user_id: str
    embedding: np.ndarray
    image_path: Path
    metadata_path: Path
    first_seen: datetime
    last_seen: datetime
    visit_count: int = 1
    total_detections: int = 1

    def to_json(self) -> Dict[str, object]:
        data = asdict(self)
        data.update(
            {
                "embedding": self.embedding.astype(float).tolist(),
                "image_path": str(self.image_path),
                "metadata_path": str(self.metadata_path),
                "first_seen": self.first_seen.isoformat(),
                "last_seen": self.last_seen.isoformat(),
            }
        )
        return data

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "UserProfile":
        embedding = np.array(payload["embedding"], dtype=np.float32)
        return cls(
            user_id=str(payload["user_id"]),
            embedding=embedding,
            image_path=Path(payload["image_path"]),
            metadata_path=Path(payload["metadata_path"]),
            first_seen=datetime.fromisoformat(payload["first_seen"]),
            last_seen=datetime.fromisoformat(payload["last_seen"]),
            visit_count=int(payload.get("visit_count", 1)),
            total_detections=int(payload.get("total_detections", 1)),
        )


class FaceRegistry:
    """Persistent store for known user profiles."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, distance_threshold: float = EMBEDDING_THRESHOLD) -> None:
        self.data_dir = data_dir
        self.faces_dir = self.data_dir / "faces"
        self.distance_threshold = distance_threshold
        self._lock = threading.Lock()
        self._profiles: Dict[str, UserProfile] = {}
        self._model = DeepFace.build_model("Facenet512")
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.faces_dir.exists():
            return
        for metadata_path in self.faces_dir.rglob("metadata.json"):
            try:
                payload = json.loads(metadata_path.read_text())
                profile = UserProfile.from_json(payload)
            except Exception as err:  # pragma: no cover - best effort import
                LOGGER.error("Failed to load metadata from %s: %s", metadata_path, err)
                continue
            self._profiles[profile.user_id] = profile
        LOGGER.info("Loaded %d existing user profiles", len(self._profiles))

    @property
    def next_user_index(self) -> int:
        if not self._profiles:
            return 1
        max_idx = max(int(profile.user_id.split("-")[-1]) for profile in self._profiles.values())
        return max_idx + 1

    def _persist(self, profile: UserProfile) -> None:
        profile.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        profile.metadata_path.write_text(json.dumps(profile.to_json(), indent=2))

    def _save_face_image(self, user_id: str, face_image: np.ndarray) -> Path:
        user_dir = self.faces_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        image_path = user_dir / "face.jpg"
        # Convert RGB to BGR before saving with OpenCV.
        bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(image_path), bgr)
        return image_path

    def _match_profile(self, embedding: np.ndarray) -> Optional[UserProfile]:
        best_match: Tuple[Optional[UserProfile], float] = (None, float("inf"))
        for profile in self._profiles.values():
            distance = np.linalg.norm(profile.embedding - embedding)
            if distance < best_match[1]:
                best_match = (profile, distance)
        matched_profile, distance = best_match
        if matched_profile and distance <= self.distance_threshold:
            return matched_profile
        return None

    def upsert_profile(self, embedding: np.ndarray, face_image: np.ndarray) -> Tuple[UserProfile, bool]:
        """Insert a new profile or update an existing one.

        Returns a tuple of (profile, created_flag).
        """
        with self._lock:
            profile = self._match_profile(embedding)
            now = datetime.utcnow()
            if profile:
                elapsed = (now - profile.last_seen).total_seconds()
                profile.last_seen = now
                profile.total_detections += 1
                if elapsed >= VISIT_GAP_SECONDS:
                    profile.visit_count += 1
                self._persist(profile)
                return profile, False

            user_id = f"user-{self.next_user_index}"
            image_path = self._save_face_image(user_id, face_image)
            metadata_path = image_path.with_name("metadata.json")
            profile = UserProfile(
                user_id=user_id,
                embedding=embedding,
                image_path=image_path,
                metadata_path=metadata_path,
                first_seen=now,
                last_seen=now,
            )
            self._profiles[user_id] = profile
            self._persist(profile)
            return profile, True

    def to_summary(self, profile: UserProfile) -> Dict[str, object]:
        return {
            "user_id": profile.user_id,
            "first_seen": profile.first_seen,
            "last_seen": profile.last_seen,
            "visit_count": profile.visit_count,
            "total_detections": profile.total_detections,
            "image_path": str(profile.image_path),
        }

    def get_profiles(self) -> List[UserProfile]:
        with self._lock:
            return list(self._profiles.values())

    @property
    def model(self):
        return self._model


@dataclass
class ActiveFace:
    profile: UserProfile
    last_seen: datetime = field(default_factory=datetime.utcnow)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "user_id": self.profile.user_id,
            "last_seen": self.last_seen.isoformat(),
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3],
            },
        }
        payload.update(
            {
                "visit_count": self.profile.visit_count,
                "total_detections": self.profile.total_detections,
                "image_path": str(self.profile.image_path),
                "first_seen": self.profile.first_seen.isoformat(),
            }
        )
        return payload


class FaceMonitor:
    """Background worker that continuously reads frames from a camera."""

    def __init__(
        self,
        registry: FaceRegistry,
        camera_index: int = 0,
        frame_interval: int = 3,
        min_size: Tuple[int, int] = (120, 120),
        sleep_time: float = 0.01,
    ) -> None:
        self.registry = registry
        self.camera_index = camera_index
        self.frame_interval = max(1, frame_interval)
        self.min_size = min_size
        self.sleep_time = sleep_time
        self._capture = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._active_faces: Dict[str, ActiveFace] = {}
        self._face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        LOGGER.info("Camera monitor started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._capture:
            self._capture.release()
        LOGGER.info("Camera monitor stopped")

    def _ensure_capture(self) -> None:
        if self._capture is None:
            self._capture = cv2.VideoCapture(self.camera_index)
            if not self._capture.isOpened():
                raise RuntimeError(f"Unable to open camera index {self.camera_index}")

    def _extract_embedding(self, face_rgb: np.ndarray) -> np.ndarray:
        # DeepFace expects RGB input. We skip detection because the ROI is already isolated.
        representations = DeepFace.represent(
            img_path=None,
            img=face_rgb,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="skip",
            model=self.registry.model,
        )
        embedding_vector = np.array(representations[0]["embedding"], dtype=np.float32)
        return embedding_vector

    def _run_loop(self) -> None:
        self._ensure_capture()
        frame_count = 0
        while not self._stop_event.is_set():
            ret, frame = self._capture.read()
            if not ret:
                LOGGER.warning("Frame capture failed; retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1
            if frame_count % self.frame_interval != 0:
                time.sleep(self.sleep_time)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self._face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_size,
            )
            for (x, y, w, h) in detections:
                face_bgr = frame[y : y + h, x : x + w]
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                try:
                    embedding = self._extract_embedding(face_rgb)
                except Exception as err:
                    LOGGER.error("Failed to compute embedding: %s", err)
                    continue

                profile, created = self.registry.upsert_profile(embedding, face_rgb)
                if created:
                    LOGGER.info("Registered new user %s", profile.user_id)
                else:
                    LOGGER.debug("Recognized existing user %s", profile.user_id)

                self._active_faces[profile.user_id] = ActiveFace(
                    profile=profile,
                    last_seen=datetime.utcnow(),
                    bbox=(int(x), int(y), int(w), int(h)),
                )

            self._cleanup_inactive_faces()
            time.sleep(self.sleep_time)

    def _cleanup_inactive_faces(self, expiry_seconds: float = 3.0) -> None:
        now = datetime.utcnow()
        to_remove = [
            user_id
            for user_id, active_face in self._active_faces.items()
            if (now - active_face.last_seen).total_seconds() > expiry_seconds
        ]
        for user_id in to_remove:
            del self._active_faces[user_id]

    def get_active_faces(self) -> List[Dict[str, object]]:
        return [active_face.to_dict() for active_face in self._active_faces.values()]

    def get_current_user(self) -> Optional[Dict[str, object]]:
        if not self._active_faces:
            return None
        active_face = max(self._active_faces.values(), key=lambda face: face.last_seen)
        return active_face.to_dict()


class CurrentUserResponse(BaseModel):
    user_id: str
    first_seen: datetime
    last_seen: datetime
    visit_count: int
    total_detections: int
    image_path: str
    bbox: Dict[str, int]


def build_app(monitor: FaceMonitor, registry: FaceRegistry) -> FastAPI:
    api = FastAPI(title="Face Monitor", version="1.0.0")

    @api.get("/current_user", response_model=Optional[CurrentUserResponse])
    def current_user() -> Optional[Dict[str, object]]:
        user = monitor.get_current_user()
        if user is None:
            return None
        profile = next(
            (profile for profile in registry.get_profiles() if profile.user_id == user["user_id"]),
            None,
        )
        if profile is None:
            return None
        summary = registry.to_summary(profile)
        summary.update({"bbox": user["bbox"]})
        return summary

    @api.get("/active_users")
    def active_users() -> JSONResponse:
        return JSONResponse(monitor.get_active_faces())

    @api.get("/known_users")
    def known_users() -> List[Dict[str, object]]:
        return [registry.to_summary(profile) for profile in registry.get_profiles()]

    return api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time face recognition server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the API server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to read from")
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=3,
        help="Process every Nth frame to balance accuracy and latency",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where face images and metadata will be stored",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=EMBEDDING_THRESHOLD,
        help="Maximum L2 distance between embeddings to consider the same user",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    registry = FaceRegistry(data_dir=args.data_dir, distance_threshold=args.distance_threshold)
    monitor = FaceMonitor(registry=registry, camera_index=args.camera, frame_interval=args.frame_interval)
    monitor.start()

    app = build_app(monitor, registry)

    def shutdown_handler(signum, frame):  # pragma: no cover - runtime shutdown handler
        LOGGER.info("Received signal %s, stopping monitor", signum)
        monitor.stop()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
