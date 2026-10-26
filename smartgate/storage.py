"""SQLite persistence for identity data, photos, embeddings and visit records."""
from __future__ import annotations

import csv
from contextlib import contextmanager
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


class Storage:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.data_dir = self.project_dir / "data"
        self.faces_dir = self.data_dir / "faces"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "smartgate.db"
        with self._conn() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    photo_file TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    distance REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attendance_date
                    ON attendance(occurred_at DESC);
            """)
        self._ensure_altyn_columns()
    def _ensure_altyn_columns(self):
        with self._conn() as db:
            db.execute("BEGIN IMMEDIATE")

            existing = {
                row["name"]
                for row in db.execute("PRAGMA table_info(users)")
            }

            additions = {
                "altyn_student_id": "TEXT",
                "altyn_photo_version": "TEXT",
                "altyn_synced_at": "TEXT",
            }

            for name, definition in additions.items():
                if name not in existing:
                    db.execute(
                        f'ALTER TABLE users ADD COLUMN "{name}" {definition}'
                    )

            db.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_users_altyn_student_id
                ON users(altyn_student_id)
                WHERE altyn_student_id IS NOT NULL
                """
            )
    @contextmanager
    def _conn(self):
        db = sqlite3.connect(str(self.db_path), timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _validate_name(name: str) -> str:
        name = " ".join(name.strip().split())
        if not name or len(name) > 80:
            raise ValueError("Введите имя длиной от 1 до 80 символов.")
        if any(ord(c) < 32 for c in name):
            raise ValueError("Имя содержит недопустимые символы.")
        return name

    def add_user(self, name: str, face_rgb: np.ndarray, embedding: np.ndarray) -> int:
        name = self._validate_name(name)
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.size < 32 or not np.isfinite(embedding).all():
            raise ValueError("Некорректный биометрический шаблон.")
        face_rgb = np.asarray(face_rgb, dtype=np.uint8)
        if face_rgb.ndim != 3 or face_rgb.shape[2] != 3 or not face_rgb.size:
            raise ValueError("Некорректная фотография.")
        with self._conn() as db:
            if db.execute("SELECT 1 FROM users WHERE name=?", (name,)).fetchone():
                raise ValueError(f"Пользователь «{name}» уже зарегистрирован.")
        filename = f"{uuid.uuid4().hex}.jpg"
        path = self.faces_dir / filename
        Image.fromarray(face_rgb).save(path, format="JPEG", quality=92)
        try:
            with self._conn() as db:
                cur = db.execute("""
                    INSERT INTO users(name, photo_file, embedding, created_at)
                    VALUES(?, ?, ?, ?)
                """, (name, filename, sqlite3.Binary(embedding.tobytes()),
                      datetime.now().isoformat(timespec="seconds")))
                return int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            path.unlink(missing_ok=True)
            raise ValueError(f"Пользователь «{name}» уже зарегистрирован.") from exc
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def list_users(self) -> list[dict]:
        with self._conn() as db:
            rows = db.execute("""
                SELECT id, name, photo_file, created_at
                FROM users ORDER BY name COLLATE NOCASE
            """).fetchall()
        return [dict(r) for r in rows]

    def face_references(self) -> list[tuple[int, str, np.ndarray]]:
        with self._conn() as db:
            rows = db.execute("SELECT id, name, embedding FROM users").fetchall()
        return [(r["id"], r["name"], np.frombuffer(r["embedding"], dtype=np.float32).copy())
                for r in rows]

    def delete_user(self, user_id: int) -> bool:
        with self._conn() as db:
            row = db.execute("SELECT photo_file FROM users WHERE id=?", (user_id,)).fetchone()
            if row is None:
                return False
            db.execute("DELETE FROM users WHERE id=?", (user_id,))
        # SQLite removes the embedding and all related attendance via ON DELETE CASCADE.
        (self.faces_dir / row["photo_file"]).unlink(missing_ok=True)
        return True


    def rename_user(self, user_id: int, new_name: str) -> bool:
        new_name = self._validate_name(new_name)

        try:
            with self._conn() as db:
                db.execute("BEGIN IMMEDIATE")

                users = db.execute(
                    "SELECT id, name FROM users"
                ).fetchall()

                if not any(row["id"] == user_id for row in users):
                    return False

                for row in users:
                    if (
                        row["id"] != user_id
                        and row["name"].casefold() == new_name.casefold()
                    ):
                        raise ValueError(
                            "Пользователь с таким именем уже существует."
                        )

                db.execute(
                    "UPDATE users SET name=? WHERE id=?",
                    (new_name, user_id)
                )

                db.execute(
                    "UPDATE attendance SET name=? WHERE user_id=?",
                    (new_name, user_id)
                )

            return True

        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "Пользователь с таким именем уже существует."
            ) from exc




    def add_attendance(self, user_id: int, distance: float) -> bool:
        with self._conn() as db:
            user = db.execute("SELECT name FROM users WHERE id=?", (user_id,)).fetchone()
            if user is None:
                return False
            db.execute("""
                INSERT INTO attendance(user_id, name, occurred_at, distance)
                VALUES(?, ?, ?, ?)
            """, (user_id, user["name"], datetime.now().isoformat(timespec="seconds"),
                  round(float(distance), 4)))
            return True

    def list_attendance(self, limit: int = 100) -> list[dict]:
        with self._conn() as db:
            rows = db.execute("""
                SELECT id, user_id, name, occurred_at, distance
                FROM attendance ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> tuple[int, int]:
        with self._conn() as db:
            users = db.execute("SELECT count(*) FROM users").fetchone()[0]
            today = datetime.now().date().isoformat()
            visits = db.execute("SELECT count(*) FROM attendance WHERE occurred_at LIKE ?",
                                (today + "%",)).fetchone()[0]
            return users, visits

    def export_csv(self, target: str | Path) -> None:
        with self._conn() as db:
            rows = db.execute("""
                SELECT name, occurred_at, distance FROM attendance
                ORDER BY occurred_at DESC
            """).fetchall()
        with open(target, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Имя", "Дата и время", "Расстояние FaceNet"])
            for row in rows:
                writer.writerow([row["name"], row["occurred_at"], row["distance"]])
