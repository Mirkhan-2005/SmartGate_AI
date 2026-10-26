import getpass
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
from PIL import Image

from .altyn_client import AltynClient, AltynError
from .storage import Storage
from .vision import FaceEngine


def _require_active(student):
    if (
        student.get("student_status") != "active"
        or student.get("access_allowed") is not True
        or not student.get("photo_url")
        or not student.get("photo_version")
    ):
        raise AltynError(
            "Студент или его фотография недоступны для импорта."
        )


def _save_student(storage, student, face, embedding):
    name = storage._validate_name(
        " ".join(
            part
            for part in (
                student["last_name"],
                student["first_name"],
                student.get("middle_name"),
            )
            if part
        )
    )

    external_id = str(student["id"])
    version = student["photo_version"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    new_path = None
    old_file = None
    committed = False

    try:
        if face is not None:
            face = np.asarray(face, dtype=np.uint8)
            vector = np.asarray(
                embedding,
                dtype=np.float32,
            ).reshape(-1)

            if (
                face.ndim != 3
                or face.shape[2] != 3
                or not face.size
                or vector.size < 32
                or not np.isfinite(vector).all()
            ):
                raise ValueError(
                    "Некорректный результат распознавания лица."
                )

            new_path = storage.faces_dir / f"{uuid4().hex}.jpg"

            Image.fromarray(face).save(
                new_path,
                format="JPEG",
                quality=92,
            )

            new_embedding = sqlite3.Binary(vector.tobytes())

        with storage._conn() as db:
            db.execute("BEGIN IMMEDIATE")

            current = db.execute(
                "SELECT * FROM users WHERE altyn_student_id=?",
                (external_id,),
            ).fetchone()

            current_id = current["id"] if current else None

            # Совпадение ФИО не доказывает, что это тот же человек.
            for row in db.execute("SELECT id, name FROM users"):
                if (
                    row["id"] != current_id
                    and row["name"].casefold() == name.casefold()
                ):
                    raise ValueError(
                        "Такое ФИО уже есть в SmartGate: "
                        f"локальный ID {row['id']}. "
                        "Автоматическое связывание отменено."
                    )

            if new_path is None:
                if (
                    current is None
                    or current["altyn_photo_version"] != version
                    or not (
                        storage.faces_dir / current["photo_file"]
                    ).is_file()
                ):
                    raise ValueError(
                        "Локальная запись изменилась. Повтори импорт."
                    )

                filename = current["photo_file"]
                blob = current["embedding"]

            else:
                filename = new_path.name
                blob = new_embedding
                old_file = current["photo_file"] if current else None

            if current:
                db.execute(
                    """
                    UPDATE users
                    SET name=?, photo_file=?, embedding=?,
                        altyn_photo_version=?, altyn_synced_at=?
                    WHERE id=?
                    """,
                    (
                        name,
                        filename,
                        blob,
                        version,
                        now,
                        current_id,
                    ),
                )

                local_id = current_id
                action = "updated"

            else:
                cursor = db.execute(
                    """
                    INSERT INTO users(
                        name, photo_file, embedding, created_at,
                        altyn_student_id,
                        altyn_photo_version,
                        altyn_synced_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        filename,
                        blob,
                        now,
                        external_id,
                        version,
                        now,
                    ),
                )

                local_id = int(cursor.lastrowid)
                action = "created"

        committed = True

    finally:
        # Если запись не сохранилась, убираем новый файл.
        if not committed and new_path is not None:
            try:
                new_path.unlink(missing_ok=True)
            except OSError:
                pass

    # Старую фотографию удаляем только после успешной записи.
    if old_file and old_file != filename:
        try:
            (storage.faces_dir / old_file).unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "local_id": local_id,
        "altyn_id": external_id,
        "name": name,
        "action": action,
        "embedding_rebuilt": new_path is not None,
    }


def import_student(client, storage, student_id):
    student = client.get_student(student_id)
    _require_active(student)

    with storage._conn() as db:
        current = db.execute(
            """
            SELECT photo_file, altyn_photo_version
            FROM users WHERE altyn_student_id=?
            """,
            (str(student["id"]),),
        ).fetchone()

    reusable = (
        current is not None
        and current["altyn_photo_version"] == student["photo_version"]
        and (storage.faces_dir / current["photo_file"]).is_file()
    )

    face = None
    embedding = None

    if not reusable:
        photo, content_type = client.get_photo(student)

        extension = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }[content_type]

        with TemporaryDirectory(
            prefix="altyn-import-",
            dir=storage.data_dir,
        ) as directory:
            source = Path(directory) / f"source{extension}"
            source.write_bytes(photo)

            # Используем существующий FaceNet-код SmartGate.
            face, embedding = FaceEngine().enroll_file(source)

    # За время обработки фото статус или фотография могли измениться.
    latest = client.get_student(student["id"])
    _require_active(latest)

    if latest["photo_version"] != student["photo_version"]:
        raise AltynError(
            "Фото обновилось во время обработки. Повтори импорт."
        )

    return _save_student(
        storage,
        latest,
        face,
        embedding,
    )


def main():
    project_dir = Path(__file__).resolve().parent.parent

    token = os.environ.get("ALTYN_API_TOKEN") or getpass.getpass(
        "Токен SmartGate: "
    )

    student_id = input("ID студента Altyn: ").strip()

    client = AltynClient(token)
    storage = Storage(project_dir)

    print("Получаем студента и подготавливаем лицо. Подожди…")

    result = import_student(client, storage, student_id)

    print(
        "Создана запись."
        if result["action"] == "created"
        else "Запись обновлена."
    )

    print("Студент:", result["name"])
    print("ID Altyn:", result["altyn_id"])
    print("Локальный ID SmartGate:", result["local_id"])

    print(
        "Эмбеддинг создан."
        if result["embedding_rebuilt"]
        else "Фото не изменилось: использован сохранённый эмбеддинг."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (AltynError, ValueError, OSError, sqlite3.Error) as error:
        print("Импорт не завершён:", error)
        raise SystemExit(1)

    except (KeyboardInterrupt, EOFError):
        print("\nИмпорт отменён.")
        raise SystemExit(1)