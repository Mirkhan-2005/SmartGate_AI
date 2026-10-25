from pathlib import Path

import numpy as np
import pytest

from smartgate.storage import Storage


def fake_person():
    return np.full((160, 160, 3), 142, dtype=np.uint8), np.linspace(0, 1, 512, dtype=np.float32)


def test_add_list_delete_cascade(tmp_path):
    db = Storage(tmp_path)
    image, emb = fake_person()
    alice = db.add_user("Ернур", image, emb)
    assert db.list_users()[0]["name"] == "Ернур"
    photo = db.faces_dir / db.list_users()[0]["photo_file"]
    assert photo.exists()
    assert len(db.face_references()) == 1
    uid, name, arr = db.face_references()[0]
    assert uid == alice and name == "Ернур"
    np.testing.assert_array_equal(arr, emb)
    assert db.add_attendance(alice, 0.47)
    assert len(db.list_attendance()) == 1
    assert db.delete_user(alice)
    assert db.list_users() == []
    assert db.face_references() == []
    assert db.list_attendance() == []
    assert not photo.exists()
    assert not db.delete_user(alice)


def test_duplicate_case_insensitive(tmp_path):
    db = Storage(tmp_path)
    image, emb = fake_person()
    db.add_user("Test Student", image, emb)
    with pytest.raises(ValueError, match="уже зарегистрирован"):
        db.add_user("test student", image, emb)
    assert len(list(db.faces_dir.iterdir())) == 1


def test_csv_export(tmp_path):
    db = Storage(tmp_path)
    image, emb = fake_person()
    uid = db.add_user("Студент", image, emb)
    assert db.add_attendance(uid, 0.75)
    path = tmp_path / "journal.csv"
    db.export_csv(path)
    contents = path.read_text("utf-8-sig")
    assert "Студент" in contents and "0.75" in contents
