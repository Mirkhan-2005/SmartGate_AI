import hashlib
import json
import re
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


BASE_URL = "https://altyn.dpdns.org"
STUDENTS_PATH = "/api/integrations/smartgate/students"
TIMEOUT_SECONDS = 20
MAX_PHOTO_BYTES = 5_000_000


class AltynError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(
        self, req, fp, code, msg, headers, newurl
    ):
        return None


def _student_id(value):
    value = str(value).strip()

    if (
        not re.fullmatch(r"[1-9][0-9]{0,18}", value)
        or int(value) > 9223372036854775807
    ):
        raise AltynError("Некорректный ID студента Altyn.")

    return value


class AltynClient:
    def __init__(self, token):
        if (
            not isinstance(token, str)
            or not 32 <= len(token) <= 512
            or not token.isascii()
            or any(char.isspace() for char in token)
        ):
            raise AltynError(
                "Некорректный формат сервисного токена."
            )

        self._token = token
        self._opener = build_opener(NoRedirects())

    def _get(self, path, *, accept, limit):
        request = Request(
            BASE_URL + path,
            headers={
                "Authorization": f"SmartGate {self._token}",
                "Accept": accept,
            },
            method="GET",
        )

        try:
            with self._opener.open(
                request,
                timeout=TIMEOUT_SECONDS,
            ) as response:
                if response.status != 200:
                    raise AltynError(
                        "Altyn вернул неожиданный статус.",
                        response.status,
                    )

                content_type = response.headers.get_content_type()
                body = response.read(limit + 1)

        except HTTPError as error:
            status = error.code
            error.close()

            messages = {
                401: "Altyn отклонил сервисный токен.",
                403: "Altyn запретил доступ.",
                404: "Студент или фотография недоступны.",
                409: "Фото изменилось. Повтори получение студента.",
                429: "Лимит запросов Altyn. Повтори позже.",
                503: "Сервис Altyn временно недоступен.",
            }

            raise AltynError(
                messages.get(
                    status,
                    f"Ошибка Altyn: HTTP {status}.",
                ),
                status,
            ) from None

        except (URLError, TimeoutError, OSError, HTTPException):
            raise AltynError(
                "Не удалось подключиться к Altyn."
            ) from None

        if len(body) > limit:
            raise AltynError(
                "Ответ Altyn превышает допустимый размер."
            )

        return body, content_type

    def get_student(self, student_id):
        student_id = _student_id(student_id)

        body, content_type = self._get(
            f"{STUDENTS_PATH}/{student_id}",
            accept="application/json",
            limit=64_000,
        )

        if content_type != "application/json":
            raise AltynError(
                "Вместо данных студента получен другой формат."
            )

        try:
            student = json.loads(body.decode("utf-8-sig"))
        except (ValueError, UnicodeError):
            raise AltynError(
                "Не удалось прочитать ответ Altyn."
            ) from None

        if (
            not isinstance(student, dict)
            or str(student.get("id")) != student_id
        ):
            raise AltynError(
                "Ответ не соответствует запрошенному студенту."
            )

        if (
            student.get("student_status")
            not in ("active", "inactive", "blocked")
            or type(student.get("access_allowed")) is not bool
        ):
            raise AltynError(
                "Не удалось определить статус доступа."
            )

        for field in ("first_name", "last_name"):
            value = student.get(field)

            if not isinstance(value, str) or len(value) > 150:
                raise AltynError(
                    "Некорректный формат имени студента."
                )

        return student

    def get_photo(self, student):
        if (
            student.get("student_status") != "active"
            or student.get("access_allowed") is not True
        ):
            raise AltynError(
                "Студенту запрещён доступ: фото не импортируем."
            )

        if not student.get("photo_url"):
            raise AltynError(
                "У студента нет доступной фотографии."
            )

        student_id = _student_id(student.get("id"))
        version = student.get("photo_version")

        if (
            not isinstance(version, str)
            or not re.fullmatch(r"[0-9a-f]{64}", version)
        ):
            raise AltynError("Некорректная версия фотографии.")

        # Используем только известный адрес Altyn.
        # Не отправляем токен на произвольный photo_url.
        body, content_type = self._get(
            f"{STUDENTS_PATH}/{student_id}/photo?version={version}",
            accept="*/*",
            limit=MAX_PHOTO_BYTES,
        )

        if content_type not in (
            "image/jpeg",
            "image/png",
            "image/webp",
        ):
            raise AltynError(
                "Сервер вернул неподдерживаемый формат фото."
            )

        if (
            not body
            or hashlib.sha256(body).hexdigest() != version
        ):
            raise AltynError(
                "Версия скачанной фотографии не совпала."
            )

        return body, content_type