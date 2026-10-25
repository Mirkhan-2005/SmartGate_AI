"""SmartGate AI — modern local desktop demo with automatic face enrollment.

Run: python smartgate_app.py  (Python 3.11; see requirements.txt)
"""
from __future__ import annotations

import csv
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from smartgate.storage import Storage
from smartgate.vision import FaceEngine

ROOT = Path(__file__).resolve().parent
BG = "#091423"
SIDE = "#0d1c2f"
PANEL = "#12243a"
PANEL2 = "#162c45"
BORDER = "#263b50"
TEXT = "#ecf4fc"
MUTED = "#9aacbf"
ACCENT = "#4bd6bd"
ACCENT_HOVER = "#36bfa7"
RED = "#ff7981"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SmartGate(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color=BG)
        self.title("SmartGate AI  ·  Распознавание лиц")
        self.geometry("1250x800")
        self.minsize(1060, 690)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.storage = Storage(ROOT)
        self.engine = FaceEngine()
        self.jobs = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SmartGate")
        self.messages: queue.Queue = queue.Queue()
        self.model_ready = False
        self.model_error = None
        self.current_page = "home"
        self.camera = None
        self.camera_mode = None
        self.camera_device = 0
        self.camera_preview = None
        self.latest_frame = None
        self.last_recognition = []
        self.recognition_busy = False
        self.enrollment_busy = False
        self.last_inference_at = 0.0
        self.last_event_time: dict[int, float] = {}
        self.cooldown_seconds = 60.0
        self.last_status = ""
        self.camera_poll_job = None
        self.last_preview_image = None
        self.nav_buttons = {}
        self.page_labels = {
            "home": ("Обзор", "Состояние системы и последние события"),
            "users": ("Пользователи", "Добавляй фото или снимай через камеру — база обновится сама"),
            "recognition": ("Распознавание", "Проверка лиц с камеры в реальном времени"),
            "attendance": ("Журнал", "История распознаваний и экспорт в CSV"),
        }

        self.build_shell()
        self.show_page("home")
        self.after(100, self.poll_messages)
        self.after(250, self.update_camera)
        self.run_background("model", self.engine.load_model)

    def label(self, parent, text, size=14, color=TEXT, weight="normal", **kwargs):
        return ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=size, weight=weight),
                            text_color=color, **kwargs)

    def button(self, parent, text, command, kind="primary", width=150, height=40, **kwargs):
        palette = {
            "primary": (ACCENT, ACCENT_HOVER, BG),
            "secondary": (PANEL2, "#25415e", TEXT),
            "danger": ("#482e3b", "#734052", "#ffadb2"),
        }
        fill, hover, txt = palette[kind]
        return ctk.CTkButton(parent, text=text, command=command, fg_color=fill,
                             hover_color=hover, text_color=txt,
                             font=ctk.CTkFont(size=13, weight="bold"),
                             corner_radius=10, width=width, height=height, **kwargs)

    def box(self, parent, **kwargs):
        return ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=15, border_width=1,
                            border_color=BORDER, **kwargs)

    def build_shell(self):
        self.sidebar = ctk.CTkFrame(self, fg_color=SIDE, width=225, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=21, pady=(30, 36))
        ctk.CTkLabel(brand, text="SG", width=48, height=48, corner_radius=12,
                     fg_color=ACCENT, text_color=BG,
                     font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        title = ctk.CTkFrame(brand, fg_color="transparent")
        title.pack(side="left", padx=11)
        self.label(title, "SMARTGATE", size=17, weight="bold").pack(anchor="w")
        self.label(title, "AI  /  Desktop", size=11, color=MUTED).pack(anchor="w")

        for key, text in [("home", "01   Обзор"), ("users", "02   Пользователи"),
                          ("recognition", "03   Распознавание"), ("attendance", "04   Журнал")]:
            b = ctk.CTkButton(self.sidebar, text=text, height=45, corner_radius=10,
                              anchor="w", font=ctk.CTkFont(size=14),
                              fg_color="transparent", hover_color="#183450", text_color=MUTED,
                              command=lambda page=key: self.show_page(page))
            b.pack(fill="x", padx=14, pady=5)
            self.nav_buttons[key] = b

        ctk.CTkFrame(self.sidebar, fg_color="transparent").pack(expand=True, fill="both")
        support = self.box(self.sidebar)
        support.pack(fill="x", padx=13, pady=16)
        self.label(support, "Локальная версия", size=12, weight="bold").pack(anchor="w", padx=13, pady=(13, 1))
        self.label(support, "Данные сохраняются на ПК", size=11, color=MUTED).pack(anchor="w", padx=13, pady=(0, 13))

        self.main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.main.pack(side="left", fill="both", expand=True)
        heading = ctk.CTkFrame(self.main, fg_color="transparent")
        heading.pack(fill="x", padx=27, pady=(24, 7))
        left = ctk.CTkFrame(heading, fg_color="transparent")
        left.pack(side="left")
        self.page_title = self.label(left, "", size=27, weight="bold")
        self.page_title.pack(anchor="w")
        self.page_subtitle = self.label(left, "", size=13, color=MUTED)
        self.page_subtitle.pack(anchor="w", pady=(3, 0))
        self.model_label = self.label(heading, "●  Модель загружается", size=12, color=MUTED)
        self.model_label.pack(side="right", padx=2)
        self.content = ctk.CTkFrame(self.main, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=25, pady=(15, 14))
        self.footer = self.label(self.main, "Готово к работе", size=12, color=MUTED, anchor="w")
        self.footer.pack(fill="x", padx=29, pady=(0, 16))

    def update_status(self, text: str):
        self.last_status = text
        self.footer.configure(text=text)

    def clear_content(self):
        for child in self.content.winfo_children():
            child.destroy()
        self.camera_preview = None

    def show_page(self, page):
        if self.camera_mode and self.current_page != page:
            self.stop_camera()
        self.current_page = page
        title, subtitle = self.page_labels[page]
        self.page_title.configure(text=title)
        self.page_subtitle.configure(text=subtitle)
        for key, b in self.nav_buttons.items():
            b.configure(fg_color="#193a51" if key == page else "transparent",
                        text_color=ACCENT if key == page else MUTED)
        self.clear_content()
        {"home": self.build_home, "users": self.build_users,
         "recognition": self.build_recognition, "attendance": self.build_attendance}[page]()

    def build_home(self):
        users, attendance_today = self.storage.counts()
        intro = self.box(self.content)
        intro.pack(fill="x", pady=(0, 16))
        self.label(intro, "Биометрическая регистрация без ручной пересборки базы", size=20,
                   weight="bold").pack(anchor="w", padx=22, pady=(21, 6))
        self.label(intro, "Загрузи фото, сделай снимок или удали пользователя. FaceNet-шаблоны обновляются сразу.",
                   color=MUTED, size=13).pack(anchor="w", padx=22, pady=(0, 18))
        row = ctk.CTkFrame(intro, fg_color="transparent")
        row.pack(anchor="w", padx=22, pady=(0, 22))
        self.button(row, "+ Добавить человека", lambda: self.show_page("users"), width=187).pack(side="left")
        self.button(row, "Включить распознавание", lambda: self.show_page("recognition"),
                    kind="secondary", width=210).pack(side="left", padx=12)

        numbers = ctk.CTkFrame(self.content, fg_color="transparent")
        numbers.pack(fill="x", pady=(0, 15))
        for title, val, desc in [("Людей в базе", users, "Всего биометрических шаблонов"),
                                 ("Сегодня распознано", attendance_today, "Событий за сегодняшний день")]:
            card = self.box(numbers)
            card.pack(side="left", expand=True, fill="x", padx=(0, 12))
            self.label(card, title, size=13, color=MUTED).pack(anchor="w", padx=19, pady=(18, 3))
            self.label(card, str(val), size=34, weight="bold", color=ACCENT).pack(anchor="w", padx=19)
            self.label(card, desc, size=12, color=MUTED).pack(anchor="w", padx=19, pady=(2, 18))

        tips = self.box(self.content)
        tips.pack(fill="both", expand=True)
        self.label(tips, "Порядок работы", size=18, weight="bold").pack(anchor="w", padx=22, pady=(20, 12))
        for no, text in [("01", "В разделе «Пользователи» введи имя и загрузи фото либо сделай снимок."),
                         ("02", "Шаблон лица сохранится автоматически в SQLite, кнопка Data не нужна."),
                         ("03", "В разделе «Распознавание» включи камеру и покажи лицо."),
                         ("04", "События появятся в журнале; его можно экспортировать в CSV.")]:
            line = ctk.CTkFrame(tips, fg_color="transparent")
            line.pack(anchor="w", fill="x", padx=22, pady=7)
            self.label(line, no, size=14, color=ACCENT, weight="bold", width=42).pack(side="left")
            self.label(line, text, size=13, color=TEXT, anchor="w").pack(side="left")

    def build_users(self):
        left = self.box(self.content)
        left.pack(side="left", fill="both", expand=False, padx=(0, 13))
        left.configure(width=430)
        left.pack_propagate(False)
        self.label(left, "Регистрация человека", size=18, weight="bold").pack(anchor="w", padx=20, pady=(20, 11))
        self.label(left, "Имя и фамилия", size=12, color=MUTED).pack(anchor="w", padx=20)
        self.name_input = ctk.CTkEntry(left, height=42, corner_radius=9, fg_color=SIDE,
                                       border_color=BORDER, placeholder_text="Например, Ернур Оразбек")
        self.name_input.pack(fill="x", padx=20, pady=(4, 15))
        self.button(left, "Загрузить фотографию (JPG / PNG)", self.upload_user_photo,
                    width=385).pack(fill="x", padx=20, pady=(0, 10))
        device_row = ctk.CTkFrame(left, fg_color="transparent")
        device_row.pack(fill="x", padx=20, pady=(0, 9))
        self.label(device_row, "Камера №", size=12, color=MUTED).pack(side="left")
        self.enroll_camera_selector = ctk.CTkOptionMenu(device_row, values=["0", "1", "2"],
            width=72, fg_color=PANEL2, button_color="#24445c")
        self.enroll_camera_selector.set(str(self.camera_device))
        self.enroll_camera_selector.pack(side="left", padx=12)
        self.camera_enroll_button = self.button(left, "Включить камеру", self.start_registration_camera,
                                               kind="secondary", width=385)
        self.camera_enroll_button.pack(fill="x", padx=20, pady=(0, 10))
        self.capture_button = self.button(left, "Сделать снимок и сохранить", self.capture_user,
                                           kind="secondary", width=385)
        self.capture_button.pack(fill="x", padx=20, pady=(0, 11))
        self.capture_button.configure(state="disabled")
        self.camera_preview = self.label(left, "Превью камеры будет здесь", color=MUTED,
                                         width=385, height=224)
        self.camera_preview.pack(padx=20, pady=(0, 9))
        self.label(left, "На фото должен быть один человек с хорошо видимым лицом.",
                   size=11, color=MUTED, wraplength=370, justify="left").pack(anchor="w", padx=20, pady=(0, 13))

        right = self.box(self.content)
        right.pack(side="left", fill="both", expand=True)
        header = ctk.CTkFrame(right, fg_color="transparent")
        header.pack(fill="x", padx=17, pady=(17, 9))
        self.user_count_label = self.label(header, "Список пользователей", size=17, weight="bold")
        self.user_count_label.pack(side="left")
        self.button(header, "Импорт старых фото", self.import_legacy, kind="secondary",
                    width=152, height=34).pack(side="right")
        self.users_scroll = ctk.CTkScrollableFrame(right, fg_color="transparent")
        self.users_scroll.pack(fill="both", expand=True, padx=15, pady=(0, 12))
        self.refresh_users()

    def refresh_users(self):
        if self.current_page != "users" or not hasattr(self, "users_scroll"):
            return
        rows = self.storage.list_users()
        self.user_count_label.configure(text=f"Пользователи  ·  {len(rows)}")
        for w in self.users_scroll.winfo_children():
            w.destroy()
        if not rows:
            self.label(self.users_scroll, "Пока никого нет. Добавь первого человека слева.",
                       color=MUTED, size=13).pack(pady=30)
            return
        for person in rows:
            item = ctk.CTkFrame(self.users_scroll, fg_color=PANEL2, corner_radius=9, height=69)
            item.pack(fill="x", pady=4)
            try:
                image = Image.open(self.storage.faces_dir / person["photo_file"]).convert("RGB")
                image = ImageOps.fit(image, (47, 47))
                thumb = ctk.CTkImage(light_image=image, dark_image=image, size=(47, 47))
                ctk.CTkLabel(item, image=thumb, text="").pack(side="left", padx=(10, 8), pady=8)
            except Exception:
                self.label(item, "●", size=18, color=ACCENT, width=52).pack(side="left")
            details = ctk.CTkFrame(item, fg_color="transparent")
            details.pack(side="left", fill="x", expand=True)
            self.label(details, person["name"], size=13, weight="bold").pack(anchor="w")
            self.label(details, "Зарегистрирован: " + person["created_at"][:10],
                       size=10, color=MUTED).pack(anchor="w")
            
            self.button(
                item,
                "Удалить",
                lambda user=person: self.delete_user(user),
                kind="danger",
                width=77,
                height=33
            ).pack(side="right", padx=(0, 10))

            self.button(
                item,
                "Изменить",
                lambda user=person: self.rename_user(user),
                kind="secondary",
                width=90,
                height=33
            ).pack(side="right", padx=(5, 6))


    def checked_name(self):
        name = " ".join(self.name_input.get().strip().split())
        if not name:
            messagebox.showwarning("Введите имя", "Сначала укажи имя пользователя.")
            return None
        if any(p["name"].casefold() == name.casefold() for p in self.storage.list_users()):
            messagebox.showwarning("Повтор", "Пользователь с таким именем уже существует.")
            return None
        return name

    def upload_user_photo(self):
        if self.enrollment_busy:
            return
        name = self.checked_name()
        if name is None:
            return
        if not self.model_ready:
            messagebox.showinfo("FaceNet", "Дождись загрузки FaceNet. Статус — справа сверху.")
            return
        path = filedialog.askopenfilename(title="Выбери фотографию", filetypes=[
            ("Фотографии", "*.jpg *.jpeg *.png"), ("Все файлы", "*.*")])
        if path:
            self.enrollment_busy = True
            self.update_status("Обработка фотографии и сохранение шаблона...")
            self.run_background("enroll", self.engine.enroll_file, path, info=name)

    def start_registration_camera(self):
        if self.camera_mode == "enroll":
            self.stop_camera()
            return
        self.camera_device = int(self.enroll_camera_selector.get())
        if self.start_camera("enroll"):
            self.camera_enroll_button.configure(text="Выключить камеру")
            self.capture_button.configure(state="normal")
            self.update_status("Камера работает. Введи имя и нажми «Сделать снимок и сохранить».")

    def capture_user(self):
        if self.enrollment_busy:
            return
        name = self.checked_name()
        if name is None:
            return
        if not self.model_ready:
            messagebox.showinfo("FaceNet", "Дождись загрузки FaceNet.")
            return
        if self.camera_mode != "enroll" or self.latest_frame is None:
            messagebox.showwarning("Камера", "Сначала включи камеру и дождись изображения.")
            return
        self.enrollment_busy = True
        self.run_background("enroll", self.engine.enroll_frame,
                            self.latest_frame.copy(), info=name)
        self.update_status("Снимок сделан. Создаю биометрический шаблон...")


    def rename_user(self, person):
        popup = ctk.CTkToplevel(self)
        popup.title("Изменить имя пользователя")
        popup.geometry("420x210")
        popup.resizable(False, False)
        popup.transient(self)
        popup.grab_set()

        self.label(
            popup,
            "Новое имя и фамилия",
            size=15,
            weight="bold"
        ).pack(pady=(22, 10))

        entry = ctk.CTkEntry(
            popup,
            width=350,
            height=40
        )
        entry.pack(pady=(0, 17))
        entry.insert(0, person["name"])
        entry.focus_set()

        def save_name():
            try:
                updated = self.storage.rename_user(
                    person["id"],
                    entry.get()
                )

                if not updated:
                    messagebox.showwarning(
                        "Не найден",
                        "Пользователь уже удалён.",
                        parent=popup
                    )
                    popup.destroy()
                    self.refresh_users()
                    return

                self.last_recognition = []
                popup.destroy()
                self.refresh_users()

                self.update_status(
                    "Имя изменено. Повторная регистрация не нужна."
                )

            except ValueError as exc:
                messagebox.showerror(
                    "Ошибка переименования",
                    str(exc),
                    parent=popup
                )

            except Exception as exc:
                messagebox.showerror(
                    "Ошибка",
                    str(exc),
                    parent=popup
                )

        self.button(
            popup,
            "Сохранить имя",
            save_name,
            width=175
        ).pack()

        popup.bind(
            "<Return>",
            lambda event: save_name()
        )




    def delete_user(self, person):
        if not messagebox.askyesno("Удаление пользователя",
                f"Удалить «{person['name']}»?\n\nБудут удалены фото, эмбеддинг и связанные события журнала."):
            return
        try:
            if self.storage.delete_user(person["id"]):
                self.last_recognition = []
                self.refresh_users()
                self.update_status("Пользователь и биометрический шаблон удалены.")
        except Exception as exc:
            messagebox.showerror("Ошибка удаления", str(exc))

    def import_legacy(self):
        legacy = ROOT / "fotoPeserta"
        if not legacy.is_dir():
            messagebox.showinfo("Импорт", "Папки fotoPeserta нет рядом с smartgate_app.py. "
                                "Добавь старые фотографии туда и повтори импорт.")
            return
        if not self.model_ready or self.enrollment_busy:
            messagebox.showinfo("FaceNet", "Подожди завершения предыдущей операции и загрузки FaceNet.")
            return
        files = sorted(p for p in legacy.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            messagebox.showinfo("Импорт", "В папке fotoPeserta нет фотографий JPG/PNG.")
            return
        self.enrollment_busy = True
        self.update_status(f"Импортирую фотографии: {len(files)} шт....")
        self.run_background("legacy", self.import_legacy_job, files)

    def import_legacy_job(self, files):
        added, skipped, errors = 0, 0, []
        for file in files:
            name = file.stem.strip()
            if not name:
                skipped += 1
                continue
            if any(row["name"].casefold() == name.casefold() for row in self.storage.list_users()):
                skipped += 1
                continue
            try:
                cropped, embedding = self.engine.enroll_file(file)
                self.storage.add_user(name, cropped, embedding)
                added += 1
            except Exception as exc:
                errors.append(f"{file.name}: {exc}")
        return added, skipped, errors

    def build_recognition(self):
        frame = self.box(self.content)
        frame.pack(fill="both", expand=True)
        control = ctk.CTkFrame(frame, fg_color="transparent")
        control.pack(fill="x", padx=18, pady=(17, 13))
        self.label(control, "Источник камеры", color=MUTED, size=12).pack(side="left", padx=(0, 10))
        self.camera_selector = ctk.CTkOptionMenu(control, values=["0", "1", "2"],
                            width=74, fg_color=PANEL2, button_color="#24445c",
                            button_hover_color="#345c77")
        self.camera_selector.set(str(self.camera_device))
        self.camera_selector.pack(side="left")
        self.start_recognition_button = self.button(control, "Начать распознавание",
                    self.toggle_recognition, width=207)
        self.start_recognition_button.pack(side="left", padx=17)
        self.label(control, "Порог FaceNet: 1.0  ·  лог: не чаще раза в 60 сек.",
                   size=11, color=MUTED).pack(side="right")
        self.camera_preview = self.label(frame, "Камера выключена\n\nНажми «Начать распознавание»",
                                         size=18, color=MUTED, height=450)
        self.camera_preview.pack(fill="both", expand=True, padx=17, pady=(0, 13))
        self.recognition_label = self.label(frame, "Ожидание изображения", color=MUTED, size=14)
        self.recognition_label.pack(anchor="w", padx=22, pady=(0, 16))

    def toggle_recognition(self):
        if self.camera_mode == "recognition":
            self.stop_camera()
            self.start_recognition_button.configure(text="Начать распознавание")
            self.recognition_label.configure(text="Камера остановлена", text_color=MUTED)
            return
        if not self.model_ready:
            messagebox.showinfo("FaceNet", "Модель ещё загружается, подожди.")
            return
        if not self.storage.list_users():
            messagebox.showinfo("Нет пользователей", "Сначала зарегистрируй хотя бы одного человека.")
            return
        self.camera_device = int(self.camera_selector.get())
        if self.start_camera("recognition"):
            self.start_recognition_button.configure(text="Остановить камеру")
            self.update_status("Распознавание включено. Данные сохраняются автоматически.")

    def build_attendance(self):
        header = self.box(self.content)
        header.pack(fill="x", pady=(0, 12))
        row = ctk.CTkFrame(header, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=17)
        self.label(row, "История распознаваний", size=18, weight="bold").pack(side="left")
        self.button(row, "Экспорт CSV", self.export_csv, width=127).pack(side="right", padx=(9, 0))
        self.button(row, "Обновить", lambda: self.show_page("attendance"), kind="secondary",
                    width=94).pack(side="right")
        table = self.box(self.content)
        table.pack(fill="both", expand=True)
        headings = ctk.CTkFrame(table, fg_color="transparent")
        headings.pack(fill="x", padx=15, pady=(13, 8))
        for text, width in [("Человек", 245), ("Дата и время", 230), ("Расстояние", 150)]:
            self.label(headings, text, size=12, color=MUTED, width=width, anchor="w").pack(side="left", padx=5)
        scroll = ctk.CTkScrollableFrame(table, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        entries = self.storage.list_attendance()
        if not entries:
            self.label(scroll, "Пока нет событий. Распознай человека в разделе «Распознавание».",
                       color=MUTED).pack(pady=35)
        for idx, entry in enumerate(entries):
            line = ctk.CTkFrame(scroll, fg_color=PANEL2 if idx % 2 == 0 else PANEL,
                                corner_radius=7, height=42)
            line.pack(fill="x", pady=3)
            for val, width, color in [(entry["name"], 245, TEXT),
                                      (entry["occurred_at"].replace("T", " "), 230, MUTED),
                                      (f"{entry['distance']:.3f}", 150, ACCENT)]:
                self.label(line, str(val), size=12, color=color, width=width,
                           anchor="w").pack(side="left", padx=5, pady=10)

    def export_csv(self):
        if not self.storage.list_attendance(1):
            messagebox.showinfo("Журнал", "В журнале пока нет записей.")
            return
        name = time.strftime("smartgate_attendance_%Y-%m-%d.csv")
        file = filedialog.asksaveasfilename(title="Экспорт журнала", defaultextension=".csv",
                   initialfile=name, filetypes=[("CSV", "*.csv")])
        if not file:
            return
        try:
            self.storage.export_csv(file)
            self.update_status(f"Журнал сохранён: {file}")
        except Exception as exc:
            messagebox.showerror("Ошибка экспорта", str(exc))

    def start_camera(self, mode: str) -> bool:
        self.stop_camera()
        index = self.camera_device
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(index)
        if not camera.isOpened():
            camera.release()
            messagebox.showerror("Камера недоступна", f"Не удалось открыть камеру {index}. "
                    "Закрой Zoom/Teams, проверь разрешения Windows или выбери индекс 1/2.")
            return False
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.camera = camera
        self.camera_mode = mode
        self.last_recognition = []
        return True

    def stop_camera(self):
        if self.camera is not None:
            self.camera.release()
        self.camera = None
        self.camera_mode = None
        self.latest_frame = None
        self.last_recognition = []
        if self.current_page == "users" and hasattr(self, "capture_button"):
            try:
                self.capture_button.configure(state="disabled")
                self.camera_enroll_button.configure(text="Включить камеру")
            except Exception:
                pass

    def update_camera(self):
        # Only Tk main thread touches UI and webcam; inference runs in a worker.
        if self.camera is not None:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                self.stop_camera()
                self.update_status("Ошибка чтения камеры. Попробуй другой индекс камеры.")
            else:
                self.latest_frame = frame.copy()
                preview = frame.copy()
                if self.camera_mode == "recognition":
                    for rect, user_id, name, distance in self.last_recognition:
                        x, y, w, h = rect
                        color = (125, 216, 75) if user_id is not None else (110, 110, 248)
                        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(preview, "MATCH" if user_id is not None else "UNKNOWN",
                                    (x, max(23, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, color, 2)
                    now = time.monotonic()
                    if (self.model_ready and not self.recognition_busy and
                            now - self.last_inference_at >= 0.85):
                        self.last_inference_at = now
                        self.recognition_busy = True
                        refs = self.storage.face_references()
                        self.run_background("recognition", self.engine.recognize,
                                            frame.copy(), refs)
                if self.camera_preview is not None and self.camera_preview.winfo_exists():
                    target = (785, 440) if self.camera_mode == "recognition" else (380, 224)
                    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    pil.thumbnail(target, Image.Resampling.BILINEAR)
                    canvas = Image.new("RGB", target, (9, 20, 35))
                    canvas.paste(pil, ((target[0] - pil.width) // 2, (target[1] - pil.height) // 2))
                    self.last_preview_image = ctk.CTkImage(light_image=canvas, dark_image=canvas, size=target)
                    self.camera_preview.configure(image=self.last_preview_image, text="")
        self.after(48, self.update_camera)

    def run_background(self, kind, function, *args, info=None):
        fut = self.jobs.submit(function, *args)
        def callback(future):
            try:
                self.messages.put((kind, "ok", future.result(), info))
            except Exception as exc:
                self.messages.put((kind, "error", str(exc), info))
        fut.add_done_callback(callback)

    def poll_messages(self):
        try:
            while True:
                kind, status, payload, info = self.messages.get_nowait()
                if kind == "model":
                    if status == "ok":
                        self.model_ready = True
                        self.model_label.configure(text="●  FaceNet готов", text_color=ACCENT)
                        self.update_status("Модель FaceNet загружена. Можно регистрировать людей.")
                    else:
                        self.model_error = payload
                        self.model_label.configure(text="●  Ошибка FaceNet", text_color=RED)
                        self.update_status("Ошибка загрузки модели: " + payload)
                        messagebox.showerror("FaceNet", "Не удалось загрузить FaceNet:\n" + payload)
                elif kind == "enroll":
                    self.enrollment_busy = False
                    if status == "ok":
                        cropped, embedding = payload
                        try:
                            self.storage.add_user(info, cropped, embedding)
                            if self.current_page == "users":
                                self.name_input.delete(0, "end")
                            if self.camera_mode == "enroll":
                                self.stop_camera()
                            self.refresh_users()
                            self.update_status(f"Пользователь «{info}» сохранён. База обновлена автоматически.")
                            messagebox.showinfo("Пользователь добавлен", f"«{info}» зарегистрирован.\n"
                                "FaceNet-шаблон сразу добавлен в базу.")
                        except Exception as exc:
                            messagebox.showerror("Ошибка сохранения", str(exc))
                    else:
                        messagebox.showerror("Не удалось добавить лицо", payload)
                        self.update_status("Ошибка регистрации: " + payload)
                elif kind == "legacy":
                    self.enrollment_busy = False
                    if status == "ok":
                        added, skipped, errors = payload
                        self.refresh_users()
                        self.update_status(f"Импорт: добавлено {added}, пропущено {skipped}, ошибок {len(errors)}.")
                        report = f"Добавлено: {added}\nПропущено: {skipped}\nОшибок: {len(errors)}"
                        if errors:
                            report += "\n\n" + "\n".join(errors[:8])
                        messagebox.showinfo("Импорт завершён", report)
                    else:
                        messagebox.showerror("Импорт", payload)
                elif kind == "recognition":
                    self.recognition_busy = False
                    if status == "ok" and self.camera_mode == "recognition":
                        self.last_recognition = payload
                        active = {p["id"] for p in self.storage.list_users()}
                        names = []
                        for rect, user_id, name, distance in payload:
                            if user_id is None or user_id not in active:
                                continue
                            names.append(name)
                            now = time.monotonic()
                            if now - self.last_event_time.get(user_id, -1e10) >= self.cooldown_seconds:
                                if self.storage.add_attendance(user_id, distance):
                                    self.last_event_time[user_id] = now
                        if self.current_page == "recognition":
                            if names:
                                self.recognition_label.configure(text="Распознано: " + ", ".join(names),
                                                                 text_color=ACCENT)
                            else:
                                self.recognition_label.configure(
                                    text="Лицо не найдено или совпадение не подтверждено",
                                    text_color=MUTED)
                    elif status == "error":
                        self.update_status("Ошибка распознавания: " + payload)
        except queue.Empty:
            pass
        self.after(100, self.poll_messages)

    def on_close(self):
        self.stop_camera()
        self.jobs.shutdown(wait=False, cancel_futures=True)
        self.destroy()


if __name__ == "__main__":
    SmartGate().mainloop()
