import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class IrisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система идентификации по радужной оболочке")

        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
        self.root.bind('<F11>', lambda e: self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen')))

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        try:
            self.model = load_model('E:/HWPython/iris-bio-model.h5')
            self.model_input_size = self.model.input_shape[1:3]

            df, labels, images = self.load_dataset('E:/HWPython/CASIA/CASIA-Iris-Thousand')
            self.raw_labels = df['Label'].astype(str).values

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.raw_labels)
            self.class_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}

        except Exception as e:
            messagebox.showerror("H81:0", f"5 C40;>AL 703@C78BL <>45;L: {str(e)}")
            self.root.destroy()
            return

        self.image = None
        self.tk_image = None

        self.create_widgets(screen_width, screen_height)

    def create_widgets(self, screen_width, screen_height):

        font_style = ('Arial', 14) if screen_width > 1920 else ('Arial', 10)
        title_font = ('Arial', 16, 'bold') if screen_width > 1920 else ('Arial', 12, 'bold')

        main_frame = tk.Frame(self.root, bg='#f5f5f5')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        self.image_frame = tk.LabelFrame(
            main_frame,
            text="Изображение глаза",
            bg='#f5f5f5',
            font=title_font
        )
        self.image_frame.pack(fill='both', expand=True, pady=(0, 20))

        self.image_label = tk.Label(self.image_frame, bg='#e0e0e0')
        self.image_label.pack(fill='both', expand=True)

        self.button_frame = tk.Frame(main_frame, bg='#f5f5f5')
        self.button_frame.pack(fill='x', pady=(0, 20))

        btn_width = 25 if screen_width > 1920 else 20
        btn_height = 3 if screen_width > 1920 else 2
        btn_font = ('Arial', 12) if screen_width > 1920 else ('Arial', 10)

        self.load_button = tk.Button(
            self.button_frame,
            text="Загрузить изображение",
            command=self.load_image,
            width=btn_width,
            height=btn_height,
            bg='#4CAF50',
            fg='white',
            font=btn_font
        )
        self.load_button.pack(side='left', expand=True, padx=10)

        self.identify_button = tk.Button(
            self.button_frame,
            text="Идентифицировать",
            command=self.identify_eye,
            width=btn_width,
            height=btn_height,
            state="disabled",
            bg='#2196F3',
            fg='white',
            font=btn_font
        )
        self.identify_button.pack(side='left', expand=True, padx=10)

        exit_fullscreen_btn = tk.Button(
            self.button_frame,
            text="Оконный режим (Esc)",
            command=lambda: self.root.attributes('-fullscreen', False),
            width=btn_width,
            height=btn_height,
            bg='#607D8B',
            fg='white',
            font=btn_font
        )
        exit_fullscreen_btn.pack(side='left', expand=True, padx=10)

        self.result_frame = tk.LabelFrame(
            main_frame,
            text="Результаты идентификации",
            bg='#f5f5f5',
            font=title_font
        )
        self.result_frame.pack(fill='x', pady=(0, 10))

        self.result_label = tk.Label(
            self.result_frame,
            text="Пожалуйста, загрузите изображение глаза для идентификации",
            wraplength=screen_width - 50,
            justify="left",
            bg='#f5f5f5',
            font=font_style
        )
        self.result_label.pack(fill='x', padx=10, pady=10)

        self.access_frame = tk.LabelFrame(
            main_frame,
            text="Статус доступа",
            bg='#f5f5f5',
            font=title_font
        )
        self.access_frame.pack(fill='x')

        self.access_label = tk.Label(
            self.access_frame,
            text="Доступ не проверен",
            wraplength=screen_width - 50,
            justify="left",
            bg='#f5f5f5',
            font=font_style
        )
        self.access_label.pack(fill='x', padx=10, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение глаза",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp"), ("Все файлы", "*.*")]
        )

        if file_path:
            try:
                self.image = Image.open(file_path)
                self.display_image()
                self.identify_button["state"] = "normal"
                self.result_label.config(text="Изображение загружено. Нажмите 'Идентифицировать'")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки изображения: {str(e)}")

    def display_image(self):

        max_width = self.root.winfo_screenwidth() - 100
        max_height = int(self.root.winfo_screenheight() * 0.6)

        img = self.image.copy()
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_image)

    def identify_eye(self):
        if self.image is None:
            messagebox.showwarning("Предупреждение", "Пожалуйста, сначала загрузите изображение")
            return

        try:
            img = self.image.resize(self.model_input_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            img_array = img_array / 255.0

            prediction = self.model.predict(img_array)

            predicted_class_id = np.argmax(prediction[0])
            confidence = np.max(prediction[0])

            text_label = self.class_mapping[predicted_class_id]

            result_text = (
                f"Идентифицированный пользователь: {text_label}\n"
                f"Точность распознавания: {confidence * 100:.2f}%"
            )

            if confidence >= 0.8:
                access_text = f"ДОСТУП РАЗРЕШЁН ✅ (Добро пожаловать, {text_label.split('_')[0]}!)"
                self.access_label.config(fg='green')
            else:
                access_text = "ДОСТУП ЗАПРЕЩЁН ❌ (Низкая уверенность распознавания)"
                self.access_label.config(fg='red')

            self.result_label.config(text=result_text)
            self.access_label.config(text=access_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при идентификации: {str(e)}")

    def load_dataset(self, path):

        labels = []
        images = []

        for folder in os.listdir(path):
            for lr in os.listdir(path + '/' + folder):
                for image in os.listdir(path + '/' + folder + '/' + lr):
                    if image.endswith('b') is False:
                        images.append(path + '/' + folder + '/' + lr + '/' + image)
                        labels.append(folder + '-' + lr)

        df = pd.DataFrame(list(zip(labels, images)), columns=['Label', 'ImagePath'])
        return df, labels, images

    def preprocess_labels(self, df):

        labels = df['Label'].astype(str)
        le = LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        return labels


if __name__ == "__main__":
    root = tk.Tk()
    app = IrisApp(root)
    root.mainloop()