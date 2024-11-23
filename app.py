import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk
import os
import re
import unicodedata
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import random

def ensure_model_directories():
    """Create necessary directories for models if they don't exist"""
    directories = [
        'models/face',
        'models/text',
        'data/emotions/trening',
        'data/emotions/walidacja',
        'data/emotions/test'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

class EmotionDetector:
    def __init__(self):
        ensure_model_directories()
        self.model_path = 'models/text/emotion_model.h5'
        self.vectorizer_path = 'models/text/vectorizer.pkl'
        self.encoder_path = 'models/text/label_encoder.pkl'
        
        if not all(os.path.exists(path) for path in [self.model_path, self.vectorizer_path, self.encoder_path]):
            print("Pre-trained text model not found. Training new model...")
            self.train_new_model()
        else:
            print("Loading pre-trained text model...")
            self.load_components()

    def load_components(self):
        try:
            print("Wczytywanie komponentów...")
            self.model = tf.keras.models.load_model(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.label_encoder = joblib.load(self.encoder_path)
            print("Komponenty wczytane pomyślnie")
        except Exception as e:
            print(f"Błąd podczas wczytywania: {e}")
            self.train_new_model()

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        text = re.sub(r'[^a-z ]', '', text)
        text = ' '.join(text.split())
        return text.strip()

    def train_new_model(self, progress_callback=None):
        try:
            # Set seeds for reproducibility
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)
            
            print("Trenowanie nowego modelu...")
            df = pd.read_csv('data/pl_emotions_from_text_2_0.csv')
            df = df.dropna(subset=['text', 'emotion'])
            
            texts = df['text'].astype(str).map(self.preprocess_text).values
            labels = df['emotion'].values
            
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
            
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X = self.vectorizer.fit_transform(texts).toarray()
            
            # Split into train/val/test
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, encoded_labels, test_size=0.3, random_state=42
            )
            
            # Split temp into val/test
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
            
            self.X_test = X_test
            self.y_test = y_test
            
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            epochs = 5
            self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: progress_callback(epoch, epochs) if progress_callback else None
                    )
                ]
            )
            
            # After training, get predictions for confusion matrix
            self.test_predictions = np.argmax(self.model.predict(X_test), axis=1)
            
            self.save_components()
            
        except Exception as e:
            raise Exception(f"Błąd podczas trenowania: {e}")

    def save_components(self):
        self.model.save(self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        joblib.dump(self.label_encoder, self.encoder_path)
        print("Model i komponenty zapisane")

    def predict(self, text):
        try:
            processed_text = self.preprocess_text(text)
            if not processed_text:
                raise ValueError("Pusty tekst po przetworzeniu")
                
            vectorized_text = self.vectorizer.transform([processed_text]).toarray()
            prediction = self.model.predict(vectorized_text)
            predicted_label = np.argmax(prediction)
            return self.label_encoder.inverse_transform([predicted_label])[0]
        except Exception as e:
            raise Exception(f"Błąd podczas predykcji: {e}")

    def get_confusion_matrix(self):
        # Calculate confusion matrix only when requested
        df = pd.read_csv('data/pl_emotions_from_text_2_0.csv')
        df = df.dropna(subset=['text', 'emotion'])
        
        texts = df['text'].astype(str).map(self.preprocess_text).values
        labels = df['emotion'].values
        
        encoded_labels = self.label_encoder.transform(labels)
        X = self.vectorizer.transform(texts).toarray()
        
        # Split data
        _, X_test, _, y_test = train_test_split(
            X, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Get predictions
        test_predictions = np.argmax(self.model.predict(X_test), axis=1)
        
        return confusion_matrix(y_test, test_predictions), self.label_encoder.classes_

class ImageDetectorHelper:
    def __init__(self, paths):
        ensure_model_directories()
        
        # Paths setup
        self.train_dir = paths['train_dir']
        self.val_dir = paths['val_dir'] 
        self.test_dir = paths['test_dir']
        self.model_path = 'models/face/emotion_detection_model.h5'

        # Hyperparameters
        self.img_height = 96
        self.img_width = 96
        self.batch_size = 64
        self.learning_rate = 0.0005
        self.epochs = 100

        # Initialize model and generators
        self.setup_generators()
        self.model = self.load_or_create_model()

    def setup_generators(self):
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )

        # Validation/Test data normalization only
        val_datagen = ImageDataGenerator(rescale=1./255)

        # Setup generators
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        self.val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.test_generator = val_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(len(os.listdir(self.train_dir)), activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_or_create_model(self):
        try:
            if os.path.exists(self.model_path):
                print(f"Loading model from: {self.model_path}")
                model = load_model(self.model_path)
                model.compile(
                    optimizer=Adam(learning_rate=self.learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                return model
            else:
                print("No pre-trained model found. Creating new model...")
                return self.create_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model instead...")
            return self.create_model()

    def train_model(self, callback=None):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        if callback:
            callbacks.append(callback)

        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )

        self.model.save(self.model_path)
        return history

    def predict_image(self, img_path):
        try:
            img_array = self.preprocess_image(img_path)
            prediction = self.model.predict(img_array)[0]
            top_indices = prediction.argsort()[-2:][::-1]
            class_labels = list(self.train_generator.class_indices.keys())
            return [(class_labels[i], prediction[i]) for i in top_indices]
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

    def preprocess_image(self, file_path):
        img = image.load_img(file_path, target_size=(self.img_height, self.img_width))
        img_array = image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def show_confusion_matrix(self):
        predictions = self.model.predict(self.test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        cm = confusion_matrix(y_true, y_pred)
        class_names = list(self.train_generator.class_indices.keys())
        return cm, class_names

class MainApplication:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Detection System")
        self.root.geometry("800x600")
        self.setup_styles()
        
        # Initialize detectors
        self.text_detector = EmotionDetector()
        self.setup_image_paths()
        self.image_detector = ImageDetectorHelper({
            'train_dir': self.train_dir,
            'val_dir': self.val_dir,
            'test_dir': self.test_dir,
            'model_path': self.model_path
        })
        
        self.create_main_menu()

    def setup_styles(self):
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Helvetica', 24, 'bold'), padding=20)
        style.configure('SubHeader.TLabel', font=('Helvetica', 16), padding=10)
        style.configure('Normal.TButton', font=('Helvetica', 12), padding=10)
        style.configure('Action.TButton', font=('Helvetica', 12, 'bold'), padding=10)
        style.configure('Result.TLabel', font=('Helvetica', 14, 'bold'), padding=5)
        
    def create_main_menu(self):
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", pady=20)
        
        ttk.Label(
            header_frame, 
            text="Emotion Detection System",
            style='Header.TLabel'
        ).pack(pady=20)

        # Main content
        content_frame = ttk.Frame(self.root)
        content_frame.pack(expand=True, fill="both", padx=40, pady=20)

        ttk.Label(
            content_frame, 
            text="Select Detection Mode:",
            style='SubHeader.TLabel'
        ).pack(pady=20)

        button_frame = ttk.Frame(content_frame)
        button_frame.pack(expand=True)

        ttk.Button(
            button_frame,
            text="Text Emotion Detection",
            command=self.show_text_detector,
            style='Action.TButton'
        ).pack(pady=10, ipadx=20)

        ttk.Button(
            button_frame,
            text="Image Emotion Detection",
            command=self.show_image_detector,
            style='Action.TButton'
        ).pack(pady=10, ipadx=20)

    def show_text_detector(self):
        text_window = tk.Toplevel(self.root)
        text_window.title("Text Emotion Detection")
        text_window.geometry("900x800")
        
        # Create main canvas with scrollbar
        main_canvas = tk.Canvas(text_window)
        scrollbar = ttk.Scrollbar(text_window, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)
        
        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding=20)
        main_frame.pack(expand=True, fill="both")

        # Rest of your existing content...
        # [Keep all the existing widgets but pack them into main_frame]
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Header
        ttk.Label(
            main_frame,
            text="Text Emotion Analysis",
            style='Header.TLabel'
        ).pack(pady=20)

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding=10)
        input_frame.pack(fill="x", padx=20, pady=10)

        text_input = scrolledtext.ScrolledText(
            input_frame, 
            width=60, 
            height=8,
            font=('Helvetica', 12)
        )
        text_input.pack(pady=10)

        # Control buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill="x", pady=10)

        ttk.Button(
            button_frame,
            text="Analyze",
            command=lambda: self.analyze_text(text_input, result_var),
            style='Action.TButton'
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame,
            text="Clear",
            command=lambda: self.clear_text(text_input, result_var),
            style='Normal.TButton'
        ).pack(side="left", padx=5)

        # Results section
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        result_frame.pack(fill="x", padx=20, pady=10)

        result_var = tk.StringVar()
        ttk.Label(
            result_frame,
            textvariable=result_var,
            style='Result.TLabel'
        ).pack(pady=10)

        # Model controls section
        model_frame = ttk.LabelFrame(main_frame, text="Model Controls", padding=10)
        model_frame.pack(fill="x", padx=20, pady=10)

        control_buttons = ttk.Frame(model_frame)
        control_buttons.pack(fill="x", pady=10)

        ttk.Button(
            control_buttons,
            text="Train New Model",
            command=lambda: self.train_text_model(text_window, progress_var),
            style='Normal.TButton'
        ).pack(side="left", padx=5)

        ttk.Button(
            control_buttons,
            text="Show Confusion Matrix",
            command=lambda: self.show_text_confusion_matrix(cm_frame),
            style='Normal.TButton'
        ).pack(side="left", padx=5)

        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            model_frame,
            variable=progress_var,
            maximum=100
        )
        progress_bar.pack(fill="x", pady=10)

        # Confusion matrix frame
        cm_frame = ttk.Frame(main_frame)
        cm_frame.pack(fill="both", expand=True, pady=10)

    def show_image_detector(self):
        image_window = tk.Toplevel(self.root)
        image_window.title("Image Emotion Detection")
        image_window.geometry("900x800")
        
        main_frame = ttk.Frame(image_window, padding=20)
        main_frame.pack(expand=True, fill="both")

        # Header
        ttk.Label(
            main_frame,
            text="Image Emotion Analysis",
            style='Header.TLabel'
        ).pack(pady=20)

        # Control buttons
        button_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        button_frame.pack(fill="x", padx=20, pady=10)

        ttk.Button(
            button_frame,
            text="Upload Image",
            command=self.upload_and_predict_image,
            style='Action.TButton'
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame,
            text="Start Webcam",
            command=self.start_webcam,
            style='Action.TButton'
        ).pack(side="left", padx=5)

        # Model controls
        model_frame = ttk.LabelFrame(main_frame, text="Model Controls", padding=10)
        model_frame.pack(fill="x", padx=20, pady=10)

        ttk.Button(
            model_frame,
            text="Train Model",
            command=lambda: self.train_image_model(cm_frame),
            style='Normal.TButton'
        ).pack(side="left", padx=5)

        ttk.Button(
            model_frame,
            text="Show Confusion Matrix",
            command=lambda: self.show_image_confusion_matrix(cm_frame),
            style='Normal.TButton'
        ).pack(side="left", padx=5)

        # Confusion matrix / visualization frame
        cm_frame = ttk.Frame(main_frame)
        cm_frame.pack(fill="both", expand=True, pady=10)

    def setup_image_paths(self):
        self.train_dir = "data/emotions/trening"
        self.val_dir = "data/emotions/walidacja" 
        self.test_dir = "data/emotions/test"
        self.model_path = "models/face/emotion_detection_model.h5"

    def run(self):
        self.root.mainloop()

    def analyze_text(self, text_input, result_var):
        try:
            text = text_input.get("1.0", tk.END).strip()
            if not text:
                messagebox.showwarning("Warning", "Please enter some text")
                return
            
            result = self.text_detector.predict(text)
            result_var.set(f"Detected emotion: {result}")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def clear_text(self, text_input, result_var):
        text_input.delete("1.0", tk.END)
        result_var.set("")

    def train_text_model(self, window, progress_var):
        try:
            def update_progress(epoch, total_epochs):
                progress = (epoch + 1) / total_epochs * 100
                progress_var.set(progress)
                window.update_idletasks()

            self.text_detector.train_new_model(progress_callback=update_progress)
            messagebox.showinfo("Success", "Model training completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")

    def show_text_confusion_matrix(self, cm_frame):
        try:
            # Clear existing content
            for widget in cm_frame.winfo_children():
                widget.destroy()

            # Create matrix frame
            matrix_frame = ttk.LabelFrame(cm_frame, text="Confusion Matrix", padding=10)
            matrix_frame.pack(fill='both', expand=True, padx=20, pady=10)

            # Get and display confusion matrix
            cm, class_names = self.text_detector.get_confusion_matrix()
            
            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(
                ax=ax,
                cmap='Blues',
                values_format='d',
                xticks_rotation=45,
                text_kw={'size': 14, 'weight': 'bold'}
            )
            
            ax.set_title('Text Emotion Classification Results', fontsize=16, pad=20, weight='bold')
            ax.set_xlabel('Predicted Label', fontsize=14, labelpad=10)
            ax.set_ylabel('True Label', fontsize=14, labelpad=10)
            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=matrix_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display confusion matrix: {str(e)}")

    def show_image_confusion_matrix(self, cm_frame):
        try:
            # Clear previous plot
            for widget in cm_frame.winfo_children():
                widget.destroy()

            # Get confusion matrix data
            cm, class_names = self.image_detector.show_confusion_matrix()
            
            # Create figure with larger size
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create confusion matrix display with better formatting
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(
                ax=ax,
                cmap='Blues',
                values_format='d',
                xticks_rotation=45,
                text_kw={'size': 12}  # Larger text size
            )
            
            # Customize plot appearance
            ax.set_title('Confusion Matrix - Image Emotions', fontsize=14, pad=20)
            ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
            ax.set_ylabel('True Label', fontsize=12, labelpad=10)
            
            # Adjust layout to prevent cutting off
            plt.tight_layout()

            # Create canvas with proper sizing
            canvas = FigureCanvasTkAgg(fig, master=cm_frame)
            canvas.draw()
            
            # Pack canvas with proper fill and expand
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True, padx=10, pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to show confusion matrix: {str(e)}")

    def train_image_model(self, cm_frame):
        try:
            history = self.image_detector.train_model()
            messagebox.showinfo("Success", "Model training completed!")
            
            # Show training history plot
            for widget in cm_frame.winfo_children():
                widget.destroy()
                
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history.history['accuracy'], label='Training Accuracy')
            ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax.set_title('Model Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=cm_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")

    def upload_and_predict_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        if file_path:
            try:
                emotion = self.image_detector.predict_image(file_path)
                messagebox.showinfo("Prediction", f"Detected emotion: {emotion}")
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_frame, (self.image_detector.img_width, self.image_detector.img_height))
            img_array = resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get prediction
            prediction = self.image_detector.model.predict(img_array)
            emotion = list(self.image_detector.train_generator.class_indices.keys())[np.argmax(prediction)]
            
            # Display result
            cv2.putText(frame, f"Emotion: {emotion}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = MainApplication()
    app.run()