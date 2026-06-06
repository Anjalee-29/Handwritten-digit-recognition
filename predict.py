import numpy as np
import keras
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk

# Load the trained model
model = keras.models.load_model("mnist_model.keras")

class DigitRecognizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digit Recognizer")
        self.root.resizable(False, False)

        # ── Tabs (Draw / Upload) ──────────────────────────────────────────
        self.mode = tk.StringVar(value="draw")

        tab_frame = tk.Frame(self.root)
        tab_frame.grid(row=0, column=0, columnspan=3, pady=(10, 0))

        tk.Radiobutton(tab_frame, text="✏️  Draw", font=('Arial', 13), variable=self.mode,
                       value="draw", command=self.switch_mode).pack(side="left", padx=20)
        tk.Radiobutton(tab_frame, text="🖼️  Upload Image", font=('Arial', 13), variable=self.mode,
                       value="upload", command=self.switch_mode).pack(side="left", padx=20)

        # ── Draw Panel ────────────────────────────────────────────────────
        self.draw_frame = tk.Frame(self.root)
        self.draw_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

        self.canvas_size = 280
        self.canvas = tk.Canvas(self.draw_frame, width=self.canvas_size, height=self.canvas_size,
                                bg='black', cursor='cross')
        self.canvas.pack()

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_img = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        draw_btn_frame = tk.Frame(self.draw_frame)
        draw_btn_frame.pack(pady=8)
        tk.Button(draw_btn_frame, text="Predict", font=('Arial', 13), bg='green', fg='white',
                  command=self.predict_draw).pack(side="left", padx=8)
        tk.Button(draw_btn_frame, text="Clear", font=('Arial', 13), bg='red', fg='white',
                  command=self.clear_canvas).pack(side="left", padx=8)

        # ── Upload Panel ──────────────────────────────────────────────────
        self.upload_frame = tk.Frame(self.root)

        self.preview_label = tk.Label(self.upload_frame, text="No image selected",
                                      font=('Arial', 12), fg='gray',
                                      width=35, height=10, relief="groove")
        self.preview_label.pack(pady=10)

        upload_btn_frame = tk.Frame(self.upload_frame)
        upload_btn_frame.pack(pady=8)
        tk.Button(upload_btn_frame, text="Browse Image", font=('Arial', 13), bg='steelblue', fg='white',
                  command=self.upload_image).pack(side="left", padx=8)
        tk.Button(upload_btn_frame, text="Predict", font=('Arial', 13), bg='green', fg='white',
                  command=self.predict_upload).pack(side="left", padx=8)

        self.uploaded_image = None

        # ── Result labels (shared) ────────────────────────────────────────
        self.result_label = tk.Label(self.root, text="Draw or upload a digit to get started!",
                                     font=('Arial', 16))
        self.result_label.grid(row=2, column=0, columnspan=3, pady=4)

        self.confidence_label = tk.Label(self.root, text="", font=('Arial', 12), fg='gray')
        self.confidence_label.grid(row=3, column=0, columnspan=3, pady=2)

        tk.Button(self.root, text="Quit", font=('Arial', 12),
                  command=self.root.quit).grid(row=4, column=0, columnspan=3, pady=10)

        self.root.mainloop()

    # ── Mode switching ─────────────────────────────────────────────────────
    def switch_mode(self):
        if self.mode.get() == "draw":
            self.upload_frame.grid_forget()
            self.draw_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10)
        else:
            self.draw_frame.grid_forget()
            self.upload_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10)
        self.result_label.config(text="Draw or upload a digit to get started!")
        self.confidence_label.config(text="")

    # ── Draw methods ───────────────────────────────────────────────────────
    def paint(self, event):
        x, y = event.x, event.y
        brush = 12
        self.canvas.create_oval(x - brush, y - brush, x + brush, y + brush,
                                fill='white', outline='white')
        self.draw_img.ellipse([x - brush, y - brush, x + brush, y + brush], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_img = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw or upload a digit to get started!")
        self.confidence_label.config(text="")

    def predict_draw(self):
        self._predict(self.image)

    # ── Upload methods ─────────────────────────────────────────────────────
    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select a digit image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if not path:
            return

        # Open as RGB so we can detect digit color before converting
        img = Image.open(path).convert("RGB")
        self.uploaded_image = self._preprocess_upload(img)

        # Show preview of the processed image so user sees what model sees
        preview = self.uploaded_image.resize((200, 200), Image.NEAREST)
        photo = ImageTk.PhotoImage(preview)
        self.preview_label.config(image=photo, text="", width=200, height=200)
        self.preview_label.image = photo
        self.result_label.config(text="Image loaded — click Predict!")
        self.confidence_label.config(text="")

    def _preprocess_upload(self, img):
        """
        Preprocess uploaded image to match MNIST format (white digit on black background).

        Handles:
        - Colored digits (red, blue, green etc.) on dark backgrounds
        - Black digits on white backgrounds
        - Noisy or non-centered images
        """
        arr = np.array(img).astype('float32')  # shape: (H, W, 3)

        # 1. Find the dominant color channel of the digit
        #    Compare max channel value per pixel to detect colored strokes
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)

        # Pick the channel with most signal (brightest overall = digit color channel)
        max_channel = np.argmax([r_mean, g_mean, b_mean])
        channel = arr[:, :, max_channel]

        # 2. Convert to PIL grayscale using the dominant channel
        img_gray = Image.fromarray(channel.astype(np.uint8))

        # 3. Check if background is dark or light
        #    If mean is low → digit is bright on dark bg (already MNIST-like)
        #    If mean is high → digit is dark on light bg → invert
        mean_val = np.mean(channel)
        if mean_val > 127:
            img_gray = ImageOps.invert(img_gray)

        # 4. Threshold to binary
        img_gray = img_gray.point(lambda p: 255 if p > 60 else 0)

        # 5. Crop tightly around the digit
        bbox = img_gray.getbbox()
        if bbox:
            img_gray = img_gray.crop(bbox)

        # 6. Add padding (like MNIST)
        img_gray = ImageOps.expand(img_gray, border=20, fill=0)

        # 7. Resize to 28x28
        img_gray = img_gray.resize((28, 28), Image.LANCZOS)

        return img_gray

    def predict_upload(self):
        if self.uploaded_image is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        self._predict(self.uploaded_image)

    # ── Shared prediction ──────────────────────────────────────────────────
    def _predict(self, img):
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img).astype('float32') / 255
        img_array = img_array.reshape(1, 28, 28, 1)

        predictions = model.predict(img_array, verbose=0)
        digit = np.argmax(predictions)
        confidence = predictions[0][digit] * 100

        self.result_label.config(text=f"Predicted Digit: {digit}", font=('Arial', 20, 'bold'))
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

DigitRecognizer()
