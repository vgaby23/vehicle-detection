# Vehicle Detection using YOLOv12n 🚗🏎️🚛

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![YOLOv12](https://img.shields.io/badge/YOLO-v12n-orange)

A lightweight, real-time vehicle detection model powered by the **YOLOv12 Nano (YOLOv12n)** architecture.

## 🌟 Features
* **Real-time Inference:** Achieves high FPS on both GPU and CPU.
* **Lightweight:** Uses the YOLOv12 `nano` variant for minimal memory footprint.
* **Multi-Class Detection:** Detects 3 vehicle types (e.g., Cars, Trucks, Van).

## 📂 Project Structure

```text
├── main.py              # main script to run streamlit interface
├── predict.py           # functions to load model and count objects
├── yolov12.pt           # Pre-trained and fine-tuned weights (.pt)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vgaby23/vehicle-detection.git
   cd vehicle-detection
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset & Performance

* **Dataset:** Containing 23339 images
* **mAP50-95:** ~87.6%
