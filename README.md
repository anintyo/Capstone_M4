# 🎯 Purwa YOLO - Object Detection App

Aplikasi Object Detection berbasis YOLOv12 dengan Streamlit untuk deteksi Construction Equipment, Vehicle, dan Fruit.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Poetry](https://img.shields.io/badge/poetry-dependency%20management-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![YOLO](https://img.shields.io/badge/YOLOv12-ultralytics-green)

## 📁 Project Structure

```
purwa_yolo/
├── models/                          # Folder untuk model .pt (gitignored)
│   └── best_construction.pt
├── src/purwa_yolo/                  # Source code aplikasi
│   ├── __init__.py
│   ├── main.py                      # Streamlit app
│   └── sample_images/               # Contoh gambar untuk demo
│       └── Construction-Safety-Engineer.jpg
├── training_code/                   # Jupyter notebooks untuk training
│   ├── Train YOLOv12 Construction.ipynb
│   ├── Train YOLOv12 Vehicle.ipynb
│   └── Train_YOLOv12_Construction.ipynb  # Jupyter notebooks hasil training
├── tests/
├── pyproject.toml                   # Poetry dependencies
├── packages.txt                     # System packages to resolve libGL on Streamlit Cloud
└── README.md
```

## 🎓 Training Model di Google Colab

### Step 1: Upload Notebook ke Google Colab

1. Buka [Google Colab](https://colab.research.google.com/)
2. Klik **File** → **Upload notebook**
3. Upload salah satu notebook dari folder `training_code/`:
   - `Train YOLOv12 Construction.ipynb`
   - `Train YOLOv12 Vehicle.ipynb`

### Step 2: Setup GPU di Colab

1. Klik **Runtime** → **Change runtime type**
2. Pilih **Hardware accelerator**: **T4 GPU** atau **A100 GPU** (jika available)
3. Klik **Save**

### Step 3: Jalankan Training

Jalankan semua cell secara berurutan:

```bash
# Cell 1: Install dependencies
!pip install ultralytics roboflow

# Cell 2: Import libraries
from ultralytics import YOLO
from roboflow import Roboflow

# Cell 3: Download dataset dari Roboflow
# (Kode akan ada di notebook)

# Cell 4: Training
model = YOLO('yolov8n.pt')  # atau yolov12n.pt
results = model.train(
    data='path/to/data.yaml',
    epochs=30,
    imgsz=320,
    batch=2,
    name='my_model'
)

# Cell 5: Validasi
model.val()

# Cell 6: Test inference
model.predict('test_image.jpg', save=True)
```

### Step 4: Download Model

Setelah training selesai:

1. Model akan tersimpan di `/content/runs/detect/my_model/weights/best.pt`
2. Download file dengan klik kanan → **Download**
3. Rename sesuai kebutuhan:
   - `best_construction.pt`
   - `best_vehicle.pt`
   - `best_fruit.pt`

### Step 5: Pindahkan Model ke Project

Setelah download, pindahkan file `.pt` ke folder `models/` di project:

```bash
# Di terminal local
mv ~/Downloads/best.pt ./models/best_construction.pt
```

## 🚀 Local Development Setup

### Prerequisites

- Python 3.11+
- Poetry (untuk dependency management)
- pyenv (recommended untuk version management)

### Installation Steps

#### 1. Clone Repository

```bash
git clone https://github.com/hermansh-id/purwa_yolo.git
cd purwa_yolo
```

#### 2. Setup Python Version

```bash
# Install Python 3.11.9
pyenv install 3.11.9

# Set local Python version
pyenv shell 3.11.9
```

#### 3. Install Dependencies dengan Poetry

```bash
# Install Poetry jika belum ada
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Atau install manual dependencies
poetry add numpy streamlit supervision ultralytics pillow
```

#### 4. Tambahkan Model Files

Pastikan file model `.pt` ada di folder `models/`:

```bash
ls models/
# Output:
# best_construction.pt
# best_vehicle.pt
# best_fruit.pt
```

> **Note**: File model tidak di-commit ke git karena ukurannya besar. 
> Anda perlu training sendiri atau mendapatkan dari tim.

## 🎯 Cara Menggunakan App

### Jalankan Aplikasi

```bash
# Menggunakan Poetry
poetry run streamlit run src/purwa_yolo/main.py

# Atau aktifkan virtual environment dulu
poetry shell
streamlit run src/purwa_yolo/main.py
```

### Akses Aplikasi

Buka browser dan akses:
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Langkah Penggunaan

1. **Pilih Model**: Pilih use case dari dropdown:
   - Construction Equipment
   - Vehicle
   - Fruit

2. **Upload Image**: Klik "Browse files" dan pilih gambar
   - Format: JPG, JPEG, PNG, WEBP

   **Atau gunakan contoh gambar yang sudah tersedia** — klik tombol "✅ Gunakan Ini" di bawah thumbnail tanpa perlu upload manual:

   ![Sample Image Demo](src/purwa_yolo/sample_images/Construction-Safety-Engineer.jpg)
   > *Contoh: Pelatihan APD di area konstruksi — safety engineer mendemonstrasikan penggunaan harness*

3. **Detect Objects**: Klik tombol "🔍 Detect Objects"

4. **Lihat Hasil**:
   - Gambar dengan bounding boxes
   - Metrics jumlah objek per class
