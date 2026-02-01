# 🧠 MedicalVision-AI

Profesyonel **Desktop Uygulaması** — **Beyin BT/MR görüntülerini AI ile otomatik analiz eden sistem.**  
Bu proje, derin öğrenme tabanlı bir ensemble model ile medikal görüntüleri sınıflandırır ve **hiperakut/akut**, **subakut** ve **normal/kronik** lezyon fazlarını tespit eder. :contentReference[oaicite:1]{index=1}

---

## 🔍 Genel Bakış

MedicalVision-AI, sağlık profesyonellerinin beyin görüntülerini daha hızlı ve isabetli değerlendirmesine yardımcı olmak için tasarlanmış bir masaüstü uygulamasıdır.  
📌 Hem **CT (BT)** hem **MR** görüntülerini destekler. :contentReference[oaicite:2]{index=2}

---

## 💡 Özellikler

### 🖥️ Kullanıcı Arayüzü
- PyQt5 ile oluşturulmuş **modern GUI**  
- Intuitif kullanıcı deneyimi  
- Tekli ve toplu dosya analizi  

### 🤖 AI Modeli
- **5-fold ensemble öğrenimi**  
- **CNN + LSTM + SE (attention)** mimarisi  
- **Focal Loss** ile dengesiz sınıf problemlerine çözüm  
- GPU/CPU otomatik algılama :contentReference[oaicite:3]{index=3}

### 📊 Format Desteği
- DICOM (.dcm)  
- .png / .jpg / .jpeg / .bmp  
- Toplu işleme (Batch Analysis) :contentReference[oaicite:4]{index=4}

---

## 📦 Kurulum & Çalıştırma

Önkoşullar:
- Python 3.8+  
- PyTorch, PyQt5 vb. paketler

1. Repo’yu klonla:
```bash
git clone https://github.com/tariktoplu/MedicalVision-AI.git
cd MedicalVision-AI```
...
2. Sanal ortam oluştur:
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
3. Paketleri Yükle:
```bash
pip install -r requirements.txt
Models/
├── best_model_fold_0.pt
├── best_model_fold_1.pt
...
5. Uygulamayı Çalıştır:

python medical_analyzer.py

🧠 Teknik Detaylar

Model mimarisi:
Input → 3D CNN → SE Attention → MaxPool → LSTM → Dense → Output (3 sınıf)

📄 Lisans

MIT License
