FROM python:3.11-slim

# opencv, yang ditarik easyocr, butuh dua pustaka sistem ini.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bobot diunduh saat build, lalu ikut masuk ke dalam image. Tanpa ini,
# setiap kali layanan bangun dari tidur ia mengunduh ulang sekitar 600 MB.
ARG MODEL_ID=microsoft/layoutlmv3-base
ENV MODEL_ID=${MODEL_ID}
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)" \
 && python -c "import os; from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification; \
LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False); \
LayoutLMv3ForTokenClassification.from_pretrained(os.environ['MODEL_ID'])"

COPY app.py .

# Railway menentukan nomor port lewat $PORT. Mengikat ke port tetap
# membuat layanan terlihat mati dari luar.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
