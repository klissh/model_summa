"""Server model Summa, tahap 1: hanya memuat model dan melaporkan memori.

Tujuan tahap ini satu saja, yaitu menjawab apakah EasyOCR dan LayoutLMv3
muat di paket Railway yang dipakai. Endpoint ekstraksi belum ada, karena
rantai pra-pemrosesannya harus disalin persis dari Notebook E sel 8, 9, 10.
Menulis ulang rantai itu dari ingatan akan membuat keluaran server berbeda
dari yang diukur di BAB IV.
"""

import os
import resource  # hanya ada di Linux, dan kontainernya Linux
import time

import psutil
from fastapi import FastAPI

MODEL_ID = os.environ.get("MODEL_ID", "microsoft/layoutlmv3-base")

_mulai = time.time()

import easyocr  # noqa: E402  (diimpor setelah pencatat waktu, supaya ikut terukur)
from transformers import (  # noqa: E402
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

pembaca = easyocr.Reader(["en"], gpu=False)

# JEBAKAN: kalau apply_ocr True, processor menjalankan OCR sendiri dan
# menimpa kotak hasil EasyOCR tanpa pesan apa pun.
proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
assert proc.image_processor.apply_ocr is False, "apply_ocr masih True"

model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_ID)
model.eval()

DETIK_MUAT = round(time.time() - _mulai, 1)

app = FastAPI()


@app.get("/")
def sehat():
    """Dipanggil untuk tahu apakah layanan hidup, dan berapa memorinya."""
    p = psutil.Process()
    return {
        "siap": True,
        "model": MODEL_ID,
        "detik_muat": DETIK_MUAT,
        "rss_gb": round(p.memory_info().rss / 1e9, 2),
        "puncak_gb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2
        ),
    }
