"""Server model Summa.

Rantai: foto -> EasyOCR -> LayoutLMv3 -> penengah hibrida -> JSON.
Disalin dari Notebook E sel 8, dengan satu tambahan: penengah hibrida,
yang di notebook itu tidak dipakai.
"""

import os
import resource  # hanya ada di Linux, dan kontainernya Linux
import tempfile
import time
import uuid

import psutil
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image

import pembanding_aturan as PA
import rantai
from hibrida_penengah import SET_ATURAN_MENANG, penengah

MODEL_ID = os.environ.get("MODEL_ID", "Klissh/layoutlmv3-struk-summa")
KUNCI = os.environ.get("KUNCI", "")
MAKS_BYTE = 10 * 1024 * 1024

# Notebook E memakai 2 utas. Kalau berbeda, angka latensi tidak sebanding.
torch.set_num_threads(int(os.environ.get("UTAS_TORCH", "2")))

MAKS_TOKEN = 512
LANGKAH = 128

_mulai = time.time()

import easyocr  # noqa: E402  (diimpor setelah pencatat waktu, supaya ikut terukur)
from transformers import (  # noqa: E402
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

pembaca = easyocr.Reader(rantai.BAHASA_OCR, gpu=False)

# JEBAKAN: kalau apply_ocr True, processor menjalankan OCR sendiri dan
# menimpa kotak hasil EasyOCR tanpa pesan apa pun.
proc = LayoutLMv3Processor.from_pretrained(MODEL_ID, apply_ocr=False)
proc.image_processor.apply_ocr = False
assert proc.image_processor.apply_ocr is False, "apply_ocr masih True"

model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_ID)
model.to("cpu").eval()
id2label = {int(i): L for i, L in model.config.id2label.items()}

DETIK_MUAT = round(time.time() - _mulai, 1)

app = FastAPI()


def ocr_mentah(jalur):
    """Notebook E sel 6. Urutan dan isinya sama dengan Notebook A sel 7."""
    hasil = pembaca.readtext(jalur, **rantai.PARAM_READTEXT)
    kotak = []
    for item in hasil:
        poly, teks, skor = item[0], item[1], item[2]
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        kotak.append(dict(
            bbox=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            teks=rantai.normalisasi_angka(teks), teks_asli=teks, skor=float(skor)))
    kotak.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return kotak


def jalankan(jalur):
    """Foto masuk, JSON terstruktur keluar. Kembalikan (hasil, waktu)."""
    w = {}

    t = time.perf_counter()
    img = Image.open(jalur).convert("RGB")
    L, T = img.size
    w["1_baca_gambar"] = time.perf_counter() - t

    t = time.perf_counter()
    kotak = ocr_mentah(jalur)
    w["2_ocr"] = time.perf_counter() - t
    if not kotak:
        return None, w

    # Urutan di sini penting. Model memakai urut_baca(), aturan memakai urutan
    # asli. kot berisi objek dict yang sama dengan kotak, hanya diurutkan ulang,
    # jadi label yang ditempel ke objek terbaca dari kedua urutan.
    t = time.perf_counter()
    kot, _ = rantai.urut_baca(kotak)
    kata = [(b["teks"] or "").strip() or "." for b in kot]
    bbx = [rantai.norm_bbox(b["bbox"], L, T) for b in kot]
    enc = proc(img, kata, boxes=bbx,
               truncation=True, padding="max_length", max_length=MAKS_TOKEN,
               stride=LANGKAH, return_overflowing_tokens=True, return_tensors="pt")
    pv = enc["pixel_values"]
    if isinstance(pv, list):
        pv = torch.stack([torch.as_tensor(x) for x in pv])
    n_jendela = enc["input_ids"].shape[0]
    if pv.shape[0] == 1 and n_jendela > 1:
        pv = pv.repeat(n_jendela, 1, 1, 1)
    w["3_siapkan"] = time.perf_counter() - t

    t = time.perf_counter()
    with torch.no_grad():
        keluar = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                       bbox=enc["bbox"], pixel_values=pv)
    peluang = torch.softmax(keluar.logits.float(), dim=-1).numpy()
    w["4_model"] = time.perf_counter() - t

    t = time.perf_counter()
    terbaik = {}
    for j in range(n_jendela):
        wid = enc.word_ids(j)
        for ti, wi in enumerate(wid):
            if wi is None: continue
            p = peluang[j, ti]
            c = float(p.max()); i = int(p.argmax())
            if wi not in terbaik or c > terbaik[wi][0]:
                terbaik[wi] = (c, i, p)
    for i, b in enumerate(kot):
        if i in terbaik:
            c, idl, p = terbaik[i]
            b["_label"] = rantai.dari_bio(id2label[idl])
            b["_yakin"] = c
            # Peluang per label datar, dipakai penengah untuk memilih kandidat
            # total terbaik ketika ada lebih dari satu.
            datar = {}
            for k, L_bio in id2label.items():
                nama = rantai.dari_bio(L_bio)
                if nama == "O": continue
                v = float(p[k])
                if v > datar.get(nama, -1.0):
                    datar[nama] = v
            b["_skor"] = datar
        else:
            b["_label"] = "O"; b["_yakin"] = 0.0; b["_skor"] = {}
    w["5_pasca"] = time.perf_counter() - t

    # --- penengah hibrida. Notebook E tidak memakai bagian ini. ---
    t = time.perf_counter()
    struk = {"kotak": kotak, "lebar": L, "tinggi": T}
    label_model = [b["_label"] for b in kotak]
    skor_model = [b["_skor"] for b in kotak]
    # PA.prediksi dipanggil sekali lagi di sini hanya untuk tahu medan mana yang
    # dimenangkan aturan. hibrida_penengah.py dibekukan, jadi tidak bisa
    # dimintai keterangan itu. Biayanya puluhan milidetik atas ratusan kotak.
    aturan = PA.prediksi(struk)
    akhir = penengah(struk, label_model, skor_model)
    for b, L_aturan, L_akhir in zip(kotak, aturan, akhir):
        b["_label"] = L_akhir
        b["_dari_aturan"] = L_aturan in SET_ATURAN_MENANG and L_aturan == L_akhir
    w["6_penengah"] = time.perf_counter() - t

    hasil = rantai.ke_json(kot, L, T)
    w["total"] = sum(v for k, v in w.items() if k != "total")
    w["n_kotak"] = len(kot)
    w["n_jendela"] = n_jendela
    return (hasil, kot), w


@app.get("/")
def sehat():
    """Dipanggil untuk tahu apakah layanan hidup, dan berapa memorinya."""
    p = psutil.Process()
    return {
        "siap": True,
        "model": MODEL_ID,
        "detik_muat": DETIK_MUAT,
        "rss_gb": round(p.memory_info().rss / 1e9, 2),
        "puncak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
        "berkunci": bool(KUNCI),
    }


@app.post("/ekstrak")
async def ekstrak(berkas: UploadFile = File(...), x_kunci: str = Header(default="")):
    if KUNCI and x_kunci != KUNCI:
        raise HTTPException(status_code=401, detail="kunci salah")

    isi = await berkas.read()
    if not isi:
        raise HTTPException(status_code=400, detail="berkas kosong")
    if len(isi) > MAKS_BYTE:
        raise HTTPException(status_code=413, detail=f"maksimal {MAKS_BYTE // 1024 // 1024} MB")

    jalur = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as f:
            f.write(isi)
            jalur = f.name
        keluar, waktu = jalankan(jalur)
        if keluar is None:
            raise HTTPException(status_code=422, detail="OCR tidak menemukan teks apa pun")
        hasil, kot = keluar
    finally:
        if jalur and os.path.exists(jalur):
            os.unlink(jalur)

    id_struk = uuid.uuid4().hex[:12]
    return {
        "aplikasi": rantai.ke_kontrak_aplikasi(hasil, kot, id_struk, MODEL_ID),
        "mentah": hasil,
        "waktu": {k: round(v, 3) if isinstance(v, float) else v for k, v in waktu.items()},
    }
