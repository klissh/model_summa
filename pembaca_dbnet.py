"""Pembaca produksi Model v2: DBNet + PP-OCRv4 (rapidocr_onnxruntime) lalu pecah_baris.

Menggantikan CRAFT + EasyOCR sebagai pembaca bawaan. Tidak memuat torch dan tidak
memuat easyocr, jadi peladen hidup tanpa model CRAFT dan RAM-nya turun.

Logika deterministik (HARGA_PAJAK, pecah_baris, baca_dbnet, siapkan_citra) disalin
dari eksperimen_kaggle/subkotak_dbnet.py, yang diukur oleh
eksperimen_kaggle/04_uji_ujung_ke_ujung_v2b.py (11 struk uji3: 41/66 = 62,1%).
Kalau salah satunya berubah, ubah keduanya dan ukur ulang.

Satu-satunya beda dari berkas eksperimen: siapkan_citra() meluruskan orientasi EXIF
(ImageOps.exif_transpose) lebih dulu. Pada foto tanpa EXIF miring keluarannya
identik dengan yang diukur.

Kembali ke Model v1: PEMBACA=craft bersama MODEL_ID=Klissh/layoutlmv3-struk-summa-ocr.
Model v1 dilatih pada kotak CRAFT, jadi dua setelan itu harus berganti bersamaan.
"""
import io
import re

import numpy as np

SISI_MAKS, MUTU = 1600, 85           # sama dengan yang diukur: sisi terpanjang <= 1600, JPEG 85
HARGA_PAJAK = re.compile(r"(\(?-?\d{1,3}(?:[.,]\d{3})*[.,]\d{2})([A-Za-z]{1,3})(\)?)")
GAP_HURUF = 0.25      # celah antar huruf >= 0,25 tinggi baris dianggap batas kata

_mesin = None


def muat():
    """Siapkan mesin DBNet + PP-OCRv4. Dipanggil saat peladen hidup; gagal keras kalau pustaka tidak ada."""
    global _mesin
    if _mesin is None:
        from rapidocr_onnxruntime import RapidOCR
        _mesin = RapidOCR()
    return _mesin


def quad_ke_bbox(quad):
    a = np.asarray(quad, dtype=float)
    return [int(a[:, 0].min()), int(a[:, 1].min()), int(np.ceil(a[:, 0].max())), int(np.ceil(a[:, 1].max()))]


def _token_huruf(huruf, tinggi):
    """huruf: [(ch, [x0,y0,x1,y1])] urut kiri ke kanan. Token = rangkaian huruf tanpa spasi dan tanpa celah besar."""
    toks, cur = [], []
    for ch, bb in huruf:
        if ch.isspace():
            if cur:
                toks.append(cur)
                cur = []
            continue
        if cur and bb[0] - cur[-1][1][2] >= GAP_HURUF * tinggi:
            toks.append(cur)
            cur = []
        cur.append((ch, bb))
    if cur:
        toks.append(cur)
    return toks


def _bb_huruf(hs):
    return [min(h[1][0] for h in hs), min(h[1][1] for h in hs), max(h[1][2] for h in hs), max(h[1][3] for h in hs)]


def pecah_baris(kotak):
    """kotak: [{teks, bbox, yakin, huruf?}] -> sub-kotak kata. Deterministik, tanpa label.
    Dengan `huruf` (kotak per huruf dari rapidocr return_word_box): koordinat KATA = gabungan kotak hurufnya.
    Tanpa `huruf`: cadangan proporsional, lebar dibagi menurut posisi huruf di dalam kotak baris.
    Kode pajak yang menempel pada harga dua desimal ("1.80SR", "3.71ZRL") dipisah dalam kedua jalur."""
    keluar = []
    for b in kotak:
        yakin = b.get("yakin", 1.0)
        if b.get("huruf"):
            tinggi = max(1, b["bbox"][3] - b["bbox"][1])
            for tok in _token_huruf(b["huruf"], tinggi):
                t = "".join(c for c, _ in tok)
                mm = HARGA_PAJAK.fullmatch(t)
                bag = [tok]
                if mm:
                    n1 = len(mm.group(1))
                    bag = [tok[:n1], tok[n1:n1 + len(mm.group(2))]]
                for h in bag:
                    keluar.append({"teks": "".join(c for c, _ in h), "bbox": _bb_huruf(h), "yakin": yakin})
            continue
        t = b["teks"] or ""
        x0, y0, x1, y1 = b["bbox"]
        n = max(1, len(t))
        for m in re.finditer(r"\S+", t):
            tok, mulai = m.group(0), m.start()
            mm = HARGA_PAJAK.fullmatch(tok)
            bagian = [(tok, 0)]
            if mm:
                depan = mm.group(1)
                bagian = [(depan, 0), (mm.group(2), len(depan))]     # ")" penutup, kalau ada, dibuang
            for s_, off in bagian:
                a = x0 + (x1 - x0) * (mulai + off) / n
                z = x0 + (x1 - x0) * (mulai + off + len(s_)) / n
                keluar.append({"teks": s_, "bbox": [int(a), y0, max(int(a) + 1, int(np.ceil(z))), y1], "yakin": yakin})
    return keluar


def siapkan_citra(sumber):
    """EXIF diluruskan, RGB, sisi terpanjang <= 1600 (LANCZOS), JPEG mutu 85, dibaca kembali.
    `sumber`: jalur berkas atau objek file. Mengembalikan citra PIL."""
    from PIL import Image, ImageOps
    with Image.open(sumber) as im:
        g = ImageOps.exif_transpose(im).convert("RGB")
        p = max(g.size)
        if p > SISI_MAKS:
            s = SISI_MAKS / p
            g = g.resize((round(g.size[0] * s), round(g.size[1] * s)), Image.LANCZOS)
    buf = io.BytesIO()
    g.save(buf, format="JPEG", quality=MUTU)
    buf.seek(0)
    with Image.open(buf) as k:
        return k.convert("RGB")


def baca_dbnet(mesin, img):
    """Kotak baris DBNet + PP-OCRv4 beserta kotak per huruf (return_word_box=True, yang juga menghasilkan spasi pada teks).
    rapidocr menganggap larik numpy berformat BGR."""
    bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
    hasil, _ = mesin(bgr, use_det=True, use_cls=False, use_rec=True, return_word_box=True)
    out = []
    for r in (hasil or []):
        q, teks, sk = r[0], r[1], r[2]
        huruf = None
        if len(r) >= 5 and r[3] and r[4] and len(r[3]) == len(teks) and len(r[4]) == len(teks) and all(len(c) == 1 for c in r[4]):
            huruf = [(c, quad_ke_bbox(qq)) for c, qq in zip(r[4], r[3]) if ord(c) < 128]
        t = re.sub(r"[^\x00-\x7f]", "", teks or "")
        if t.strip():
            out.append({"teks": t, "bbox": quad_ke_bbox(q), "yakin": float(sk), "huruf": huruf})
    out.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return out


def kotak_dbnet(img, normalisasi, mesin=None):
    """Citra PIL -> kotak kata berbentuk sama dengan app.ocr_mentah (bbox, teks, teks_asli, skor).
    `normalisasi` = rantai.normalisasi_angka, disuntikkan supaya modul ini tidak mengimpor rantai."""
    mesin = mesin or muat()
    L, T = img.size
    kotak = []
    for b in pecah_baris(baca_dbnet(mesin, img)):
        x0, y0, x1, y1 = b["bbox"]
        bb = [min(L, max(0, x0)), min(T, max(0, y0)), min(L, max(0, x1)), min(T, max(0, y1))]
        if bb[2] <= bb[0] or bb[3] <= bb[1] or not b["teks"].strip():
            continue
        kotak.append(dict(bbox=bb, teks=normalisasi(b["teks"]), teks_asli=b["teks"], skor=float(b["yakin"])))
    kotak.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return kotak
