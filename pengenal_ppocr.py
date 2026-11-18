"""Pengenal huruf PP-OCRv4 di atas kotak CRAFT, menggantikan CRNN EasyOCR.

Yang diganti HANYA teks di dalam kotak. Koordinat kotak hasil CRAFT EasyOCR
dipakai apa adanya, jadi sebaran fitur ruang yang masuk ke LayoutLMv3 tidak
berubah sedikit pun dan model tidak perlu dilatih ulang untuk ini.

Dasarnya terukur, lihat `dataset/uji3/BUKTI-BATAS-SUMMA.md` bagian 7b:

    pengenal                      plafon OCR 66 medan    waktu 11 struk
    EasyOCR 1.7.2 (CRNN)          43 dari 66 = 65,2%     sekitar 65 dtk
    PP-OCRv4 pengenal saja        55 dari 66 = 83,3%     20 dtk

Teks berubah walau koordinatnya tidak, dan model dilatih pada teks EasyOCR
beserta galatnya. Pergeseran itu kecil dan arahnya baik, tetapi BELUM diuji
pada data bersih mana pun. Karena itu lapisan ini **dimatikan secara bawaan**.

    PENGENAL=easyocr   (bawaan) jalur lama, tidak satu baris pun berubah
    PENGENAL=ppocr     kotak CRAFT dibaca ulang oleh PP-OCRv4

Dua sikap terhadap kegagalan, dan keduanya dipilih dengan sengaja:

  * Salah setel GAGAL KERAS saat muat. Kalau PENGENAL=ppocr tetapi pustakanya
    tidak ada, peladen menolak hidup. Diam-diam kembali ke EasyOCR membuat
    angka yang dilaporkan tidak bisa dipercaya, sebab tidak ada yang tahu
    pengenal mana yang sebenarnya berjalan.
  * Satu kotak yang gagal dibaca MENURUN HALUS. Teks EasyOCR untuk kotak itu
    dipertahankan dan kejadiannya dihitung, sebab menjatuhkan satu struk utuh
    karena satu potongan kecil lebih buruk daripada satu kata yang keliru.

shortcut: tinggi potongan 48 piksel diambil dari tinggi masukan baku pengenal
PP-OCR, bukan dari sapuan. Sesuaikan kalau versi pengenalnya berganti.
"""
import os

# Kotak dari width_ths=0.0 adalah PECAHAN token, sering hanya beberapa piksel
# tingginya, sedangkan pengenal PP-OCR dilatih pada potongan setinggi satu
# baris. Tanpa pembesaran ini perbandingannya tidak adil bagi PP-OCR.
TINGGI_REC = 48
# Supaya huruf di tepi kotak tidak terpotong oleh pembulatan koordinat.
MARGIN = 2
# Potongan di bawah ukuran ini tidak mungkin memuat huruf.
SISI_MIN = 3

_mesin = None
_gagal = 0


def aktif():
    return os.environ.get("PENGENAL", "easyocr").strip().lower() == "ppocr"


def nama():
    return "PP-OCRv4 rec (rapidocr_onnxruntime)" if aktif() else "EasyOCR CRNN"


def muat():
    """Siapkan pengenal. Dipanggil sekali saat peladen hidup, bukan per permintaan.

    Sengaja gagal keras: kalau PENGENAL=ppocr tetapi pustakanya tidak ada,
    lebih baik peladen tidak hidup daripada hidup dengan pengenal yang bukan
    seperti yang tertulis di setelan.
    """
    global _mesin
    if not aktif() or _mesin is not None:
        return _mesin
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:
        raise RuntimeError(
            "PENGENAL=ppocr tetapi rapidocr_onnxruntime tidak terpasang. "
            "Pasang 'rapidocr_onnxruntime' atau setel PENGENAL=easyocr.") from e
    _mesin = RapidOCR()
    return _mesin


def _baca_potongan(potongan):
    """Teks dan keyakinan satu potongan, atau (None, None) kalau gagal."""
    import numpy as np

    hasil, _ = _mesin(np.array(potongan), use_det=False, use_cls=False,
                      use_rec=True)
    if not hasil:
        return None, None
    # Tanpa deteksi, bentuknya [[teks, yakin], ...]
    teks = " ".join(str(r[0]) for r in hasil).strip()
    skor = [float(r[1]) for r in hasil if len(r) > 1]
    return (teks or None), (sum(skor) / len(skor) if skor else None)


def baca_ulang(sumber, kotak, normalisasi=None):
    """Ganti `teks` setiap kotak dengan bacaan PP-OCR. Bbox tidak disentuh.

    `sumber` boleh jalur berkas atau gambar PIL, sebab jalur penyelamat
    bekerja pada gambar yang sudah diputar dan bukan pada berkas aslinya.
    `normalisasi` adalah rantai.normalisasi_angka, disuntikkan supaya modul
    ini tidak perlu mengimpor rantai.

    Kotak dikembalikan sebagai objek yang SAMA, diubah di tempat, sebab sisa
    rantai menempelkan label ke objek kotak dan bukan ke posisinya.
    """
    global _gagal
    if not aktif() or not kotak:
        return kotak
    from PIL import Image

    muat()
    if isinstance(sumber, str):
        with Image.open(sumber) as im:
            gambar = im.convert("RGB")
    else:
        gambar = sumber.convert("RGB")
    W, H = gambar.size

    for b in kotak:
        x0, y0, x1, y1 = (int(round(v)) for v in b["bbox"])
        x0 = max(0, x0 - MARGIN)
        y0 = max(0, y0 - MARGIN)
        x1 = min(W, x1 + MARGIN)
        y1 = min(H, y1 + MARGIN)
        if x1 - x0 < SISI_MIN or y1 - y0 < SISI_MIN:
            continue
        potongan = gambar.crop((x0, y0, x1, y1))
        if potongan.height != TINGGI_REC:
            skala = TINGGI_REC / potongan.height
            potongan = potongan.resize(
                (max(8, int(round(potongan.width * skala))), TINGGI_REC),
                Image.LANCZOS)
        try:
            teks, skor = _baca_potongan(potongan)
        except Exception:
            # Satu potongan gagal tidak boleh menjatuhkan satu struk utuh.
            _gagal += 1
            continue
        if teks is None:
            continue
        b["teks_asli"] = teks
        b["teks"] = normalisasi(teks) if normalisasi else teks
        if skor is not None:
            b["skor"] = skor
    return kotak


def gagal():
    """Berapa potongan yang gagal dibaca sejak peladen hidup."""
    return _gagal
