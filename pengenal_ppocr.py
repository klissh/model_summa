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

    PENGENAL=ppocr     (bawaan) deteksi CRAFT saja, lalu dibaca PP-OCRv4
    PENGENAL=easyocr   jalur lama, CRNN EasyOCR 1.7.2
    UTAS_PENGENAL=4    utas onnxruntime untuk pengenal ini

Saat ppocr dipakai, pengenal CRNN EasyOCR TIDAK dijalankan sama sekali:
`kotak_craft` memakai deteksi saja, sebab teksnya akan ditimpa seluruhnya.

Kebijakan kegagalan dibedakan menurut SIAPA yang memilih, sebab sejak ppocr
jadi bawaan, pustaka yang tidak terpasang tidak lagi berarti salah setel:

  * PENGENAL=ppocr DISETEL EKSPLISIT, pustakanya tidak ada: GAGAL KERAS.
    Operator meminta pengenal tertentu, dan diam-diam memakai yang lain
    membuat angka yang dilaporkan tidak bisa dipercaya.
  * ppocr dipakai dari BAWAAN, pustakanya tidak ada: MUNDUR ke EasyOCR dengan
    peringatan, dan `nama()` beserta endpoint sehat menyebutkan kemundurannya.
    Satu roda yang gagal dipasang tidak boleh berarti peladen mati.
  * Satu potongan yang gagal dibaca MENURUN HALUS dan kejadiannya dihitung,
    sebab menjatuhkan satu struk utuh karena satu potongan kecil lebih buruk
    daripada satu kata yang keliru.

Huruf di luar ASCII dibuang dari keluaran pengenal. Model PP-OCRv4 bawaan
rapidocr adalah model Tionghoa ditambah Inggris, dan pada potongan berderau ia
mengeluarkan huruf Han. Terukur pada 1326 kotak uji3: 17 kotak memuat huruf
non-ASCII dan 12 di antaranya huruf Han, sedangkan EasyOCR nol. Struk Malaysia
tidak memuat huruf Han dan LayoutLMv3 dilatih pada teks beraksara Latin, jadi
huruf itu derau yang masuk langsung ke masukan model.

shortcut: tinggi potongan 48 piksel diambil dari tinggi masukan baku pengenal
PP-OCR, bukan dari sapuan. Sesuaikan kalau versi pengenalnya berganti.
"""
import os
import time

# Kotak dari width_ths=0.0 adalah PECAHAN token, sering hanya beberapa piksel
# tingginya, sedangkan pengenal PP-OCR dilatih pada potongan setinggi satu
# baris. Tanpa pembesaran ini perbandingannya tidak adil bagi PP-OCR.
TINGGI_REC = 48
# Supaya huruf di tepi kotak tidak terpotong oleh pembulatan koordinat.
MARGIN = 2
# Potongan di bawah ukuran ini tidak mungkin memuat huruf.
SISI_MIN = 3
# Utas onnxruntime. Terukur pada struk 181 kotak: bawaan 3,05 detik, empat
# utas 2,07 detik, delapan utas 2,03 detik. Jadi empat sudah menangkap hampir
# seluruh manfaatnya tanpa memborong inti peladen dari torch.
UTAS = int(os.environ.get("UTAS_PENGENAL", "4"))

BAWAAN = "ppocr"

_mesin = None
_gagal = 0
_detik = 0.0
_mundur = False


def _setelan():
    return os.environ.get("PENGENAL", BAWAAN).strip().lower()


def diminta():
    """Benar kalau ppocr DISETEL eksplisit, bukan dipakai dari bawaan."""
    return os.environ.get("PENGENAL", "").strip().lower() == "ppocr"


def aktif():
    return _setelan() == "ppocr" and not _mundur


def nama():
    if aktif():
        return "PP-OCRv4 rec (rapidocr_onnxruntime)"
    if _mundur:
        return "EasyOCR CRNN (MUNDUR, rapidocr tidak terpasang)"
    return "EasyOCR CRNN"


def _bersihkan(teks):
    """Buang huruf di luar ASCII dari keluaran pengenal. Lihat docstring modul."""
    return "".join(c for c in teks if ord(c) < 128).strip()


def muat():
    """Siapkan pengenal. Dipanggil sekali saat peladen hidup, bukan per permintaan.

    Sengaja gagal keras: kalau PENGENAL=ppocr tetapi pustakanya tidak ada,
    lebih baik peladen tidak hidup daripada hidup dengan pengenal yang bukan
    seperti yang tertulis di setelan.
    """
    global _mesin, _mundur
    if not aktif() or _mesin is not None:
        return _mesin
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:
        if diminta():
            raise RuntimeError(
                "PENGENAL=ppocr tetapi rapidocr_onnxruntime tidak terpasang. "
                "Pasang 'rapidocr_onnxruntime' atau setel PENGENAL=easyocr."
            ) from e
        _mundur = True
        print("PERINGATAN: rapidocr_onnxruntime tidak terpasang, mundur ke "
              "pengenal EasyOCR. Endpoint sehat menyebutkan kemunduran ini.",
              flush=True)
        return None
    try:
        _mesin = RapidOCR(intra_op_num_threads=UTAS)
    except TypeError:
        # Versi rapidocr yang tidak menerima setelan utas. Tetap jalan, hanya
        # lebih lambat.
        _mesin = RapidOCR()
    return _mesin


def kotak_craft(pembaca, jalur, width_ths=0.0):
    """Kotak dari deteksi CRAFT saja, tanpa menjalankan pengenal CRNN EasyOCR.

    Dipakai hanya saat PENGENAL=ppocr, sebab teks CRNN akan ditimpa semuanya
    dan menjalankannya berarti pekerjaan yang terbuang.

    Boleh dilakukan karena terukur: pada 11 struk uji3 kotaknya sama dengan
    readtext() kecuali 3 kotak dari 1326 yang bergeser 1 piksel, dan
    readtext() sendiri pun tidak mengulang angka itu persis antarjalan. Jadi
    bedanya derau pembulatan, bukan jalur deteksi yang berbeda.

    Hematnya 1,4 sampai 1,8 detik per struk.

    Harganya satu: tidak ada lagi teks EasyOCR sebagai jaring pengaman kalau
    satu potongan gagal dibaca PP-OCR. Potongan itu jadi kosong, dan app.py
    menggantinya dengan titik. Pada 1326 kotak uji3, nol potongan gagal.
    """
    mendatar, bebas = pembaca.detect(jalur, width_ths=width_ths)
    kotak = []
    for xa, xb, ya, yb in (mendatar[0] if mendatar else []):
        kotak.append(dict(bbox=[int(xa), int(ya), int(xb), int(yb)],
                          teks="", teks_asli="", skor=0.0))
    for poly in (bebas[0] if bebas else []):
        xs = [float(q[0]) for q in poly]
        ys = [float(q[1]) for q in poly]
        kotak.append(dict(bbox=[int(min(xs)), int(min(ys)),
                                int(max(xs)), int(max(ys))],
                          teks="", teks_asli="", skor=0.0))
    kotak.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return kotak


def _baca_potongan(potongan):
    """Teks dan keyakinan satu potongan, atau (None, None) kalau gagal."""
    import numpy as np

    hasil, _ = _mesin(np.array(potongan), use_det=False, use_cls=False,
                      use_rec=True)
    if not hasil:
        return None, None
    # Tanpa deteksi, bentuknya [[teks, yakin], ...]
    teks = _bersihkan(" ".join(str(r[0]) for r in hasil))
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
    global _gagal, _detik
    if not aktif() or not kotak:
        return kotak
    from PIL import Image

    if muat() is None:
        # Mundur ke EasyOCR terjadi di sini kalau pustakanya ternyata tidak
        # ada. Teks yang sudah ada dibiarkan apa adanya.
        return kotak
    _t0 = time.perf_counter()
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
    _detik += time.perf_counter() - _t0
    return kotak


def gagal():
    """Berapa potongan yang gagal dibaca sejak peladen hidup."""
    return _gagal


def detik():
    """Total detik yang dipakai membaca ulang sejak peladen hidup."""
    return round(_detik, 2)
