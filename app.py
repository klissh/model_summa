"""Server model Summa.

Rantai: foto -> EasyOCR -> LayoutLMv3 -> penengah hibrida -> JSON.
Disalin dari Notebook E sel 8, dengan satu tambahan: penengah hibrida,
yang di notebook itu tidak dipakai.
"""

import io
import os
import resource  # hanya ada di Linux, dan kontainernya Linux
import tempfile
import time
import uuid

import psutil
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image

import gerbang_vlm
import pembanding_aturan as PA
import pengenal_ppocr
import pisah_tahap
import rantai
import vlm
from hibrida_penengah import SET_ATURAN_MENANG, penengah

MODEL_ID = os.environ.get("MODEL_ID", "Klissh/layoutlmv3-struk-summa-ocr")
KUNCI = os.environ.get("KUNCI", "")
MAKS_BYTE = 10 * 1024 * 1024
SATU_TOTAL = False  # False = varian P2 yang dipra-registrasikan, True = P2U

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

# quantize=False WAJIB. Nilai bawaannya True, dan kuantisasi int8 hanya
# diterapkan di CPU. Data gold dibuat Notebook A di GPU, jadi tanpa kuantisasi.
# Dengan kuantisasi, pengenal huruf membaca "RIDGI (LAY" untuk kata yang di
# gold terbaca "KIDDY CLAY", dan model menerima teks yang asing baginya.
pembaca = easyocr.Reader(rantai.BAHASA_OCR, gpu=False, quantize=False)

# Sengaja di sini, saat peladen hidup, bukan saat permintaan pertama datang.
# Salah setel PENGENAL harus menjatuhkan peladen di deploy, bukan diam-diam
# berjalan dengan pengenal yang bukan seperti tertulis di setelan.
pengenal_ppocr.muat()

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
    """Notebook E sel 6. Urutan dan isinya sama dengan Notebook A sel 7.

    Saat PENGENAL=ppocr, pengenal CRNN EasyOCR dilewatkan sepenuhnya: teksnya
    akan ditimpa seluruhnya, jadi menjalankannya hanya pekerjaan terbuang
    seharga 1,4 sampai 1,8 detik per struk. Kotaknya terukur sama, lihat
    pengenal_ppocr.kotak_craft.
    """
    # muat() dipanggil lebih dulu supaya kemunduran ke EasyOCR sudah diketahui
    # sebelum memutuskan melewatkan pengenal CRNN.
    pengenal_ppocr.muat()
    if pengenal_ppocr.aktif():
        kotak = pengenal_ppocr.kotak_craft(
            pembaca, jalur, rantai.PARAM_READTEXT["width_ths"])
    else:
        kotak = []
        for item in pembaca.readtext(jalur, **rantai.PARAM_READTEXT):
            poly, teks, skor = item[0], item[1], item[2]
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            kotak.append(dict(
                bbox=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                teks=rantai.normalisasi_angka(teks), teks_asli=teks,
                skor=float(skor)))
        kotak.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    # Bbox TIDAK disentuh, jadi masukan ruang untuk LayoutLMv3 tetap sama.
    # Tanpa PENGENAL=ppocr baris ini tidak mengubah apa pun.
    return pengenal_ppocr.baca_ulang(jalur, kotak, rantai.normalisasi_angka)


def _labeli(img, L, T, kotak, w):
    """Dari kotak OCR jadi JSON: LayoutLMv3, penengah hibrida, lalu perakitan.

    Dipisah dari jalankan() supaya jalur penyelamat bisa memakainya kembali
    atas kumpulan kotak yang lain, tanpa menyalin isinya.

    Kembalikan (hasil, kot, n_jendela).
    """
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
    # satu_total=False, yaitu varian P2. P2U memaksa tepat satu total per struk,
    # dan pengukuran 16 struk uji menunjukkan itu tidak menaikkan ketepatan
    # (29,7 lawan 30,7 persen) sambil menghasilkan total yang salah dengan
    # percaya diri, misalnya 436,00 untuk struk yang totalnya 6,35.
    akhir = penengah(struk, label_model, skor_model, satu_total=SATU_TOTAL)
    for b, L_aturan, L_akhir in zip(kotak, aturan, akhir):
        b["_label"] = L_akhir
        b["_dari_aturan"] = L_aturan in SET_ATURAN_MENANG and L_aturan == L_akhir
    w["6_penengah"] = time.perf_counter() - t

    # ke_json_lengkap, bukan ke_json. Yang polos tetap ada karena dipakai
    # menyusun jawaban gold di ukur_server.py, dan kalau gold ikut ditambal
    # pengukurannya membandingkan sistem dengan dirinya sendiri.
    # PA.prediksi memang jadi dipanggil dua kali, di sini dan di dalam
    # penambalan. Biayanya puluhan milidetik atas rantai yang 7 detik.
    hasil = rantai.ke_json_lengkap(kot, L, T)
    return hasil, kot, n_jendela


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

    hasil, kot, n_jendela = _labeli(img, L, T, kotak, w)

    # Jalur penyelamat, hanya saat TIDAK SATU PUN item dapat harga. Hasilnya
    # dipakai cuma kalau item berharganya lebih banyak, jadi struk yang sudah
    # jalan tidak bisa menjadi lebih buruk karena jalur ini. Lihat
    # pisah_tahap.py untuk angka dan alasannya.
    dipakai = False
    if pisah_tahap.perlu(hasil):
        t = time.perf_counter()
        img2, kotak2 = pisah_tahap.ocr(jalur, pembaca)
        if img2 is not None and kotak2:
            # Kotak penyelamat berada di ruang koordinat gambar yang SUDAH
            # diputar, jadi yang dibaca ulang gambar itu, bukan berkas asli.
            kotak2 = pengenal_ppocr.baca_ulang(img2, kotak2,
                                               rantai.normalisasi_angka)
        w["7_penyelamat_ocr"] = time.perf_counter() - t
        if kotak2:
            w2 = {}
            hasil2, kot2, nj2 = _labeli(img2, img2.size[0], img2.size[1], kotak2, w2)
            w["8_penyelamat_rantai"] = sum(w2.values())
            if pisah_tahap.n_harga_item(hasil2) > pisah_tahap.n_harga_item(hasil):
                hasil, kot, n_jendela, dipakai = hasil2, kot2, nj2, True
                # kotaknya kini di ruang koordinat gambar yang sudah diputar,
                # jadi tinggi acuan gerbang VLM ikut berganti
                T = img2.size[1]

    w["total"] = sum(v for k, v in w.items() if k != "total")
    w["n_kotak"] = len(kot)
    w["n_jendela"] = n_jendela
    w["penyelamat"] = dipakai
    return (hasil, kot, T), w


@app.get("/")
def sehat():
    """Dipanggil untuk tahu apakah layanan hidup, dan berapa memorinya."""
    p = psutil.Process()
    return {
        "siap": True,
        "model": MODEL_ID,
        "detik_muat": DETIK_MUAT,
        "easyocr": easyocr.__version__,
        "pengenal": pengenal_ppocr.nama(),
        "pengenal_gagal": pengenal_ppocr.gagal(),
        "pengenal_detik": pengenal_ppocr.detik(),
        "kuantisasi": False,
        "satu_total": SATU_TOTAL,
        "vlm": vlm.MODEL if vlm.aktif() else False,
        "vlm_gagal": vlm.gagal(),
        "vlm_detik": vlm.detik(),
        "vlm_galat": vlm.galat(),
        "vlm_pakai_hari_ini": vlm.pakai_hari_ini(),
        "vlm_batas_hari": vlm.BATAS_HARI,
        "rss_gb": round(p.memory_info().rss / 1e9, 2),
        "puncak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
        "berkunci": bool(KUNCI),
        # Tanpa ini tidak ada cara tahu kode mana yang sedang hidup, dan
        # pengukuran bisa salah dikira berasal dari versi yang baru.
        "revisi": os.environ.get("RAILWAY_GIT_COMMIT_SHA", "")[:7] or "tidak diketahui",
    }


@app.post("/ekstrak")
async def ekstrak(berkas: UploadFile = File(...), rinci: int = 0,
                  x_kunci: str = Header(default="")):
    if KUNCI and x_kunci != KUNCI:
        raise HTTPException(status_code=401, detail="kunci salah")

    isi = await berkas.read()
    if not isi:
        raise HTTPException(status_code=400, detail="berkas kosong")
    if len(isi) > MAKS_BYTE:
        raise HTTPException(status_code=413, detail=f"maksimal {MAKS_BYTE // 1024 // 1024} MB")

    try:
        with Image.open(io.BytesIO(isi)) as im:
            akhiran = rantai.ekstensi_gambar(im.format)
    except Exception:
        raise HTTPException(status_code=400, detail="bukan berkas gambar")
    if akhiran is None:
        raise HTTPException(status_code=415, detail="format gambar tidak didukung")

    jalur = None
    try:
        with tempfile.NamedTemporaryFile(suffix=akhiran, delete=False) as f:
            f.write(isi)
            jalur = f.name
        keluar, waktu = jalankan(jalur)
        if keluar is None:
            raise HTTPException(status_code=422, detail="OCR tidak menemukan teks apa pun")
        hasil, kot, tinggi = keluar
    finally:
        if jalur and os.path.exists(jalur):
            os.unlink(jalur)

    id_struk = uuid.uuid4().hex[:12]
    aplikasi = rantai.ke_kontrak_aplikasi(hasil, kot, id_struk, MODEL_ID)
    minta = gerbang_vlm.perlu_tambal(aplikasi, kot, tinggi)
    tertulis = []
    if minta and vlm.aktif():
        _t = time.perf_counter()
        # Citra ASLI yang dikirim, bukan gambar jalur penyelamat yang sudah
        # diputar: VLM menangani kemiringan lebih baik daripada rantai ini.
        # Berkas sementaranya sudah dihapus, jadi dipakai `isi` yang masih ada.
        with Image.open(io.BytesIO(isi)) as _im:
            jawaban = vlm.tanya(_im, minta, uang=rantai._uang)
        tertulis = gerbang_vlm.tambal(aplikasi, jawaban, kot, tinggi)
        # Masuk ke `total` juga, sebab `total` dipakai mengukur latensi
        # peladen dan panggilan VLM itu waktu peladen yang sungguhan.
        waktu["9_vlm"] = time.perf_counter() - _t
        waktu["total"] = waktu.get("total", 0.0) + waktu["9_vlm"]
    jawab = {
        "aplikasi": aplikasi,
        # Medan yang BOLEH ditanyakan ke VLM, menurut gerbang di
        # gerbang_vlm.py. Peladen ini sendiri tidak memanggil VLM apa pun:
        # daftarnya dilaporkan supaya gerbangnya bisa diukur pada data nyata
        # sebelum ada VLM, dan supaya aplikasi tahu medan mana yang kosong
        # karena OCR, bukan karena struknya memang tidak memuatnya.
        "gerbang_vlm": minta,
        "vlm_tertulis": tertulis,
        "mentah": hasil,
        "waktu": {k: round(v, 3) if isinstance(v, float) else v for k, v in waktu.items()},
    }
    # ?rinci=1 memunculkan kotak OCR beserta labelnya. Dipakai untuk menelusuri
    # kesalahan agregasi, yang tanpa ini hanya bisa ditebak dari hasil akhir.
    # Aplikasi tidak memintanya, jadi muatan jawaban sehari-hari tidak berubah.
    if rinci:
        # yakin TIDAK dibulatkan. ke_json memilih entitas lewat max(yakin),
        # jadi pembulatan bisa mengubah pemenang saat dua entitas berdekatan,
        # dan hasil putar ulang di luar server tidak lagi sama dengan server.
        jawab["kotak"] = [{"teks": b["teks"], "bbox": b["bbox"],
                           "label": b["_label"], "yakin": b["_yakin"]}
                          for b in kot]
    return jawab
