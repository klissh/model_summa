"""Server model Summa.

Rantai bawaan (Model v2): foto -> DBNet + PP-OCRv4 -> pecah_baris -> LayoutLMv3
v2 -> penengah hibrida -> JSON. EasyOCR/CRAFT TIDAK dimuat.
Rantai lama (Model v1, PEMBACA=craft + MODEL_ID=...-ocr): foto -> EasyOCR ->
LayoutLMv3 -> penengah hibrida -> JSON, disalin dari Notebook E sel 8.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import io
import json
import os
import resource  # hanya ada di Linux, dan kontainernya Linux
import tempfile
import time
import uuid

import psutil
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, ImageOps

# Batas keras 25 megapiksel untuk mencegah decompression bomb yang memicu OOM
Image.MAX_IMAGE_PIXELS = 25_000_000

import gerbang_vlm
import pembanding_aturan as PA
import pembaca_dbnet
import pengenal_ppocr
import pisah_tahap
import rantai
import vlm
from hibrida_penengah import SET_ATURAN_MENANG, penengah

MODEL_ID = os.environ.get("MODEL_ID", "Klissh/layoutlmv3-struk-summa-v2-dbnet")
# "dbnet" (bawaan, cocok dengan Model v2) atau "craft" (EasyOCR, cocok dengan Model v1).
# Pembaca harus cocok dengan model: Model v1 dilatih pada kotak CRAFT, Model v2 pada kotak DBNet.
PEMBACA = os.environ.get("PEMBACA", "dbnet").strip().lower()
assert PEMBACA in ("dbnet", "craft"), "PEMBACA harus dbnet atau craft"
KUNCI = os.environ.get("KUNCI", "")
MAKS_BYTE = 10 * 1024 * 1024
SATU_TOTAL = False  # False = varian P2 yang dipra-registrasikan, True = P2U

# Notebook E memakai 2 utas. Kalau berbeda, angka latensi tidak sebanding.
torch.set_num_threads(int(os.environ.get("UTAS_TORCH", "2")))

MAKS_TOKEN = 512
LANGKAH = 128

_mulai = time.time()

from transformers import (  # noqa: E402
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

if PEMBACA == "craft":
    import easyocr  # noqa: E402  (jalur Model v1 saja; diimpor setelah pencatat waktu supaya ikut terukur)

    # quantize=False WAJIB. Nilai bawaannya True, dan kuantisasi int8 hanya
    # diterapkan di CPU. Data gold dibuat Notebook A di GPU, jadi tanpa kuantisasi.
    # Dengan kuantisasi, pengenal huruf membaca "RIDGI (LAY" untuk kata yang di
    # gold terbaca "KIDDY CLAY", dan model menerima teks yang asing baginya.
    pembaca = easyocr.Reader(rantai.BAHASA_OCR, gpu=False, quantize=False)
    # Sengaja di sini, saat peladen hidup, bukan saat permintaan pertama datang.
    # Salah setel PENGENAL harus menjatuhkan peladen di deploy, bukan diam-diam
    # berjalan dengan pengenal yang bukan seperti tertulis di setelan.
    pengenal_ppocr.muat()
else:
    # Model v2: tanpa EasyOCR/CRAFT sama sekali, jadi RAM startup jauh lebih kecil.
    easyocr = None
    pembaca = None
    pembaca_dbnet.muat()

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

# Membatasi inferensi serentak maksimal 1 agar tidak memicu OOM di Railway 8 GB
_SEMAPHORE_KONKURENSI = asyncio.Semaphore(1)
_ANTREAN_MAKS = 3
_BATAS_TUNGGU_SEMAPHORE = 45.0
_antrean_aktif = 0


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
    if PEMBACA == "dbnet":
        # EXIF diluruskan di dalam siapkan_citra; sisi <= 1600 dan JPEG 85, seperti yang diukur
        img = pembaca_dbnet.siapkan_citra(jalur)
    else:
        with Image.open(jalur) as _im_awal:
            _orientasi = _im_awal.getexif().get(0x0112, 1)
            if _orientasi and _orientasi != 1:
                _im_tegak = ImageOps.exif_transpose(_im_awal).convert("RGB")
                _im_tegak.save(jalur, quality=95)
                img = _im_tegak
            else:
                img = _im_awal.convert("RGB")
    L, T = img.size
    w["1_baca_gambar"] = time.perf_counter() - t

    t = time.perf_counter()
    if PEMBACA == "dbnet":
        kotak = pembaca_dbnet.kotak_dbnet(img, rantai.normalisasi_angka)
    else:
        kotak = ocr_mentah(jalur)
    w["2_ocr"] = time.perf_counter() - t
    if not kotak:
        return None, w

    hasil, kot, n_jendela = _labeli(img, L, T, kotak, w)

    # Jalur penyelamat, hanya saat TIDAK SATU PUN item dapat harga. Hasilnya
    # dipakai cuma kalau item berharganya lebih banyak, jadi struk yang sudah
    # jalan tidak bisa menjadi lebih buruk karena jalur ini. Lihat
    # pisah_tahap.py untuk angka dan alasannya. Bergantung pada CRAFT, jadi
    # hanya ada di jalur Model v1; di jalur DBNet tidak dijalankan (juga tidak
    # dijalankan saat Model v2 diukur).
    dipakai = False
    if pembaca is not None and pisah_tahap.perlu(hasil):
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
        "pembaca": PEMBACA,
        "easyocr": easyocr.__version__ if easyocr else None,
        "pengenal": pengenal_ppocr.nama() if PEMBACA == "craft" else "PP-OCRv4 rec (DBNet, rapidocr_onnxruntime)",
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
    # Fail-closed: KUNCI wajib ada dan cocok
    if not KUNCI:
        raise HTTPException(status_code=500, detail="server belum disetel KUNCI")
    if x_kunci != KUNCI:
        raise HTTPException(status_code=401, detail="kunci salah")

    isi = await berkas.read()
    if not isi:
        raise HTTPException(status_code=400, detail="berkas kosong")
    if len(isi) > MAKS_BYTE:
        raise HTTPException(status_code=413, detail=f"maksimal {MAKS_BYTE // 1024 // 1024} MB")

    try:
        with Image.open(io.BytesIO(isi)) as im:
            if im.width * im.height > Image.MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="resolusi gambar terlalu besar (maks 25 MP)")
            akhiran = rantai.ekstensi_gambar(im.format)
    except HTTPException:
        raise
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail="resolusi gambar terlalu besar (maks 25 MP)")
    except Exception:
        raise HTTPException(status_code=400, detail="bukan berkas gambar")
    if akhiran is None:
        raise HTTPException(status_code=415, detail="format gambar tidak didukung")

    global _antrean_aktif
    if _antrean_aktif >= _ANTREAN_MAKS:
        raise HTTPException(
            status_code=503,
            detail="server sedang sibuk memproses antrean struk, coba beberapa detik lagi",
        )
    _antrean_aktif += 1
    try:
        try:
            await asyncio.wait_for(_SEMAPHORE_KONKURENSI.acquire(), timeout=_BATAS_TUNGGU_SEMAPHORE)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="server sedang sibuk, waktu tunggu antrean habis")
        try:
            def _eksekusi_inferensi():
                jalur = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=akhiran, delete=False) as f:
                        f.write(isi)
                        jalur = f.name
                    keluar, waktu = jalankan(jalur)
                    if keluar is None:
                        return None, None, None, waktu
                    hasil, kot, tinggi = keluar
                    return hasil, kot, tinggi, waktu
                finally:
                    if jalur and os.path.exists(jalur):
                        os.unlink(jalur)
                    gc.collect()

            # Inferensi berat jalan di utas pekerja; event loop tetap bebas melayani antrean.
            hasil, kot, tinggi, waktu = await asyncio.get_running_loop().run_in_executor(
                None, _eksekusi_inferensi)
        finally:
            _SEMAPHORE_KONKURENSI.release()
    finally:
        _antrean_aktif -= 1

    if hasil is None:
        raise HTTPException(status_code=422, detail="OCR tidak menemukan teks apa pun")

    id_struk = uuid.uuid4().hex[:12]
    aplikasi = rantai.ke_kontrak_aplikasi(hasil, kot, id_struk, MODEL_ID)
    minta = gerbang_vlm.perlu_tambal(aplikasi, kot, tinggi)
    minta_item = gerbang_vlm.perlu_tambal_item(aplikasi)
    tertulis = []
    item_vlm = []
    if (minta or minta_item) and vlm.aktif():
        def _eksekusi_vlm():
            _tertulis = []
            _item_vlm = []
            if minta and minta_item:
                def _panggil_ringkasan():
                    with Image.open(io.BytesIO(isi)) as _im1:
                        return vlm.tanya(_im1, minta, uang=rantai._uang)

                def _panggil_item():
                    with Image.open(io.BytesIO(isi)) as _im2:
                        return vlm.tanya_item(_im2, uang=rantai._uang)

                with ThreadPoolExecutor(max_workers=2) as _pool:
                    _f_ringkasan = _pool.submit(_panggil_ringkasan)
                    _f_item = _pool.submit(_panggil_item)
                    _jwb_ringkasan = _f_ringkasan.result()
                    _jwb_item = _f_item.result()
                _tertulis = gerbang_vlm.tambal(aplikasi, _jwb_ringkasan, kot, tinggi)
                _item_vlm = gerbang_vlm.tambal_item(aplikasi, _jwb_item)
            else:
                with Image.open(io.BytesIO(isi)) as _im:
                    if minta:
                        jawaban = vlm.tanya(_im, minta, uang=rantai._uang)
                        _tertulis = gerbang_vlm.tambal(aplikasi, jawaban, kot, tinggi)
                    if minta_item:
                        _item_vlm = gerbang_vlm.tambal_item(
                            aplikasi, vlm.tanya_item(_im, uang=rantai._uang))
            return _tertulis, _item_vlm

        _t = time.perf_counter()
        # ThreadPoolExecutor di dalam _eksekusi_vlm menunggu .result() di utas pekerja, bukan di event loop.
        tertulis, item_vlm = await asyncio.get_running_loop().run_in_executor(
            None, _eksekusi_vlm)
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
        # Apakah ada baris barang yang harganya kosong, dan id baris mana saja
        # yang harganya akhirnya diisi model bahasa visual. Daftar kosong
        # berarti tidak ada lubang yang namanya dapat dicocokkan.
        "gerbang_vlm_item": minta_item,
        "vlm_item_tertulis": item_vlm,
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
    print(json.dumps({
        "event": "ekstrak_selesai",
        "id": id_struk,
        "waktu": jawab["waktu"],
        "vlm_tertulis": tertulis,
        "vlm_item_tertulis": item_vlm,
        "rss_gb": round(psutil.Process().memory_info().rss / 1e9, 2),
    }), flush=True)
    return jawab
