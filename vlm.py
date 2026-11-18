"""Pemanggil VLM untuk menambal medan yang gerbang_vlm.py meloloskan.

Bawaannya MATI. Tanpa `VLM=hidup` dan `KUNCI_VLM`, modul ini tidak pernah
menyentuh jaringan, sehingga perilaku peladen tidak berubah sedikit pun
sampai kunci API benar-benar ada.

Jalurnya satu endpoint bergaya OpenAI chat completions, sebab bentuk itu
diterima OpenRouter, Alibaba Model Studio mode kompatibel, Together,
Fireworks, dan juga Gemini lewat endpoint kompatibelnya. Satu jalur kode
untuk banyak penyedia, dan penyedianya dipilih lewat VLM_URL dan VLM_MODEL
tanpa menyentuh kode.

Tanpa dependensi baru: urllib dan base64 dari pustaka standar. requirements
peladen tidak bertambah.

Kebijakan kegagalan sama dengan pengenal_ppocr: satu permintaan yang gagal
TIDAK boleh menjatuhkan satu struk. Galat apa pun dihitung di `gagal()` dan
jawabannya kosong, sehingga medan yang bersangkutan tetap kosong seperti
sebelum VLM ada.

Yang modul ini TIDAK lakukan, dan itu disengaja:

  - tidak memutuskan medan mana yang ditanyakan, itu tugas gerbang_vlm.py
  - tidak menulis ke kontrak, itu tugas gerbang_vlm.tambal() yang menolak
    menimpa medan yang sudah berisi
  - tidak menghitung apa pun dari jawaban, misalnya memeriksa subtotal
    ditambah pajak sama dengan total. Persamaan itu TERUKUR menolak jawaban
    benar pada uji3_01 selisih 8,06 dan uji3_04 selisih 21,78, sebab struk
    Malaysia memuat service charge tanpa kolom sendiri.
"""
import base64
import io
import json
import os
import re
import time
import urllib.request

URL = os.environ.get(
    "VLM_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "qwen/qwen2.5-vl-7b-instruct")
BATAS_DETIK = float(os.environ.get("VLM_BATAS_DETIK", "40"))

# shortcut: 1600 piksel sisi terpanjang dan mutu 85, mencerminkan yang
# diunggah aplikasi. Naikkan kalau terukur VLM salah baca angka kecil.
SISI_MAKS = 1600
MUTU = 85

_gagal = 0
_detik = 0.0

MEDAN_UANG = ("subtotal", "pajak", "total", "tunai", "kembalian")

# Jawaban uang harus BERBENTUK angka sebelum diserahkan ke rantai._uang.
# Sebabnya terukur: _uang("tidak terbaca") mengembalikan 1.0, sebab ia
# memetakan I jadi 1 lalu menyaring sisa hurufnya, sehingga kalimat VLM yang
# jujur berubah menjadi RM 1,00 yang percaya diri. Perilaku _uang itu TIDAK
# diubah, sebab ia ikut mengukur angka pra-registrasi; penjagaan ditaruh di
# sini, di sisi masukan VLM saja.
_POLA_UANG = re.compile(r"^\s*(?:RM|MYR|\$)?\s*-?\d[\d.,\s]*$", re.I)

# Perintah dalam bahasa Inggris, sebab model terbuka seperti Qwen2.5-VL
# terukur lebih patuh pada bahasa Inggris daripada bahasa Indonesia untuk
# keluaran berformat. Tiga larangan yang penting: jangan menghitung, jangan
# menebak, dan jawab null kalau tidak tercetak. Tanpa larangan menghitung,
# VLM akan mengarang total dari penjumlahan barang dan terdengar yakin.
PERINTAH = (
    "You are reading a Malaysian retail receipt photo.\n"
    "Return ONLY a JSON object, no prose, no markdown fence.\n"
    "Keys to return: %s\n"
    "Rules:\n"
    "1. Report only what is PRINTED on the receipt. Do not compute, do not "
    "add up items, do not infer.\n"
    "2. If a field is not printed on this receipt, return null for it. "
    "Returning null is correct and expected.\n"
    "3. Money fields: digits only, dot as decimal separator, no currency "
    "symbol, for example 93.90\n"
    "4. jam: copy the printed time exactly as printed, including AM or PM "
    "and seconds if printed.\n"
    "5. tanggal: copy the printed date exactly as printed.\n"
    "6. toko: the shop name as printed at the top.\n"
)

_ARTI = {
    "toko": "shop name",
    "tanggal": "date as printed",
    "jam": "time as printed",
    "subtotal": "subtotal before tax",
    "pajak": "tax or SST amount, not the percentage",
    "total": "final amount payable",
    "tunai": "cash tendered",
    "kembalian": "change given",
}


def aktif():
    """Hidup hanya kalau DIMINTA eksplisit DAN kuncinya ada.

    Dua syarat, bukan satu: kunci yang tertinggal di lingkungan tidak boleh
    membuat peladen mulai memanggil jaringan tanpa diminta.
    """
    return (os.environ.get("VLM", "mati").strip().lower() == "hidup"
            and bool(os.environ.get("KUNCI_VLM", "").strip()))


def gagal():
    return _gagal


def detik():
    return round(_detik, 3)


def _citra_base64(sumber):
    """JPEG terkecil yang masih wajar, sebagai base64. Jalur atau gambar PIL."""
    from PIL import Image
    if isinstance(sumber, str):
        with Image.open(sumber) as im:
            g = im.convert("RGB")
    else:
        g = sumber.convert("RGB")
    panjang = max(g.size)
    if panjang > SISI_MAKS:
        s = SISI_MAKS / panjang
        g = g.resize((max(1, round(g.size[0] * s)), max(1, round(g.size[1] * s))),
                     Image.LANCZOS)
    buf = io.BytesIO()
    g.save(buf, format="JPEG", quality=MUTU)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _muatan(medan, b64):
    kunci = ", ".join('"%s" (%s)' % (m, _ARTI.get(m, m)) for m in medan)
    return {
        "model": MODEL,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PERINTAH % kunci},
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + b64}},
        ]}],
    }


def _kirim_http(muatan):
    """POST ke endpoint bergaya OpenAI. Kuncinya tidak pernah dicetak."""
    permintaan = urllib.request.Request(
        URL, data=json.dumps(muatan).encode("utf-8"), method="POST")
    permintaan.add_header("Content-Type", "application/json")
    permintaan.add_header(
        "Authorization", "Bearer " + os.environ.get("KUNCI_VLM", "").strip())
    with urllib.request.urlopen(permintaan, timeout=BATAS_DETIK) as r:
        return json.loads(r.read().decode("utf-8"))


def _teks_jawaban(balas):
    """Isi pesan pertama, atau None. Bentuknya sama di semua penyedia."""
    try:
        return balas["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _json_pertama(teks):
    """Objek JSON pertama di dalam teks, atau {}.

    VLM sering membungkus jawabannya dengan ```json walaupun dilarang, jadi
    menguraikan seluruh balasan sebagai JSON akan gagal pada jawaban yang
    sebenarnya benar.
    """
    if not teks:
        return {}
    m = re.search(r"\{.*\}", teks, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}


def tanya(sumber, medan, uang=None, kirim=None):
    """{medan: nilai} dari VLM, hanya untuk `medan` yang diminta.

    `uang` adalah rantai._uang, disuntikkan supaya modul ini tidak perlu
    mengimpor rantai. Medan uang yang tidak bisa diubah jadi angka DIBUANG,
    bukan diteruskan sebagai teks, sebab kontrak aplikasi menuntut angka.

    `kirim` menggantikan pengirim HTTP, dipakai uji. Galat apa pun jatuh ke
    jawaban kosong, sehingga medannya tetap kosong seperti sebelum ada VLM.
    """
    global _gagal, _detik
    if not medan or not aktif():
        return {}
    nama = [m["medan"] if isinstance(m, dict) else m for m in medan]
    t0 = time.perf_counter()
    try:
        muatan = _muatan(nama, _citra_base64(sumber))
        balas = (kirim or _kirim_http)(muatan)
        mentah = _json_pertama(_teks_jawaban(balas))
    except Exception:
        _gagal += 1
        _detik += time.perf_counter() - t0
        return {}
    _detik += time.perf_counter() - t0

    keluar = {}
    for m in nama:
        if m not in mentah:
            continue
        v = mentah[m]
        if v is None:
            continue
        if m in MEDAN_UANG:
            if not _POLA_UANG.match(str(v)):
                continue
            v = uang(v) if uang else v
            if not isinstance(v, (int, float)):
                continue
        else:
            v = str(v).strip()
            if not v or v.lower() in ("null", "none", "n/a", "-"):
                continue
        keluar[m] = v
    return keluar
