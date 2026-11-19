"""Pemanggil VLM untuk menambal medan yang gerbang_vlm.py meloloskan.

Bawaannya MATI. Tanpa `VLM=hidup` dan `KUNCI_VLM`, modul ini tidak pernah
menyentuh jaringan, sehingga perilaku peladen tidak berubah sedikit pun
sampai kunci API benar-benar ada.

Jalurnya satu endpoint bergaya OpenAI chat completions, sebab bentuk itu
diterima OpenRouter, Alibaba Model Studio mode kompatibel, Together,
Fireworks, dan juga Gemini lewat endpoint kompatibelnya. Satu jalur kode
untuk banyak penyedia, dan penyedianya dipilih lewat VLM_URL dan VLM_MODEL
tanpa menyentuh kode.

Untuk HF Router, penyedia TIDAK perlu disebut: router memilih yang hidup
sendiri kalau VLM_MODEL hanya berisi nama model. Menyebut satu penyedia
dilakukan dengan menambahkan titik dua, misalnya
Qwen/Qwen2.5-VL-7B-Instruct:featherless-ai

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

def _env(nama, bawaan):
    """Nilai lingkungan, dengan RAGAM KOSONG diperlakukan sebagai tidak disetel.

    Bedanya menentukan. os.environ.get(nama, bawaan) mengembalikan teks kosong
    kalau ragamnya ada tetapi kosong, dan Railway menyimpan ragam kosong
    sebagai ragam yang ada. Akibatnya URL jadi "" dan urllib melemparkan
    ValueError dalam hitungan milidetik, 11 kali berturut-turut, tanpa satu
    byte pun menyentuh jaringan.
    """
    return (os.environ.get(nama) or "").strip() or bawaan


URL = _env("VLM_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL = _env("VLM_MODEL", "qwen/qwen2.5-vl-7b-instruct")
BATAS_DETIK = float(_env("VLM_BATAS_DETIK", "40"))

# shortcut: 1600 piksel sisi terpanjang dan mutu 85, mencerminkan yang
# diunggah aplikasi. Naikkan kalau terukur VLM salah baca angka kecil.
SISI_MAKS = 1600
MUTU = 85

# Batas keluaran. Jawaban yang diminta cuma objek JSON delapan medan, jadi 300
# token sudah lapang. Batas ini ada untuk ONGKOS: tanpa itu, VLM yang mulai
# mengarang penjelasan panjang dibayar per token sampai penyedianya berhenti
# sendiri.
MAKS_TOKEN = 300

# Batas keras panggilan VLM per hari UTC. Ini pagar ONGKOS, bukan pagar lalu
# lintas: kunci peladen tertanam di aplikasi ponsel, dan aplikasi ponsel bisa
# dibongkar, jadi kunci itu harus dianggap bisa bocor. Saat batas tercapai VLM
# diam dan medan tetap kosong, persis seperti sebelum VLM ada, bukan galat
# yang menjatuhkan pemindaian.
BATAS_HARI = int(_env("VLM_BATAS_HARI", "200"))

_gagal = 0
_detik = 0.0
# Galat terakhir, dilaporkan endpoint sehat. Tanpa ini satu-satunya gejala
# kegagalan adalah pencacah yang naik, dan sebabnya hanya bisa ditebak.
_galat = ""
_hari = ""
_pakai = 0

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

# JANGAN tambahkan aturan "medan uang hanya sah kalau barisnya berlabel medan
# itu". Aturan itu pernah dipasang pada 29239be dan DIUKUR MERUGIKAN: pada 11
# struk uji3 ia tidak membetulkan satu pun dari dua kesalahan pajak yang jadi
# alasannya (uji3_05 tetap 9,20 dan uji3_11 tetap 0,02, nilainya identik
# antarlari), sementara ia membuat VLM menolak dua jawaban yang BENAR, yaitu
# tunai dan kembalian pada uji3_07. Hasilnya 49 dari 66 turun jadi 47.
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


def galat():
    return _galat


def pakai_hari_ini():
    return _pakai if _hari == time.strftime("%Y-%m-%d", time.gmtime()) else 0


def _ambil_jatah():
    """True kalau masih ada jatah hari ini, sekaligus memakainya satu."""
    global _hari, _pakai
    kini = time.strftime("%Y-%m-%d", time.gmtime())
    if kini != _hari:
        _hari, _pakai = kini, 0
    if _pakai >= BATAS_HARI:
        return False
    _pakai += 1
    return True


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
        "max_tokens": MAKS_TOKEN,
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
    global _gagal, _detik, _galat
    if not medan or not aktif():
        return {}
    if not _ambil_jatah():
        _galat = "jatah harian %d panggilan habis" % BATAS_HARI
        return {}
    nama = [m["medan"] if isinstance(m, dict) else m for m in medan]
    t0 = time.perf_counter()
    try:
        muatan = _muatan(nama, _citra_base64(sumber))
        balas = (kirim or _kirim_http)(muatan)
        mentah = _json_pertama(_teks_jawaban(balas))
    except Exception as e:
        _gagal += 1
        _galat = "%s: %s" % (type(e).__name__, str(e)[:160])
        k = (os.environ.get("KUNCI_VLM") or "").strip()
        if k and k in _galat:
            _galat = _galat.replace(k, "<kunci>")
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
