"""Uji mandiri untuk rantai.py. Jalankan: python uji.py

Sengaja tidak memuat model, supaya bisa dijalankan di laptop tanpa torch.
Yang diuji adalah bagian yang paling mudah rusak diam-diam: normalisasi teks,
urutan baca, dan bentuk JSON yang dikirim ke aplikasi.
"""

from rantai import (
    _harga_baris,
    _mirip_uang,
    _uang,
    ekstensi_gambar,
    ke_json,
    ke_kontrak_aplikasi,
    kelompok_baris,
    norm_bbox,
    normalisasi_angka,
    urut_baca,
)


def kotak(teks, x0, y0, x1, y1, label="O", yakin=0.9, dari_aturan=False):
    return {"teks": teks, "bbox": [x0, y0, x1, y1],
            "_label": label, "_yakin": yakin, "_dari_aturan": dari_aturan}


# --- 1. normalisasi, sama dengan assert di Notebook A sel 5 ---
for a, b in [("3,50", "3.50"), ("1,200", "1,200"), ("14:03", "14:03"), ("Nasi", "Nasi")]:
    assert normalisasi_angka(a) == b, (a, normalisasi_angka(a), b)

# --- 1b. _uang: titik di ujung bukan pemisah desimal ---
# Tiga baris pertama dulu menghasilkan nilai salah yang percaya diri, dan
# 2725,0 memang tercatat sebagai kegagalan di G7. Sisanya ada di sini supaya
# tambalan rstrip(".") terbukti tidak menyentuh yang sudah benar.
for _t, _v in [("04.36.;", 4.36), ("27.25.", 27.25), ("12.50.", 12.5),
               ("6.35", 6.35), ("0.50", 0.50), ("170", 170.0),
               ("1.234.56", 1234.56), ("87.55", 87.55), ("100.00", 100.0),
               (".75", 0.75), ("12-", -12.0), ("RM 5.00", 5.0),
               (".", None), ("", None), (None, None)]:
    assert _uang(_t) == _v, (_t, _uang(_t), _v)

# --- 1c. garis bawah itu titik desimal yang salah terbaca ---
# Struk 99 Speed Mart 20 Okt 2026: EasyOCR memberi "10.99" sebagai dua kotak,
# "10 _" lalu "99". Sebelum ini _mirip_uang menolak potongan kirinya, jadi
# _harga_baris cuma melihat "99" dan harga Colgate tercatat 99,0.
for _t, _v in [("10 _", 10.0), ("10 _ 99", 10.99), ("3 _ 99", 3.99), ("_", None)]:
    assert _uang(_t) == _v, (_t, _uang(_t), _v)
assert _mirip_uang("10 _") and _mirip_uang("99") and not _mirip_uang("RM")

_bar = [kotak("823", 20, 800, 90, 835, "menu.num"),
        kotak("COLGATE", 100, 800, 300, 835, "menu.nm"),
        kotak("RM", 600, 800, 650, 835),
        kotak("10 _", 740, 795, 800, 835),
        kotak("99", 805, 795, 845, 835)]
assert _harga_baris(_bar) == 10.99, _harga_baris(_bar)

# Cacat deteksi di tepi kertas: kotak setinggi 563 piksel di baris setinggi 78
# terbaca "0" dan letaknya paling kanan, jadi ia memenangkan aturan gugus
# terkanan dan harga Colgate jadi 0,0. Angka bbox diambil apa adanya dari
# struk 99 Speed Mart 20 Okt 2026.
_bar_cacat = _bar + [kotak("0", 1944, 1344, 2088, 1907)]
assert _harga_baris(_bar_cacat) == 10.99, _harga_baris(_bar_cacat)

# --- 1d. harga item lewat _uang_potongan, bukan _uang ---
# Struk Lotus's: harga dua desimal terpecah dan pemisahnya terbaca koma, jadi
# "2," + "40" dulu jadi 240,0. Diukur atas 43 struk: 38 harga item berubah,
# dan jumlah item yang cocok subtotal gold naik dari 5 jadi 8, nol memburuk.
_bar_koma = [kotak("0955", 20, 800, 90, 835, "menu.num"),
             kotak("FRESHZY", 100, 800, 300, 835, "menu.nm"),
             kotak("2,", 740, 795, 790, 835),
             kotak("40", 795, 795, 845, 835)]
assert _harga_baris(_bar_koma) == 2.40, _harga_baris(_bar_koma)

# Titik desimal hilang sama sekali: dipulihkan dari dua angka ekor.
_bar_hilang = [kotak("0955", 20, 800, 90, 835, "menu.num"),
               kotak("BIOZIP", 100, 800, 300, 835, "menu.nm"),
               kotak("8", 740, 795, 780, 835),
               kotak("70", 790, 795, 845, 835)]
assert _harga_baris(_bar_hilang) == 8.70, _harga_baris(_bar_hilang)

# Tanpa menu.nm jalurnya lewat _angka_baris, dan itu pun harus ikut.
_bar_tanpa_nama = [kotak("2,", 740, 795, 790, 835, "menu.price"),
                   kotak("40", 795, 795, 845, 835, "menu.price")]
assert _harga_baris(_bar_tanpa_nama) == 2.40, _harga_baris(_bar_tanpa_nama)

# --- 2. urutan baca dan normalisasi bbox ---
assert norm_bbox([450, 800, 900, 1600], 900, 1600) == [500, 500, 1000, 1000]
_k = [kotak("A", 10, 10, 50, 25), kotak("B", 55, 10, 110, 25), kotak("C", 10, 40, 50, 55)]
_u, _r = urut_baca(_k)
assert [b["teks"] for b in _u] == ["A", "B", "C"] and _r == [0, 0, 1]

# --- 3. JEBAKAN urutan: label ditempel ke objek, bukan ke posisi ---
# Model memakai urut_baca(), aturan memakai urutan asli. Keduanya menunjuk
# objek dict yang sama, jadi label harus terbaca benar dari kedua urutan.
asli = [kotak("BAWAH", 10, 90, 50, 100), kotak("ATAS", 10, 10, 50, 20)]
urut, _ = urut_baca(asli)
assert [b["teks"] for b in urut] == ["ATAS", "BAWAH"], "urut_baca tidak menaik"
for b, L in zip(urut, ["store.nm", "total.total_price"]):
    b["_label"] = L
assert [b["_label"] for b in asli] == ["total.total_price", "store.nm"], \
    "label tidak ikut objek, penengah akan salah sasaran"

# --- 4. ke_json pada satu struk kecil ---
kot = [
    kotak("KEDAI", 10, 10, 90, 25, "store.nm", 0.95),
    kotak("23/01/2018", 10, 30, 90, 45, "info.date", 0.97),
    kotak("NASI", 10, 60, 60, 75, "menu.nm", 0.93),
    kotak("6.60", 200, 60, 250, 75, "menu.price", 0.91),
    kotak("16.98", 200, 100, 250, 115, "sub_total.subtotal_price", 0.99, True),
    kotak("1.02", 200, 120, 250, 135, "sub_total.tax_price", 0.96),
    kotak("18.00", 200, 140, 250, 155, "total.total_price", 0.98, True),
]
h = ke_json(kot, 300, 200)
assert h["toko"] == "KEDAI", h["toko"]
assert h["subtotal"]["nilai"] == 16.98
assert h["pajak"]["nilai"] == 1.02
assert h["total"]["nilai"] == 18.00
assert len(h["item"]) == 1 and h["item"][0]["harga"] == 6.60, h["item"]

# rantai aritmatika yang dipakai aplikasi: subtotal + pajak = total
assert abs(h["subtotal"]["nilai"] + h["pajak"]["nilai"] - h["total"]["nilai"]) <= 0.11

# --- 5. bentuk yang dikirim ke aplikasi harus cocok dengan tipe.ts ---
app = ke_kontrak_aplikasi(h, kot, "uji1", "model-uji")
assert set(app) == {"id", "versiModel", "toko", "tanggal", "jam", "item", "ringkasan"}, set(app)
assert set(app["ringkasan"]) == {"subtotal", "pajak", "total", "tunai",
                                 "kartu", "emoney", "kembalian"}, set(app["ringkasan"])
for nama, m in app["ringkasan"].items():
    assert set(m) == {"nilai", "teks", "keyakinan", "asal"}, (nama, set(m))
assert set(app["item"][0]) == {"id", "nama", "jumlah", "hargaSatuan", "harga", "keyakinan"}

assert app["tanggal"] == "23/01/2018", app["tanggal"]
assert app["ringkasan"]["total"]["nilai"] == 18.00
assert app["ringkasan"]["total"]["asal"] == "aturan", "asal tidak terbaca dari _dari_aturan"
assert app["ringkasan"]["pajak"]["asal"] == "model"
# medan yang tidak ditemukan tetap ada, dengan nilai null
assert app["ringkasan"]["tunai"] == {"nilai": None, "teks": None, "keyakinan": None, "asal": None}

# --- 6. struk tanpa total sama sekali. Tiga dari 162 struk seperti ini. ---
tanpa_total = [b for b in kot if b["_label"] != "total.total_price"]
h2 = ke_json(tanpa_total, 300, 200)
assert h2["total"] is None
app2 = ke_kontrak_aplikasi(h2, tanpa_total, "uji2", "model-uji")
assert app2["ringkasan"]["total"]["nilai"] is None, "aplikasi harus tetap jalan tanpa total"

# --- 7. akhiran berkas sementara. Pernah bikin 500 di produksi. ---
# ".img" membuat skimage memilih pembaca citra medis SimpleITK lalu gagal,
# jadi akhiran harus datang dari format yang dibaca Pillow, bukan dari
# nama berkas yang dikirim pengguna.
assert ekstensi_gambar("JPEG") == ".jpg"
assert ekstensi_gambar("PNG") == ".png"
assert ekstensi_gambar(None) is None
assert ekstensi_gambar("GIF") is None, "format tak didukung harus ditolak, bukan ditebak"

# --- 8. struk miring: nama barang tidak boleh lepas dari harganya ---
# urut_baca() pecah di atas kira-kira 3 derajat. kelompok_baris() tidak boleh.
N_BARIS = 8


def _struk_miring(kemiringan):
    kot = []
    for r in range(N_BARIS):
        y = 50 + r * 30
        for teks, x0, x1, label in [("ROTI", 20, 90, "menu.nm"),
                                    ("TAWAR", 100, 180, "menu.nm"),
                                    (f"{r + 1}.50", 240, 300, "menu.price")]:
            geser = int(x0 * kemiringan)
            kot.append(kotak(teks, x0, y + geser, x1, y + geser + 18, label))
    return kot


for _m in [0.0, 0.05, 0.10, 0.15]:
    _kot = _struk_miring(_m)
    _r = kelompok_baris(_kot)
    assert len(set(_r)) == N_BARIS, f"miring {_m}: {len(set(_r))} baris"
    _h = ke_json(_kot, 400, 350)
    assert len(_h["item"]) == N_BARIS, f"miring {_m}: {len(_h['item'])} item"
    assert [i["harga"] for i in _h["item"]] == [r + 1.5 for r in range(N_BARIS)], _h["item"]
    # Urutan kata dalam satu baris harus kiri ke kanan, bukan tercampur baris lain.
    assert all(i["nama"] == "ROTI TAWAR" for i in _h["item"]), _h["item"]

# Pembanding: cara lama memang rusak, jadi uji di atas bukan uji kosong.
assert len(set(urut_baca(_struk_miring(0.10))[1])) != N_BARIS, \
    "urut_baca ternyata tahan miring, alasan kelompok_baris gugur"

# --- 9. harga yang terpotong di titik desimal harus disatukan lagi ---
# Kasus nyata 20 Oktober 2026: "RM 10.90" di struk 99 Speed Mart terbaca
# sebagai dua kotak, dan aplikasi menampilkan 90,00.
_pecah = [
    kotak("LIFEBUOY", 20, 50, 100, 68, "menu.nm"),
    kotak("PELEMBAP", 105, 50, 185, 68, "menu.nm"),
    kotak("RM", 200, 50, 225, 68, "menu.price"),
    kotak("10.", 230, 50, 260, 68, "menu.price"),
    kotak("90", 262, 50, 290, 68, "menu.price"),
]
_h = ke_json(_pecah, 400, 200)
assert len(_h["item"]) == 1, _h["item"]
assert _h["item"][0]["harga"] == 10.90, _h["item"][0]["harga"]
assert _h["item"][0]["nama"] == "LIFEBUOY PELEMBAP", _h["item"][0]["nama"]

# --- 10. tanpa kotak menu.cnt, jumlah barang adalah satu, bukan nol ---
assert _h["item"][0]["jumlah"] == 1.0, _h["item"][0]["jumlah"]
_dgn_cnt = _pecah + [kotak("3", 10, 50, 18, 68, "menu.cnt")]
assert ke_json(_dgn_cnt, 400, 200)["item"][0]["jumlah"] == 3.0


# --- 11. harga dari letak, saat model diam ---
def _harga(kotak2):
    return ke_json(kotak2, 400, 200)["item"][0]["harga"]


_nama = [kotak("LIFEBUOY", 60, 50, 140, 68, "menu.nm"),
         kotak("PELEMBAP", 145, 50, 225, 68, "menu.nm")]

# a. kedua potongan jatuh ke O. Ini kasus COLGATE: model diam sama sekali.
assert _harga(_nama + [kotak("10.", 280, 50, 310, 68),
                       kotak("90", 312, 50, 340, 68)]) == 10.90

# b. hanya potongan kanan yang dilabeli. Ini kasus LIFEBUOY di ponsel,
#    yang menghasilkan 90,00 sebelum perbaikan ini.
assert _harga(_nama + [kotak("10.", 280, 50, 310, 68),
                       kotak("90", 312, 50, 340, 68, "menu.price")]) == 10.90

# c. kode barang di KIRI nama tidak boleh dikira harga.
assert _harga([kotak("5108", 18, 50, 50, 68)] + _nama
              + [kotak("4.99", 280, 50, 320, 68)]) == 4.99

# d. kolom kuantitas yang jauh tidak boleh ikut tertelan jadi 210,90.
assert _harga(_nama + [kotak("2", 235, 50, 245, 68),
                       kotak("10.90", 300, 50, 350, 68)]) == 10.90

# e. kata bukan angka di kanan nama diabaikan.
assert _harga(_nama + [kotak("KULI", 230, 50, 270, 68, "menu.nm"),
                       kotak("4.99", 300, 50, 340, 68)]) == 4.99

# f. tanpa kotak angka sama sekali, tetap kosong, bukan mengarang.
assert _harga(_nama) is None

# --- 12. penambalan agregasi V6 ---
from rantai import _POLA_JAM, _uang_potongan, ke_json_lengkap  # noqa: E402

# a. titik desimal yang tidak terbaca OCR dipulihkan dari dua angka ekor
assert _uang_potongan(["11", "10"]) == 11.10
assert _uang_potongan(["56", "95"]) == 56.95
assert _uang_potongan(["2 .", "05"]) == 2.05      # titiknya terbaca
assert _uang_potongan(["170", "00"]) == 170.00
# b. bilangan utuh satu kotak TIDAK boleh diotak-atik
assert _uang_potongan(["170"]) == 170.0
assert _uang_potongan(["1.50"]) == 1.50
# c. gabungan yang tidak masuk akal menghasilkan None, bukan angka ngawur
assert _uang_potongan(["123", "73", "Q0"]) is None
# d. koma atau petik di SAMBUNGAN itu titik desimal, bukan pemisah ribuan
assert _uang_potongan(["2,", "40"]) == 2.40
assert _uang_potongan(["10 ,", "55"]) == 10.55
assert _uang_potongan(["8", ",59"]) == 8.59
assert _uang_potongan(["8", "'99"]) == 8.99
assert _uang_potongan(["10 _", "29"]) == 10.29   # garis bawah lewat _uang
# pemisah ribuan ada di DALAM satu potongan, jadi tidak boleh ikut berubah
assert _uang_potongan(["1,234.50"]) == 1234.50

# d. harga bukan jam. Cacat ini pernah ada dan menaruh 7,95 di medan jam.
assert not _POLA_JAM.match("7.95")
assert not _POLA_JAM.match("1.75")
assert _POLA_JAM.match("03:29Pm") and _POLA_JAM.match("3:42")

# e. PEMISAHAN YANG WAJIB DIJAGA.
# ke_json() dipakai menyusun jawaban gold saat mengukur. Kalau penambalan
# V6 ikut masuk ke sana, gold ikut ditambal dan seluruh angka ketepatan
# membandingkan sistem dengan dirinya sendiri. Struk di bawah punya total
# yang terpotong jadi dua kotak berlabel sama; ke_json WAJIB tetap salah.
_terpotong = [
    kotak("TOTAL", 20, 50, 90, 70),
    kotak("56", 200, 50, 232, 70, "total.total_price", 0.9),
    kotak("95", 240, 50, 272, 70, "total.total_price", 0.9),
]
assert ke_json(_terpotong, 300, 100)["total"]["nilai"] == 5695.0, \
    "ke_json ternyata sudah ditambal, jawaban gold jadi tidak sah"
assert ke_json_lengkap(_terpotong, 300, 100)["total"]["nilai"] == 56.95

# f. kata kunci tidak boleh tertangkap dari nama toko
_subway = [
    kotak("SUBWAY", 20, 50, 100, 70, "store.nm", 0.9),
    kotak("12.00", 200, 50, 260, 70),
]
assert ke_json_lengkap(_subway, 300, 100)["subtotal"] is None, \
    "SUBWAY tertangkap sebagai baris subtotal"

# === 9. pisah_tahap: geometri dan pemicu ===
# EasyOCR tidak dipanggil di sini, jadi uji ini tetap cepat. Yang diperiksa
# bagian yang bisa salah sendiri: penaksir sudut, pita teks, dan pemicunya.
import numpy as _np  # noqa: E402

import pisah_tahap as _PT  # noqa: E402

# Gambar buatan: lima baris teks mendatar di tengah kertas putih.
_g = _np.full((200, 300), 240, _np.uint8)
for _y in range(30, 180, 30):
    _g[_y:_y + 8, 80:220] = 20
_m = _PT._peta_teks(_g)
_x0, _x1 = _PT._pita_teks(_m)
# Pita harus jatuh di sekitar tulisan, bukan selebar gambar.
assert 60 <= _x0 <= 95 and 205 <= _x1 <= 245, (_x0, _x1)
# Sudah mendatar, jadi sudut terbaiknya nol.
assert abs(_PT._sudut(_m, _x0, _x1)) <= 0.25, _PT._sudut(_m, _x0, _x1)

# Gambar yang sama dimiringkan 3 derajat harus menghasilkan taksiran -3,
# yaitu putaran yang mengembalikannya jadi mendatar.
import cv2 as _cv  # noqa: E402

_miring = _cv.warpAffine(_g, _cv.getRotationMatrix2D((150.0, 100.0), 3.0, 1.0),
                         (300, 200), borderValue=240)
_mm = _PT._peta_teks(_miring)
_a0, _a1 = _PT._pita_teks(_mm)
assert abs(_PT._sudut(_mm, _a0, _a1) + 3.0) <= 0.5, _PT._sudut(_mm, _a0, _a1)

# Kertas kosong tidak boleh membuat _pita_teks melempar atau memberi pita nol.
_kosong = _PT._peta_teks(_np.full((50, 60), 200, _np.uint8))
_k0, _k1 = _PT._pita_teks(_kosong)
assert _k1 > _k0

# Pemicu: menyala HANYA kalau tidak satu pun item berharga.
assert _PT.n_harga_item({"item": [{"harga": 1.5}, {"harga": None}]}) == 1
assert _PT.n_harga_item({"item": [{"harga": True}]}) == 0      # bool bukan harga
assert _PT.perlu({"item": [{"harga": None}, {"harga": None}]})
assert _PT.perlu({"item": []})
assert not _PT.perlu({"item": [{"harga": 0.6}]})
assert not _PT.perlu(None)

# --- 13. struk LOGAM M&A: koma di dalam potongan, barcode, dan dua baris ---
from rantai import _buang_barcode, _satukan_dua_baris  # noqa: E402

# a. koma dengan TEPAT dua angka di ujung itu titik desimal, bukan ribuan.
# Diukur pada struk LOGAM: "8, 00" dulu jadi 800,0 dan "~0 , 50" jadi 50,0.
assert _uang("8, 00") == 8.00, _uang("8, 00")
assert _uang("8 , 50") == 8.50, _uang("8 , 50")
assert _uang("~0 , 50") == 0.50, _uang("~0 , 50")
assert _uang("1,234,50") == 1234.50, _uang("1,234,50")
# tiga angka setelah koma itu pemisah ribuan sungguhan, jangan disentuh
assert _uang("1,234") == 1234.0, _uang("1,234")
assert _uang("1,250") == 1250.0, _uang("1,250")
assert _uang("26,00") == 26.00, _uang("26,00")

# b. barcode yang menempel di depan nama barang dipotong.
assert _buang_barcode('"08993175566438  NABAT I') == "NABAT I"
assert _buang_barcode("08993175566438") == ""
assert _buang_barcode("48a555149276688 ANGGUR  C/M SHI") == "ANGGUR C/M SHI"
assert _buang_barcode("NABAT I AHH") == "NABAT I AHH"
# nama yang kebetulan diawali angka pendek tidak boleh ikut terpotong
assert _buang_barcode("100PLUS 1.5L") == "100PLUS 1.5L"
assert _buang_barcode("680 BRUSH") == "680 BRUSH"

# c. nama di baris atas, harga di baris bawah, disatukan jadi satu item.
_dua = [
    kotak("80ML", 20, 50, 80, 68, "menu.nm"),
    kotak("CA", 85, 50, 115, 68, "menu.nm"),
    kotak("GLUE", 120, 50, 170, 68, "menu.nm"),
    kotak("50100026", 20, 90, 100, 108, "menu.num"),
    kotak("1", 200, 90, 210, 108, "menu.cnt"),
    kotak("8.00", 250, 90, 300, 108, "menu.unitprice"),
    kotak("8, 00", 330, 90, 390, 108, "menu.price"),
]
_hd = ke_json(_dua, 400, 200)
assert len(_hd["item"]) == 1, _hd["item"]
assert _hd["item"][0]["nama"] == "80ML CA GLUE", _hd["item"][0]["nama"]
assert _hd["item"][0]["harga"] == 8.00, _hd["item"][0]["harga"]

# Struk satu baris tidak boleh tersentuh: dua item tetap dua item.
_satu = [
    kotak("KOPI", 20, 50, 80, 68, "menu.nm"),
    kotak("3.00", 330, 50, 390, 68, "menu.price"),
    kotak("TEH", 20, 90, 80, 108, "menu.nm"),
    kotak("4.00", 330, 90, 390, 108, "menu.price"),
]
assert len(ke_json(_satu, 400, 200)["item"]) == 2

# Baris nama yang sudah punya harga tidak boleh menelan baris di bawahnya.
_takut = [{"nama": "A", "harga": 1.0, "harga_satuan": None, "diskon": None,
           "jumlah": 1.0, "yakin": 0.9},
          {"nama": None, "harga": 2.0, "harga_satuan": None, "diskon": None,
           "jumlah": 1.0, "yakin": 0.9}]
assert len(_satukan_dua_baris(_takut)) == 2
# Baris diskon tanpa nama sesudah baris yang sudah lengkap tetap berdiri sendiri.
assert len(_satukan_dua_baris(_takut[1:])) == 1

# --- 14. pengaman harga item mustahil ---
from rantai import _buang_harga_mustahil, _uang_potongan as _up  # noqa: E402

# P5. Spasi dengan tepat dua angka ekor itu titik desimal.
assert _up(["RM76 90"]) == 76.90, _up(["RM76 90"])
assert _up(["27", "90"]) == 27.90
# tiga angka setelah spasi itu pemisah ribuan, jangan disentuh
assert _up(["1 234"]) == 1234.0, _up(["1 234"])
# dan _uang POLOS tidak boleh ikut berubah, sebab ia menyusun jawaban gold
assert _uang("76 90") == 7690.0, _uang("76 90")


def _it(*harga):
    return [{"nama": f"B{i}", "harga": h, "jumlah": 1.0, "harga_satuan": None,
             "diskon": None, "yakin": 0.9} for i, h in enumerate(harga)]


# P1. Bilangan bulat empat angka atau lebih itu kode barang, bukan harga.
_h = _buang_harga_mustahil(_it(2.6, 2.8, 72780.0, 0.6))
assert [x["harga"] for x in _h] == [2.6, 2.8, None, 0.6], [x["harga"] for x in _h]
# harga empat angka BERDESIMAL tidak tersentuh, ia bisa jadi harga sungguhan
_h = _buang_harga_mustahil(_it(1200.50, 1100.0, 1300.0))
assert _h[0]["harga"] == 1200.50, _h[0]["harga"]

# P2. Lebih dari 15 kali median harga item lain.
_h = _buang_harga_mustahil(_it(2.0, 2.0, 2.0, 90.0))
assert [x["harga"] for x in _h] == [2.0, 2.0, 2.0, None], [x["harga"] for x in _h]
# 10 kali median masih wajar: satu barang mahal di antara barang murah
_h = _buang_harga_mustahil(_it(2.0, 2.0, 2.0, 20.0))
assert [x["harga"] for x in _h] == [2.0, 2.0, 2.0, 20.0], [x["harga"] for x in _h]
# satu item saja tidak punya pembanding, jadi tidak boleh dibuang
assert _buang_harga_mustahil(_it(999.0))[0]["harga"] == 999.0
# harga None dan nama tetap utuh, yang dikosongkan hanya harganya
_h = _buang_harga_mustahil(_it(2.0, None, 99999.0))
assert _h[2]["nama"] == "B2" and _h[2]["harga"] is None


# --- 12. total lewat kata kunci sebaris, dari diagnosis uji3 ---
# Model hanya melabeli total.total_price pada 3 dari 11 struk uji3. Kata TOTAL
# juga muncul di baris yang BUKAN jumlah akhir, jadi yang diuji di sini adalah
# pembatal dan pengambilan baris paling bawah.
from rantai import ke_json_lengkap as _kjl  # noqa: E402


def _baris_uang(teks_kiri, teks_kanan, y):
    """Satu baris: kata kunci di kiri, nilainya di kanan."""
    return [kotak(teks_kiri, 10, y, 200, y + 20),
            kotak(teks_kanan, 400, y, 480, y + 20)]


_struk = (_baris_uang("Total Item", "4", 10)
          + _baris_uang("Sub Total", "39.23", 40)
          + _baris_uang("Total Savings", "5.32", 70)
          + _baris_uang("Total", "39.25", 100))
_h = _kjl(_struk, 500, 200)
# baris paling bawah yang lolos pembatal, bukan Sub Total dan bukan Savings
assert (_h["total"] or {}).get("nilai") == 39.25, _h["total"]
assert (_h["subtotal"] or {}).get("nilai") == 39.23, _h["subtotal"]

# Tanpa baris TOTAL murni, medan total dibiarkan kosong, tidak diisi Sub Total
_h = _kjl(_baris_uang("Sub Total", "39.23", 40)
          + _baris_uang("Total Qty", "4", 70), 500, 200)
assert (_h["total"] or {}).get("nilai") is None, _h["total"]

# TOTAL SALES EXCLUSIVE SST bukan jumlah akhir, yang inclusive baru jumlah akhir
_h = _kjl(_baris_uang("Total Sales Exclusive SST", "10.19", 10)
          + _baris_uang("Total Sales inclusive of SST", "10.80", 40), 500, 200)
assert (_h["total"] or {}).get("nilai") == 10.80, _h["total"]

# --- 13. jumlah nol itu mustahil, dari diagnosis uji3 ---
# Struk 01 memberi jumlah 0 pada dua barang, terambil dari kolom diskon 0.00.
_brs = [kotak("Perkedel", 10, 10, 120, 30, label="menu.nm"),
        kotak("0", 200, 10, 220, 30, label="menu.cnt"),
        kotak("3.50", 400, 10, 460, 30, label="menu.price")]
_it1 = ke_json_lengkap(_brs, 500, 50)["item"]
assert len(_it1) == 1, _it1
assert _it1[0]["jumlah"] == 1.0, _it1[0]

# jumlah yang sah tidak diubah
_brs[1] = kotak("2", 200, 10, 220, 30, label="menu.cnt")
assert ke_json_lengkap(_brs, 500, 50)["item"][0]["jumlah"] == 2.0

# --- 14. harga satuan mustahil dikosongkan, dari diagnosis uji3 ---
from rantai import _buang_harga_mustahil as _bhm  # noqa: E402


def _is(hs, hr, jm=1.0):
    return [{"nama": "A", "jumlah": jm, "harga_satuan": hs,
             "harga": hr, "diskon": None, "yakin": 0.9}]


# 15900,0 untuk baris 15,90 dan 50,0 untuk baris 3,50: kolom tertukar
assert _bhm(_is(15900.0, 15.90))[0]["harga_satuan"] is None
assert _bhm(_is(50.0, 3.50))[0]["harga_satuan"] is None
# selisih wajar tidak disentuh
assert _bhm(_is(5.00, 5.45))[0]["harga_satuan"] == 5.00
assert _bhm(_is(12.20, 8.99))[0]["harga_satuan"] == 12.20
# barang timbangan: 0,15 kg seharga RM 20 per kg, jumlah di bawah satu
assert _bhm(_is(20.0, 3.00, jm=0.15))[0]["harga_satuan"] == 20.0
# harga baris tidak diketahui, tidak ada dasar membuang
assert _bhm(_is(15900.0, None))[0]["harga_satuan"] == 15900.0
# --- 15. huruf yang _uang petakan ke angka juga lolos _mirip_uang ---
# Terukur di uji3: "Service Taxe6% RMO 48" untuk pajak 0,48 dan
# "TAXABLE AMT RM3O . 80" untuk 30,80. Tanpa ini kotak berhuruf disaring
# habis sebelum _uang berkesempatan memulihkannya, dan yang tersisa hanya
# ekor desimalnya, yaitu 48,0 dan 80,0.
from rantai import _mirip_uang as _mu  # noqa: E402

assert _mu("RMO") is True            # RM0, nol ringgit
assert _mu("RM3O") is True           # RM30
assert _mu("1O.50") is True
assert _mu("l0.50") is True
assert _mu("RM32_") is True
# Huruf tanpa lambang mata uang TIDAK boleh jadi uang
assert _mu("O") is False
assert _mu("OO") is False
assert _mu("I") is False
assert _mu("LOT") is False
assert _mu("NO") is False
assert _mu("Total") is False
assert _mu("") is False
assert _mu("RM") is False            # lambang saja, tanpa angka

# --- 16. kata kunci dinilai pada jendela di kiri angka, bukan seluruh baris ---
# Struk 11 mencetak dua pasangan dalam SATU baris:
#     Total Saving 5. 32 Total 39, 25
# Menilai seluruh baris membuat pembatal SAVING membatalkan 39,25 yang justru
# jawaban benarnya. Jendela berhenti pada angka pertama ke kiri, jadi yang
# dinilai hanya kata "Total" yang menempel pada angka terpilih.
_gabung = [kotak("Total", 10, 10, 90, 30), kotak("Saving", 100, 10, 180, 30),
           kotak("5.", 200, 10, 230, 30), kotak("32", 235, 10, 265, 30),
           kotak("Total", 300, 10, 380, 30), kotak("39,", 400, 10, 430, 30),
           kotak("25", 435, 10, 465, 30)]
_h = _kjl(_gabung, 500, 50)
assert (_h["total"] or {}).get("nilai") == 39.25, _h["total"]

# Pembatal tetap bekerja ketika ia memang menempel pada angkanya
_h = _kjl([kotak("Total", 10, 10, 90, 30), kotak("Saving", 100, 10, 180, 30),
           kotak("5.", 200, 10, 230, 30), kotak("32", 235, 10, 265, 30)], 500, 50)
assert (_h["total"] or {}).get("nilai") is None, _h["total"]

# Sub Total sebaris tetap dibatalkan walau kata Total ada tepat di kiri angka
_h = _kjl([kotak("Total", 10, 10, 60, 30), kotak("Item", 70, 10, 120, 30),
           kotak("Sub", 200, 10, 240, 30), kotak("Total", 245, 10, 300, 30),
           kotak("39.23", 400, 10, 470, 30)], 500, 50)
assert (_h["total"] or {}).get("nilai") is None, _h["total"]

# --- 17. pajak dari pasangan angka sebaris di baris ringkasan SST ---
# Struk Malaysia mencetak tarif, dasar pengenaan, lalu nilai pajak dalam satu
# baris. Mengambil gugus paling kanan tanpa memeriksa apa pun memberi 0,04
# (pembulatan) di struk 01 dan 0,0 di struk 04.
from rantai import (_cadangan_pajak_sebaris as _cps,  # noqa: E402
                    _angka_tarif as _at, kelompok_baris as _kbr)

_sst = [kotak("SST", 10, 10, 60, 30), kotak("6%", 70, 10, 110, 30),
        kotak("81.00", 250, 10, 330, 30), kotak("4.84", 400, 10, 470, 30)]

# pajak salah ditimpa oleh pasangan yang terbukti rasionya, dasar jadi subtotal
_h = {"pajak": {"teks": "0.04", "nilai": 0.04, "yakin": 0.9, "bbox": [0, 0, 1, 1]},
      "subtotal": None}
_cps(_sst, _kbr(_sst), _h)
assert _h["pajak"]["nilai"] == 4.84, _h["pajak"]
assert _h["subtotal"]["nilai"] == 81.00, _h["subtotal"]

# pajak yang SUDAH memenuhi tarif tidak pernah diganti
_h = {"pajak": {"teks": "4.84", "nilai": 4.84, "yakin": 0.9, "bbox": [0, 0, 1, 1]},
      "subtotal": None}
_cps(_sst, _kbr(_sst), _h)
assert _h["pajak"]["nilai"] == 4.84, _h["pajak"]

# subtotal yang sudah terisi tidak ditimpa dasar pengenaan pajak
_h = {"pajak": None,
      "subtotal": {"teks": "80.00", "nilai": 80.00, "yakin": 0.9, "bbox": [0, 0, 1, 1]}}
_cps(_sst, _kbr(_sst), _h)
assert _h["pajak"]["nilai"] == 4.84, _h["pajak"]
assert _h["subtotal"]["nilai"] == 80.00, _h["subtotal"]

# baris pajak tanpa pasangan yang memenuhi tarif tidak menyentuh apa pun
_bukan = [kotak("SST", 10, 10, 60, 30), kotak("12.00", 250, 10, 330, 30),
          kotak("9.50", 400, 10, 470, 30)]
_h = {"pajak": None, "subtotal": None}
_cps(_bukan, _kbr(_bukan), _h)
assert _h["pajak"] is None, _h["pajak"]

# baris yang dibatalkan tidak dipakai walau rasionya kebetulan cocok
_batal = [kotak("Tax", 10, 10, 50, 30), kotak("Not", 55, 10, 90, 30),
          kotak("Applicable", 95, 10, 190, 30),
          kotak("81.00", 250, 10, 330, 30), kotak("4.84", 400, 10, 470, 30)]
_h = {"pajak": None, "subtotal": None}
_cps(_batal, _kbr(_batal), _h)
assert _h["pajak"] is None, _h["pajak"]

# angka tarif ditolak sebagai dasar pengenaan pajak
assert _at([kotak("6%", 0, 0, 1, 1)], 6.0) is True
assert _at([kotak("6", 0, 0, 1, 1)], 6.0) is True
assert _at([kotak("8", 0, 0, 1, 1)], 8.0) is True
assert _at([kotak("6.00", 0, 0, 1, 1)], 6.0) is False      # bertitik, nilai uang
assert _at([kotak("81.00", 0, 0, 1, 1)], 81.0) is False

# Tarif 6 tidak boleh jadi dasar: pasangan (6, 0.48) lolos pita 8 persen
# kalau penolakan tarif dilepas, dan itu terjadi di struk 09.
_tarif = [kotak("Service", 10, 10, 80, 30), kotak("Tax", 85, 10, 120, 30),
          kotak("6", 130, 10, 150, 30), kotak("0.48", 400, 10, 470, 30)]
_h = {"pajak": None, "subtotal": None}
_cps(_tarif, _kbr(_tarif), _h)
assert _h["subtotal"] is None, _h["subtotal"]

# gugus kiri ke kanan, bukan hanya yang paling kanan
from rantai import _gugus_semua as _gs, _ambang_baris as _ab  # noqa: E402

_g = _gs(_sst, _ab(_sst))
assert [g[0]["teks"] for g in _g] == ["81.00", "4.84"], _g

# --- 18. penyambungan pengenal huruf PP-OCR ---
# Yang diuji penyambungannya, bukan ketepatan pengenalnya: bahwa jalur bawaan
# tidak berubah sedikit pun, bahwa bbox tidak pernah disentuh, dan bahwa objek
# kotak yang sama dikembalikan. Yang terakhir penting sebab sisa rantai
# menempelkan label ke OBJEK kotak, bukan ke posisinya di daftar.
import os  # noqa: E402

import pengenal_ppocr as _pp  # noqa: E402

_simpan = os.environ.get("PENGENAL")
try:
    # Bawaan: mati, dan baca_ulang tidak mengubah apa pun
    os.environ.pop("PENGENAL", None)
    assert _pp.aktif() is False
    assert _pp.nama() == "EasyOCR CRNN"
    assert _pp.muat() is None
    _kt = [{"bbox": [10, 10, 60, 30], "teks": "RMO", "teks_asli": "RMO",
            "skor": 0.5}]
    _hasil = _pp.baca_ulang("berkas-yang-tidak-ada.jpg", _kt)
    assert _hasil is _kt                      # objek yang sama
    assert _kt[0]["teks"] == "RMO"            # tidak disentuh
    assert _kt[0]["bbox"] == [10, 10, 60, 30]

    os.environ["PENGENAL"] = "easyocr"
    assert _pp.aktif() is False
    os.environ["PENGENAL"] = "  PPOCR  "      # huruf besar dan spasi
    assert _pp.aktif() is True
    assert "PP-OCRv4" in _pp.nama()

    # Daftar kotak kosong tidak pernah menyentuh pengenal, walau dinyalakan
    assert _pp.baca_ulang("berkas-yang-tidak-ada.jpg", []) == []

    # Jalur nyala hanya diuji kalau pustakanya ada, supaya build tanpa
    # rapidocr tetap bisa menjalankan seluruh uji ini.
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        print("  bagian 18: jalur ppocr dilewati, rapidocr tidak terpasang")
    else:
        from PIL import Image, ImageDraw

        _img = Image.new("RGB", (240, 60), "white")
        ImageDraw.Draw(_img).text((10, 20), "1234", fill="black")
        _kt = [{"bbox": [5, 15, 70, 45], "teks": "lama", "teks_asli": "lama",
                "skor": 0.1}]
        _sebelum = _pp.gagal()
        _hasil = _pp.baca_ulang(_img, _kt, normalisasi=lambda t: t)
        assert _hasil is _kt                          # objek yang sama
        assert _kt[0]["bbox"] == [5, 15, 70, 45]      # bbox tidak disentuh
        assert isinstance(_kt[0]["teks"], str)
        assert _pp.gagal() == _sebelum                # tidak ada potongan gagal

        # Potongan yang terlalu kecil dilewati, bukan menjatuhkan permintaan
        _kecil = [{"bbox": [0, 0, 1, 1], "teks": "tetap", "teks_asli": "tetap",
                   "skor": 0.1}]
        _pp.baca_ulang(_img, _kecil, normalisasi=lambda t: t)
        assert _kecil[0]["teks"] == "tetap"
finally:
    if _simpan is None:
        os.environ.pop("PENGENAL", None)
    else:
        os.environ["PENGENAL"] = _simpan

print("semua uji lolos")
