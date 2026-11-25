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


# --- 14b. total lewat kata kunci sebaris, dari diagnosis uji3 ---
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

# --- 14c. jumlah nol itu mustahil, dari diagnosis uji3 ---
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

# --- 14d. harga satuan mustahil dikosongkan, dari diagnosis uji3 ---
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
    # Bawaannya ppocr sejak 18 November 2026, dan PENGENAL yang tidak disetel
    # TIDAK dianggap permintaan eksplisit. Bedanya menentukan apakah pustaka
    # yang hilang menjatuhkan peladen atau hanya membuatnya mundur.
    os.environ.pop("PENGENAL", None)
    assert _pp.BAWAAN == "ppocr"
    assert _pp.aktif() is True
    assert _pp.diminta() is False

    # Disetel eksplisit: aktif, dan dianggap permintaan
    os.environ["PENGENAL"] = "  PPOCR  "      # huruf besar dan spasi
    assert _pp.aktif() is True
    assert _pp.diminta() is True
    assert "PP-OCRv4" in _pp.nama()

    # Dimatikan: baca_ulang tidak mengubah apa pun
    os.environ["PENGENAL"] = "easyocr"
    assert _pp.aktif() is False
    assert _pp.diminta() is False
    assert _pp.nama() == "EasyOCR CRNN"
    assert _pp.muat() is None
    _kt = [{"bbox": [10, 10, 60, 30], "teks": "RMO", "teks_asli": "RMO",
            "skor": 0.5}]
    _hasil = _pp.baca_ulang("berkas-yang-tidak-ada.jpg", _kt)
    assert _hasil is _kt                      # objek yang sama
    assert _kt[0]["teks"] == "RMO"            # tidak disentuh
    assert _kt[0]["bbox"] == [10, 10, 60, 30]

    # Daftar kotak kosong tidak pernah menyentuh pengenal, walau dinyalakan
    os.environ["PENGENAL"] = "ppocr"
    assert _pp.baca_ulang("berkas-yang-tidak-ada.jpg", []) == []

    # Huruf di luar ASCII dibuang. Model PP-OCRv4 bawaan rapidocr adalah model
    # Tionghoa ditambah Inggris, dan pada potongan berderau ia mengeluarkan
    # huruf Han: 17 dari 1326 kotak uji3, 12 di antaranya huruf Han, sedangkan
    # EasyOCR nol. Struk Malaysia tidak memuat huruf Han.
    assert _pp._bersihkan("Sub Total \u660e 16. 75") == "Sub Total  16. 75"
    assert _pp._bersihkan("\u54c1") == ""
    assert _pp._bersihkan("2-55\u00b0") == "2-55"
    assert _pp._bersihkan("57\u3001") == "57"
    assert _pp._bersihkan("  RM0 48  ") == "RM0 48"
    assert _pp._bersihkan("TOTAL 16.75") == "TOTAL 16.75"

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

# --- 19. kotak dari deteksi CRAFT saja, tanpa pengenal CRNN ---
# Dipakai saat PENGENAL=ppocr, sebab teks CRNN akan ditimpa seluruhnya.
# Yang diuji perubahan bentuknya: detect memberi [x_kiri, x_kanan, y_atas,
# y_bawah] sedangkan sisa rantai memakai [x0, y0, x1, y1], dan tertukar di
# sini berarti setiap kotak salah tempat tanpa satu pun pesan galat.


class _PembacaPalsu:
    """Pengganti easyocr.Reader yang hanya mengembalikan kotak yang diatur."""

    def __init__(self, mendatar, bebas):
        self.mendatar = mendatar
        self.bebas = bebas
        self.dipanggil = None

    def detect(self, jalur, width_ths=None):
        self.dipanggil = (jalur, width_ths)
        return [self.mendatar], [self.bebas]


# satu kotak mendatar di bawah, satu di atas, satu poligon bebas paling bawah
# kotak_craft murni perubahan bentuk, jadi ia tidak bergantung pada PENGENAL
_pb = _PembacaPalsu([[100, 200, 50, 70], [10, 60, 10, 30]],
                    [[[5, 100], [40, 102], [40, 120], [5, 118]]])
_kt = _pp.kotak_craft(_pb, "foto.jpg", 0.0)

# width_ths diteruskan apa adanya, bukan dibiarkan memakai bawaan EasyOCR 0,5
assert _pb.dipanggil == ("foto.jpg", 0.0), _pb.dipanggil
# urutan baca, yaitu dari atas ke bawah lalu kiri ke kanan
assert [b["bbox"] for b in _kt] == [[10, 10, 60, 30],
                                    [100, 50, 200, 70],
                                    [5, 100, 40, 120]], [b["bbox"] for b in _kt]
# teks dibiarkan kosong, sebab pengenalnya yang mengisinya
assert all(b["teks"] == "" and b["teks_asli"] == "" for b in _kt)
assert all(b["skor"] == 0.0 for b in _kt)

# tidak ada kotak sama sekali: daftar kosong, bukan galat
assert _pp.kotak_craft(_PembacaPalsu([], []), "foto.jpg") == []

# poligon bebas saja, tanpa kotak mendatar
_kt = _pp.kotak_craft(_PembacaPalsu([], [[[9, 9], [21, 11], [20, 31], [8, 29]]]),
                      "foto.jpg")
assert [b["bbox"] for b in _kt] == [[8, 9, 21, 31]], [b["bbox"] for b in _kt]

# pencatat waktu ada dan berupa angka, dipakai endpoint sehat
assert isinstance(_pp.detik(), float)


# ---------------------------------------------------------------------------
# 20. GERBANG VLM VERSI AMAN
#
# Yang dijaga bagian ini bukan ketepatan VLM, sebab tidak ada VLM di sini,
# melainkan dua janji gerbangnya: nilai 0 tidak pernah dianggap kosong, dan
# medan yang sudah berisi tidak pernah ditimpa. Keduanya yang membuat gerbang
# usulan terukur menyerahkan 12 dari 39 medan benar uji3 ke VLM.
# ---------------------------------------------------------------------------
import gerbang_vlm as _gv  # noqa: E402

_TINGGI = 1000
_ATAS = [{"bbox": [10, 20, 100, 40]}, {"bbox": [10, 50, 100, 70]}]
_BAWAH = [{"bbox": [10, 900, 100, 920]}, {"bbox": [10, 930, 100, 950]}]


def _kontrak(**medan):
    """Kontrak aplikasi tiruan, semua medan kosong kecuali yang diberikan."""
    r = {k: {"nilai": None, "teks": None, "keyakinan": None, "asal": None}
         for k in _gv.MEDAN_UANG}
    for k, v in medan.items():
        if k in r:
            r[k]["nilai"] = v
    return {"toko": medan.get("toko"), "tanggal": medan.get("tanggal"),
            "jam": medan.get("jam"), "ringkasan": r}


def _nama(daftar):
    return [m["medan"] for m in daftar]


# semua kosong dan kedua zona berisi kotak: kedelapan medan diserahkan
_semua = _nama(_gv.perlu_tambal(_kontrak(), _ATAS + _BAWAH, _TINGGI))
assert _semua == ["toko", "tanggal", "jam", "subtotal", "pajak", "total",
                  "tunai", "kembalian"], _semua

# NOL BUKAN KOSONG. Empat medan jawaban benar uji3 memang nol, dan gerbang
# usulan yang menganggapnya kosong itulah salah satu sebab terbesar ia
# menyerahkan medan yang sudah benar.
_n = _nama(_gv.perlu_tambal(_kontrak(total=0.0), _ATAS + _BAWAH, _TINGGI))
assert "total" not in _n, _n
assert len(_n) == 7, _n
_n = _nama(_gv.perlu_tambal(_kontrak(kembalian=0), _ATAS + _BAWAH, _TINGGI))
assert "kembalian" not in _n, _n

# medan teks yang sudah berisi tidak diserahkan, yang berisi spasi saja ya
_n = _nama(_gv.perlu_tambal(_kontrak(toko="KEDAI"), _ATAS + _BAWAH, _TINGGI))
assert "toko" not in _n, _n
_n = _nama(_gv.perlu_tambal(_kontrak(toko="   "), _ATAS + _BAWAH, _TINGGI))
assert "toko" in _n, _n

# PENJAGA ZONA. Kalau pengenal tidak menemukan teks di kaki struk, medan uang
# tidak boleh ditanyakan: tidak ada yang bisa diperiksa terhadap jawaban VLM.
assert _nama(_gv.perlu_tambal(_kontrak(), _ATAS, _TINGGI)) ==     ["toko", "tanggal", "jam"]
assert _nama(_gv.perlu_tambal(_kontrak(), _BAWAH, _TINGGI)) ==     ["subtotal", "pajak", "total", "tunai", "kembalian"]
assert _gv.perlu_tambal(_kontrak(), [], _TINGGI) == []
# satu kotak saja belum cukup: medan menuntut label dan nilainya
assert _gv.perlu_tambal(_kontrak(), _BAWAH[:1], _TINGGI) == []
# tinggi citra tidak diketahui: gerbang menutup, bukan menebak
assert _gv.perlu_tambal(_kontrak(), _ATAS + _BAWAH, 0) == []

# zona dihitung dari TINGGI CITRA, bukan dari rentang kotak. Kalau dari
# rentang kotak, kedua kotak kepala ini akan ikut mengisi zona bawah dan
# penjaganya kehilangan guna.
assert _nama(_gv.perlu_tambal(_kontrak(), _ATAS, 10000)) ==     ["toko", "tanggal", "jam"]

# penambalan menulis, menandai asalnya, dan mencatat daftarnya di akar
_k = _kontrak()
_tulis = _gv.tambal(_k, {"total": 93.90, "jam": "19:55:44"},
                    _ATAS + _BAWAH, _TINGGI)
assert sorted(_tulis) == ["jam", "total"], _tulis
assert _k["ringkasan"]["total"]["nilai"] == 93.90
assert _k["ringkasan"]["total"]["asal"] == "vlm"
assert _k["ringkasan"]["total"]["keyakinan"] is None
assert _k["jam"] == "19:55:44"
assert _k["asalVlm"] == ["jam", "total"], _k["asalVlm"]

# TIDAK PERNAH MENIMPA. Jawaban VLM untuk medan yang sudah berisi dibuang,
# walaupun pemanggilnya memaksakannya.
_k = _kontrak(total=16.75, toko="99 SPEED MART")
_tulis = _gv.tambal(_k, {"total": 99.0, "toko": "SALAH"},
                    _ATAS + _BAWAH, _TINGGI)
assert _tulis == [], _tulis
assert _k["ringkasan"]["total"]["nilai"] == 16.75
assert _k["ringkasan"]["total"]["asal"] is None
assert _k["toko"] == "99 SPEED MART"
assert "asalVlm" not in _k

# SATU STRUK SATU CARA BAYAR. Kalau kartu atau dompet digital sudah berisi,
# tunai yang kosong itu memang kosong. Tanpa ini, struk ST Rosyam Mart yang
# dibayar AMBANK TNGWALLET menampilkan pembayaran 61,20 DUA KALI di aplikasi,
# sebab model mengisi emoney lalu VLM mengisi tunai dari baris yang sama.
_k = _kontrak()
_k["ringkasan"]["emoney"] = {"nilai": 61.20, "teks": "61.20", "keyakinan": 0.9,
                             "asal": "model"}
assert "tunai" not in _nama(_gv.perlu_tambal(_k, _ATAS + _BAWAH, _TINGGI))
_k = _kontrak()
_k["ringkasan"]["kartu"] = {"nilai": 61.20, "teks": "61.20", "keyakinan": 0.9,
                            "asal": "model"}
assert "tunai" not in _nama(_gv.perlu_tambal(_k, _ATAS + _BAWAH, _TINGGI))
# tanpa pembayaran nontunai, tunai tetap ditanyakan
assert "tunai" in _nama(_gv.perlu_tambal(_kontrak(), _ATAS + _BAWAH, _TINGGI))
# medan lain tidak ikut terkunci
_k = _kontrak()
_k["ringkasan"]["emoney"] = {"nilai": 61.20, "teks": "61.20", "keyakinan": 0.9,
                             "asal": "model"}
assert "kembalian" in _nama(_gv.perlu_tambal(_k, _ATAS + _BAWAH, _TINGGI))

# PITA PAJAK. Dua kesalahan VLM yang terukur pada uji3 keduanya di medan
# pajak, dan memperketat perintah terbukti tidak menolong sama sekali.
# Angka-angka di bawah ini diambil persis dari dua struk itu.

# uji3_11: pajaknya Service Tax@6% RM0.48, dan VLM menjawab Rounding Adj 0,02
_k = _kontrak(subtotal=8.00)
assert _gv.tambal(_k, {"pajak": 0.48}, _ATAS + _BAWAH, _TINGGI) == ["pajak"]
assert _k["ringkasan"]["pajak"]["nilai"] == 0.48
_k = _kontrak(subtotal=8.00)
assert _gv.tambal(_k, {"pajak": 0.02}, _ATAS + _BAWAH, _TINGGI) == []
assert _k["ringkasan"]["pajak"]["nilai"] is None

# uji3_05: subtotal BELUM ada, jadi dasarnya total. Others 9,20 itu sisa
# bayaran, bukan pajak, dan struk itu memang tidak memuat pajak.
_k = _kontrak(total=57.80)
assert _gv.tambal(_k, {"pajak": 9.20}, _ATAS + _BAWAH, _TINGGI) == []

# subtotal yang ikut dijawab VLM pada permintaan yang sama tetap jadi dasar,
# jadi hasilnya tidak bergantung pada urutan penulisan
_k = _kontrak()
_t = _gv.tambal(_k, {"subtotal": 57.80, "pajak": 9.20}, _ATAS + _BAWAH, _TINGGI)
assert _t == ["subtotal"], _t

# 8 persen diterima, dan pita dihitung terhadap total juga
assert _gv.tambal(_kontrak(subtotal=100.0), {"pajak": 8.00},
                  _ATAS + _BAWAH, _TINGGI) == ["pajak"]
assert _gv.tambal(_kontrak(total=8.50), {"pajak": 0.48},
                  _ATAS + _BAWAH, _TINGGI) == ["pajak"]

# TANPA DASAR, jawaban diloloskan. Menolak tanpa bukti itu persis kesalahan
# veto matematika yang sudah terukur membuang jawaban benar.
_k = _kontrak()
assert _gv.tambal(_k, {"pajak": 9.20}, _ATAS + _BAWAH, _TINGGI) == ["pajak"]

# pita hanya berlaku untuk pajak, medan lain tidak disentuh
_k = _kontrak(subtotal=8.00)
assert _gv.tambal(_k, {"total": 999.0}, _ATAS + _BAWAH, _TINGGI) == ["total"]

# medan yang gerbangnya tolak karena zona pun tidak tertulis
_k = _kontrak()
assert _gv.tambal(_k, {"total": 93.90}, _ATAS, _TINGGI) == []
assert _k["ringkasan"]["total"]["nilai"] is None

# jawaban kosong dari VLM bukan jawaban
_k = _kontrak()
assert _gv.tambal(_k, {"total": None, "jam": ""}, _ATAS + _BAWAH, _TINGGI) == []
assert _gv.tambal(_k, {}, _ATAS + _BAWAH, _TINGGI) == []
assert "asalVlm" not in _k

print("20. gerbang VLM: lolos")


# ---------------------------------------------------------------------------
# 21. PEMANGGIL VLM
#
# Tidak ada VLM sungguhan di sini. Yang dijaga: bawaannya mati, galat tidak
# menjatuhkan struk, dan VLM TIDAK BISA menulis ke medan yang tidak ditanyakan.
# Yang terakhir itu penting: tanpa penyaringan ini, satu balasan yang murah
# hati bisa mengisi medan yang gerbangnya tidak pernah membuka.
# ---------------------------------------------------------------------------
import vlm as _vlm  # noqa: E402
from PIL import Image as _Img  # noqa: E402

_CITRA = _Img.new("RGB", (40, 90), (255, 255, 255))
_simpan_vlm = (os.environ.get("VLM"), os.environ.get("KUNCI_VLM"))


def _balas(isi):
    """Pengirim palsu yang mengembalikan satu balasan bergaya OpenAI."""
    def kirim(muatan):
        kirim.muatan = muatan
        return {"choices": [{"message": {"content": isi}}]}
    kirim.muatan = None
    return kirim


try:
    # RAGAM KOSONG sama dengan tidak disetel. Railway menyimpan ragam kosong
    # sebagai ragam yang ada, dan os.environ.get dengan bawaan akan
    # mengembalikan teks kosong itu, sehingga URL jadi "" dan urllib
    # melemparkan ValueError dalam milidetik tanpa menyentuh jaringan.
    os.environ["VLM_URL"] = ""
    assert _vlm._env("VLM_URL", "bawaan") == "bawaan"
    os.environ["VLM_URL"] = "   "
    assert _vlm._env("VLM_URL", "bawaan") == "bawaan"
    os.environ["VLM_URL"] = " https://contoh/v1 "
    assert _vlm._env("VLM_URL", "bawaan") == "https://contoh/v1"
    os.environ.pop("VLM_URL", None)
    assert _vlm._env("VLM_URL", "bawaan") == "bawaan"

    # BAWAAN MATI. Dua syarat, bukan satu: kunci yang tertinggal di
    # lingkungan tidak boleh membuat peladen memanggil jaringan.
    os.environ.pop("VLM", None)
    os.environ.pop("KUNCI_VLM", None)
    assert _vlm.aktif() is False
    os.environ["VLM"] = "hidup"
    assert _vlm.aktif() is False           # hidup tetapi tanpa kunci
    os.environ["KUNCI_VLM"] = "kunci-palsu-untuk-uji"
    assert _vlm.aktif() is True
    os.environ["VLM"] = "mati"
    assert _vlm.aktif() is False

    # mati: pengirim TIDAK pernah dipanggil
    _k = _balas("{}")
    assert _vlm.tanya(_CITRA, ["total"], kirim=_k) == {}
    assert _k.muatan is None

    os.environ["VLM"] = "hidup"

    # daftar medan kosong: tetap tidak menyentuh jaringan
    _k = _balas("{}")
    assert _vlm.tanya(_CITRA, [], kirim=_k) == {}
    assert _k.muatan is None

    # balasan terbungkus pagar markdown, walaupun dilarang di perintah
    _pagar = "```json" + chr(10) +         '{"total": "RM 93.90", "jam": "7:55:44 PM"}' + chr(10) + "```"
    _k = _balas(_pagar)
    _h = _vlm.tanya(_CITRA, ["total", "jam"], uang=_uang, kirim=_k)
    assert _h == {"total": 93.9, "jam": "7:55:44 PM"}, _h

    # medan gerbang berupa dict, bukan nama polos
    _h = _vlm.tanya(_CITRA, [{"medan": "total", "zona": "bawah"}],
                    uang=_uang, kirim=_balas('{"total": "93.90"}'))
    assert _h == {"total": 93.9}, _h

    # VLM TIDAK BISA menulis ke medan yang tidak ditanyakan
    _h = _vlm.tanya(_CITRA, ["total"], uang=_uang,
                    kirim=_balas('{"total": "10.00", "toko": "DIKARANG"}'))
    assert _h == {"total": 10.0}, _h

    # null, teks "null", dan teks kosong semuanya BUKAN jawaban
    _h = _vlm.tanya(_CITRA, ["total", "jam", "toko"], uang=_uang,
                    kirim=_balas('{"total": null, "jam": "null", "toko": "  "}'))
    assert _h == {}, _h

    # Jawaban uang yang bukan angka DIBUANG. Ini bukan uji sepele:
    # _uang("tidak terbaca") mengembalikan 1.0, sebab ia memetakan I jadi 1
    # lalu menyaring sisa hurufnya. Tanpa penjaga bentuk di vlm.py, kalimat
    # VLM yang jujur berubah menjadi RM 1,00 yang percaya diri. Uji ini
    # MENANGKAPNYA, dan itulah sebab penjaga itu ada.
    _h = _vlm.tanya(_CITRA, ["total"], uang=_uang,
                    kirim=_balas('{"total": "tidak terbaca"}'))
    assert _h == {}, _h

    # balasan yang bukan JSON sama sekali
    assert _vlm.tanya(_CITRA, ["total"], uang=_uang,
                      kirim=_balas("Maaf, saya tidak bisa membantu.")) == {}

    # GALAT TIDAK MENJATUHKAN STRUK, dan dihitung
    def _meledak(muatan):
        raise RuntimeError("jaringan mati")

    _g0 = _vlm.gagal()
    assert _vlm.tanya(_CITRA, ["total"], uang=_uang, kirim=_meledak) == {}
    assert _vlm.gagal() == _g0 + 1
    assert isinstance(_vlm.detik(), float)

    # muatan memuat medan yang diminta dan citra base64, bukan jalur berkas
    _k = _balas('{"total": "1.00"}')
    _vlm.tanya(_CITRA, ["total", "kembalian"], uang=_uang, kirim=_k)
    _teks = _k.muatan["messages"][0]["content"][0]["text"]
    assert '"total"' in _teks and '"kembalian"' in _teks
    assert '"jam"' not in _teks
    assert _k.muatan["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,")
    assert _k.muatan["temperature"] == 0
    # batas keluaran ada, supaya jawaban yang mengarang panjang tidak dibayar
    # per token sampai penyedianya berhenti sendiri
    assert _k.muatan["max_tokens"] == 300

    # BATAS HARIAN. Pagar ongkos, bukan pagar lalu lintas: kunci peladen
    # tertanam di aplikasi ponsel dan harus dianggap bisa bocor. Saat jatah
    # habis, VLM DIAM, bukan melempar galat, sehingga pemindaian tetap jalan
    # dan medannya tetap kosong seperti sebelum VLM ada.
    _simpan_batas, _vlm.BATAS_HARI = _vlm.BATAS_HARI, 2
    _vlm._hari, _vlm._pakai = "", 0
    _k = _balas('{"total": "1.00"}')
    assert _vlm.tanya(_CITRA, ["total"], uang=_uang, kirim=_k) == {"total": 1.0}
    assert _vlm.tanya(_CITRA, ["total"], uang=_uang, kirim=_k) == {"total": 1.0}
    assert _vlm.pakai_hari_ini() == 2
    assert _vlm.tanya(_CITRA, ["total"], uang=_uang, kirim=_k) == {}
    assert "jatah harian" in _vlm.galat(), _vlm.galat()
    # hari berganti, jatah pulih
    _vlm._hari = "2000-01-01"
    assert _vlm.pakai_hari_ini() == 0
    assert _vlm.tanya(_CITRA, ["total"], uang=_uang, kirim=_k) == {"total": 1.0}
    _vlm.BATAS_HARI = _simpan_batas

    # gerbang dan pemanggil bersambung: jawaban VLM hanya masuk ke medan yang
    # gerbangnya buka, dan medan berisi tetap tidak tertimpa
    _kon = _kontrak(total=16.75)
    _minta = _gv.perlu_tambal(_kon, _ATAS + _BAWAH, _TINGGI)
    _jwb = _vlm.tanya(_CITRA, _minta, uang=_uang,
                      kirim=_balas('{"total": "99.00", "kembalian": "0.50"}'))
    assert _jwb == {"kembalian": 0.5}, _jwb
    _tulis = _gv.tambal(_kon, _jwb, _ATAS + _BAWAH, _TINGGI)
    assert _tulis == ["kembalian"], _tulis
    assert _kon["ringkasan"]["total"]["nilai"] == 16.75
    assert _kon["ringkasan"]["kembalian"]["nilai"] == 0.5
    assert _kon["ringkasan"]["kembalian"]["asal"] == "vlm"
finally:
    for _n, _v in zip(("VLM", "KUNCI_VLM"), _simpan_vlm):
        if _v is None:
            os.environ.pop(_n, None)
        else:
            os.environ[_n] = _v

print("21. pemanggil VLM: lolos")


# ---------------------------------------------------------------------------
# 22. BAYARAN NONTUNAI TIDAK DIBACA SEBAGAI TUNAI
#
# Dasarnya dua struk ST Rosyam Mart yang BUKAN bagian uji3. Yang pertama
# dibayar "AMBANK TNGWALLET 61.20" dan sistem membacanya tunai 61,20, padahal
# jawaban benarnya emoney 61,20 dan tunai kosong. Yang kedua dibayar "CASH
# 12.00" dan harus TETAP tunai.
#
# Diukur sebelum dipasang: uji3 tetap 35 dari 66, tidak satu medan pun
# berubah, sedangkan struk pertama naik dari 3 jadi 5 dari 6 medan uang.
# ---------------------------------------------------------------------------
from rantai import _pindah_nontunai  # noqa: E402


def _baris_bayar(teks_kiri, nilai_teks):
    """Satu baris: kata alat bayar di kiri, angkanya di kanan."""
    kot = []
    x = 0
    for t in teks_kiri.split():
        kot.append({"teks": t, "bbox": [x, 100, x + 40, 120],
                    "_label": "O", "_yakin": 0.9})
        x += 45
    kot.append({"teks": nilai_teks, "bbox": [300, 100, 360, 120],
                "_label": "total.cashprice", "_yakin": 0.9})
    idb = [0] * len(kot)
    h = {"tunai": {"teks": nilai_teks, "nilai": 61.20, "yakin": 0.9,
                   "bbox": [300, 100, 360, 120]}}
    return kot, idb, h


# dompet elektronik: pindah ke emoney, tunai jadi kosong
_kot, _idb, _h = _baris_bayar("AMBANK TNGWALLET", "61.20")
_h = _pindah_nontunai(_kot, _idb, _h)
assert "tunai" not in _h, _h
assert _h["emoney"]["nilai"] == 61.20, _h

# DuitNow QR juga
_kot, _idb, _h = _baris_bayar("DUITNOW QR", "61.20")
assert _pindah_nontunai(_kot, _idb, _h).get("emoney", {}).get("nilai") == 61.20

# kartu pindah ke kartu, bukan emoney
_kot, _idb, _h = _baris_bayar("VISA CARD", "61.20")
_h = _pindah_nontunai(_kot, _idb, _h)
assert "tunai" not in _h and _h["kartu"]["nilai"] == 61.20, _h

# CASH TETAP TUNAI. Ini yang menjaga struk kedua tidak ikut rusak.
_kot, _idb, _h = _baris_bayar("CASH", "61.20")
_h = _pindah_nontunai(_kot, _idb, _h)
assert _h["tunai"]["nilai"] == 61.20 and "emoney" not in _h, _h

# medan tujuan yang SUDAH berisi tidak ditimpa, dan tunai dibiarkan
_kot, _idb, _h = _baris_bayar("AMBANK TNGWALLET", "61.20")
_h["emoney"] = {"teks": "9.00", "nilai": 9.00, "yakin": 0.9,
                "bbox": [300, 200, 360, 220]}
_h = _pindah_nontunai(_kot, _idb, _h)
assert _h["emoney"]["nilai"] == 9.00 and _h["tunai"]["nilai"] == 61.20, _h

# tunai kosong tidak diapa-apakan
assert _pindah_nontunai([], [], {}) == {}
assert _pindah_nontunai([], [], {"tunai": None}) == {"tunai": None}

print("22. bayaran nontunai: lolos")


# ---------------------------------------------------------------------------
# 23. KUANTITAS YANG DITAMPILKAN APLIKASI
#
# Dua kerusakan nyata di layar aplikasi pada struk ST Rosyam Mart: kuantitas
# 404, 503, 480 dan 991 yang mustahil, serta kuantitas 0 untuk barang yang
# dijual per kilogram. Keduanya berasal dari int() atas kuantitas pecahan.
# ---------------------------------------------------------------------------
from rantai import jumlah_tampil  # noqa: E402

assert jumlah_tampil(1) == 1
assert jumlah_tampil(1.0) == 1
assert jumlah_tampil(2.0) == 2
# PECAHAN DIPERTAHANKAN. 0,404 kg itu kuantitas yang sah, dan int() membuatnya
# tampil 0 di aplikasi.
assert jumlah_tampil(0.404) == 0.404
assert jumlah_tampil(1.022) == 1.022
assert jumlah_tampil(2.0964) == 2.096
# MUSTAHIL DIBUANG, bukan ditebak ulang
assert jumlah_tampil(404) is None
assert jumlah_tampil(503) is None
assert jumlah_tampil(991) is None
assert jumlah_tampil(100) == 100          # batasnya sendiri masih diterima
assert jumlah_tampil(100.5) is None
assert jumlah_tampil(0) is None
assert jumlah_tampil(-1) is None
assert jumlah_tampil(None) is None
assert jumlah_tampil("2") is None
assert jumlah_tampil(True) is None        # bool bukan kuantitas

# --- 23b. kuantitas dari kolom "satuan kali kuantitas" ---
# Seluruh angka di bawah ini diambil apa adanya dari struk ST Rosyam Mart,
# termasuk cara pengenal memecah kotaknya.
from rantai import _angka_bintang  # noqa: E402


def _kk(*pasang):
    """Kotak sebaris: (teks, x0, x1). Tinggi dibuat tetap 20 piksel."""
    return [{"teks": t, "bbox": [x0, 100, x1, 120]} for t, x0, x1 in pasang]


_AMB = 12  # 0,6 x 20

# "3.29*0.404" terpecah jadi "29*0." lalu "404", dan harga baris jauh di kanan
assert _angka_bintang(_kk(("29*0.", 100, 160), ("404", 165, 200),
                          ("1.33", 300, 340)), _AMB) == 0.404
# titik ganda di sambungan: "1." + ".022"
assert _angka_bintang(_kk((".99*1.", 100, 160), (".022", 165, 200),
                          ("7.14", 300, 340)), _AMB) == 1.022
# kuantitas bulat, dan harga baris TIDAK ikut tertelan sebab jaraknya jauh
assert _angka_bintang(_kk(("1.99*1", 100, 160), (".99", 300, 340)), _AMB) == 1.0
# "FRZ SOTONG RING*KG" bukan kolom kuantitas: tidak ada angka sebelum bintang
assert _angka_bintang(_kk(("RING*KG", 100, 180)), _AMB) is None
# "85GM *5'S" juga bukan: bintangnya didahului spasi
assert _angka_bintang(_kk(("*5", 100, 140)), _AMB) is None
assert _angka_bintang(_kk(("85GM", 100, 140), ("*5", 145, 170)), _AMB) is None
# hasil mustahil ditolak
assert _angka_bintang(_kk(("2*500", 100, 160)), _AMB) is None
# tanpa bintang
assert _angka_bintang(_kk(("2.30", 100, 140)), _AMB) is None
# bintang tanpa ekor angka
assert _angka_bintang(_kk(("5*", 100, 140), ("KG", 145, 170)), _AMB) is None

print("23. kuantitas tampil: lolos")

# ---------------------------------------------------------------------------
# 24. DELAPAN MEDAN YANG MENJADI DASAR ANGKA SKRIPSI
#
# Seluruh angka ketepatan skripsi dihitung atas delapan medan ini, yaitu 66
# medan dari 11 struk. Menambah atau mengurangi satu medan mengubah penyebut
# itu tanpa menimbulkan galat apa pun saat dijalankan, sehingga angka lama dan
# angka baru diam-diam tidak lagi sebanding. Karena itu daftarnya dipaku.
# ---------------------------------------------------------------------------
_DELAPAN = ("toko", "tanggal", "jam", "subtotal", "pajak", "total", "tunai",
            "kembalian")
assert tuple(_gv.MEDAN_ATAS + _gv.MEDAN_UANG) == _DELAPAN
# kartu dan emoney WAJIB di luar, sebab angka resmi diukur sebelum keduanya ada
assert "kartu" not in _DELAPAN and "emoney" not in _DELAPAN
assert len(set(_DELAPAN)) == 8

print("24. delapan medan dasar angka: lolos")


# ---------------------------------------------------------------------------
# 25. LENGAN BARIS BARANG VLM
#
# Yang dijaga di sini penguraian senarai JSON beserta penyaringannya, sebab
# kegagalan di bagian itu menghasilkan daftar kosong TANPA galat apa pun,
# sehingga baris barang hilang dari aplikasi dan tidak ada yang tahu
# sebabnya. Tidak ada jaringan di sini: pengirimnya dipalsukan.
# ---------------------------------------------------------------------------
os.environ["VLM"] = "hidup"
os.environ["KUNCI_VLM"] = "kunci-palsu-untuk-uji"
assert _vlm.aktif() is True

_ITEM_OK = ('[{"nama":"CHN CARROT","jumlah":"0.404","harga":"1.33"},'
            '{"nama":"INDOMIE","jumlah":null,"harga":"4.50"}]')
_h = _vlm.tanya_item(_CITRA, uang=float, kirim=_balas(_ITEM_OK))
assert [x["harga"] for x in _h] == [1.33, 4.50], _h
assert [x["jumlah"] for x in _h] == [0.404, None], _h
assert _h[0]["nama"] == "CHN CARROT"
# perintahnya harus perintah baris barang, bukan perintah delapan medan
_k = _balas(_ITEM_OK)
_vlm.tanya_item(_CITRA, uang=float, kirim=_k)
_teks_perintah = _k.muatan["messages"][0]["content"][0]["text"]
assert "JSON array" in _teks_perintah
assert "Do not include subtotal" in _teks_perintah
# batas token harus lebih besar, kalau tidak jawaban panjang terpotong
assert _k.muatan["max_tokens"] == _vlm.MAKS_TOKEN_ITEM > _vlm.MAKS_TOKEN

# dibungkus pagar markdown walaupun dilarang: tetap harus terurai
_h = _vlm.tanya_item(_CITRA, uang=float,
                     kirim=_balas("```json\n" + _ITEM_OK + "\n```"))
assert len(_h) == 2, _h

# harga bukan angka DIBUANG jadi None, barisnya tetap ada karena namanya ada
_h = _vlm.tanya_item(_CITRA, uang=float,
                     kirim=_balas('[{"nama":"X","harga":"tidak terbaca"}]'))
assert _h == [{"nama": "X", "jumlah": None, "harga": None}], _h
# baris tanpa nama DAN tanpa harga dibuang seluruhnya
_h = _vlm.tanya_item(
    _CITRA, uang=float,
    kirim=_balas('[{"nama":"","harga":null},{"nama":"Y","harga":"2.00"}]'))
assert _h == [{"nama": "Y", "jumlah": None, "harga": 2.0}], _h
# bukan senarai, senarai kosong, dan sampah: jatuh ke daftar kosong
assert _vlm.tanya_item(_CITRA, uang=float, kirim=_balas('{"nama":"X"}')) == []
assert _vlm.tanya_item(_CITRA, uang=float, kirim=_balas("[]")) == []
assert _vlm.tanya_item(_CITRA, uang=float, kirim=_balas("maaf tidak bisa")) == []
assert _vlm.tanya_item(_CITRA, uang=float, kirim=_balas('["bukan objek"]')) == []


def _meledak(muatan):
    raise OSError("jaringan mati")


# galat jaringan jatuh ke daftar kosong, bukan melempar
assert _vlm.tanya_item(_CITRA, uang=float, kirim=_meledak) == []
# BAWAAN MATI juga berlaku bagi lengan ini
os.environ.pop("KUNCI_VLM", None)
assert _vlm.tanya_item(_CITRA, uang=float, kirim=_balas(_ITEM_OK)) == []
for _n, _v in zip(("VLM", "KUNCI_VLM"), _simpan_vlm):
    if _v is None:
        os.environ.pop(_n, None)
    else:
        os.environ[_n] = _v

print("25. lengan baris barang VLM: lolos")


# ---------------------------------------------------------------------------
# 26. GERBANG BARIS BARANG, HANYA MENGISI YANG KOSONG
#
# Larangan menimpa adalah aturan rancangan proyek ini, dan bagian ini yang
# menegakkannya pada baris barang. Rancangan serah blok pernah dipasang pada
# commit 28daed2 lalu ditarik: ia membuang 24 harga yang sudah benar.
#
# Empat kekeliruan yang dijaga di sini, dan tidak satu pun menimbulkan galat
# saat dijalankan sehingga semuanya akan lolos tanpa uji ini:
#
#   1. Harga yang sudah ada tertimpa jawaban model bahasa visual.
#   2. Nama barang tertimpa, padahal hanya harganya yang boleh diisi.
#   3. Jumlah baris berubah, yaitu blok lama tergantikan.
#   4. Lubang yang namanya cocok dengan DUA baris tetap diisi. Yang rancu
#      harus dibiarkan kosong, sebab mengisinya berarti menebak.
# ---------------------------------------------------------------------------
def _it(*pasang):
    """Blok baris barang bergaya kontrak aplikasi: (nama, harga) per baris."""
    return {"item": [{"id": "i%d" % (n + 1), "nama": nm, "jumlah": 1.0,
                      "hargaSatuan": None, "harga": h, "keyakinan": 0.9}
                     for n, (nm, h) in enumerate(pasang)]}


def _v(*pasang):
    return [{"nama": nm, "jumlah": None, "harga": h} for nm, h in pasang]


# --- pemicunya ---
assert _gv.perlu_tambal_item({"item": []}) is False
assert _gv.perlu_tambal_item({}) is False
assert _gv.perlu_tambal_item(_it(("INDOMIE GORENG", 1.5))) is False
# harga 0 BUKAN kosong, aturan yang sama dengan medan ringkasan
assert _gv.perlu_tambal_item(_it(("INDOMIE GORENG", 0))) is False
assert _gv.perlu_tambal_item(_it(("INDOMIE GORENG", None))) is True
assert _gv.perlu_tambal_item(_it(("A", 1.5), ("INDOMIE GORENG", None))) is True

# --- pencocokan namanya ---
# imbuhan harga satuan yang ditempelkan VLM dibuang lebih dahulu
assert _gv.cocok_nama_item("CHN CARROT (KG)", "CHN CARROT @ 3.5000")
assert _gv.cocok_nama_item("kopi susu", "KOPI SUSU BESAR")
# nama pendek TIDAK boleh cocok, kalau tidak setiap lubang terisi keliru
assert not _gv.cocok_nama_item("CHN", "CHN CARROT (KG)")
assert not _gv.cocok_nama_item("", "APA SAJA")
assert not _gv.cocok_nama_item("ROTI", "ROTI MANIS")
# ambangnya dapat disetel, dan nilai bawaannya 6
assert _gv.MIN_NAMA_ITEM == 6
assert _gv.cocok_nama_item("ROTI", "ROTI MANIS", 4)

# --- pengisian yang benar ---
_k = _it(("INDOMIE GORENG", None), ("CHN CARROT (KG)", 2.5))
_t = _gv.tambal_item(_k, _v(("INDOMIE GORENG 85G", 4.6),
                            ("CHN CARROT @ 3.5000", 9.99)))
assert _t == ["i1"], _t
assert _k["item"][0]["harga"] == 4.6
assert _k["item"][0]["asal"] == "vlm"
assert _k["item"][0]["keyakinan"] is None
assert _k["itemAsalVlm"] == ["i1"]
# BARIS KEDUA TIDAK DISENTUH walaupun jawabannya memuat harga lain
assert _k["item"][1]["harga"] == 2.5, _k["item"][1]
assert "asal" not in _k["item"][1]
assert _k["item"][1]["keyakinan"] == 0.9
# nama tidak pernah berubah, hanya harganya yang diisi
assert _k["item"][0]["nama"] == "INDOMIE GORENG"
assert _k["item"][1]["nama"] == "CHN CARROT (KG)"
# jumlah baris tidak pernah berubah
assert len(_k["item"]) == 2
assert [x["id"] for x in _k["item"]] == ["i1", "i2"]

# --- yang rancu dibiarkan kosong ---
_k = _it(("KOPI SUSU", None))
assert _gv.tambal_item(_k, _v(("KOPI SUSU BESAR", 3.0),
                              ("KOPI SUSU KECIL", 2.0))) == []
assert _k["item"][0]["harga"] is None
assert "itemAsalVlm" not in _k
# nama yang tidak dikenali juga dibiarkan kosong
_k = _it(("BARANG ANEH SEKALI", None))
assert _gv.tambal_item(_k, _v(("INDOMIE GORENG", 4.6))) == []
assert _k["item"][0]["harga"] is None

# --- jawaban yang tidak dapat dipakai ---
for _kosong in ([], None):
    _k = _it(("INDOMIE GORENG", None))
    assert _gv.tambal_item(_k, _kosong) == []
    assert _k["item"][0]["harga"] is None
# baris VLM tanpa harga tidak dapat mengisi apa pun
_k = _it(("INDOMIE GORENG", None))
assert _gv.tambal_item(_k, _v(("INDOMIE GORENG 85G", None))) == []
# baris VLM tanpa nama tidak dipakai, sebab namanya yang menjadi buktinya
_k = _it(("INDOMIE GORENG", None))
assert _gv.tambal_item(_k, _v(("", 4.6))) == []
# bukan objek: diabaikan tanpa melempar
_k = _it(("INDOMIE GORENG", None))
assert _gv.tambal_item(_k, ["bukan objek", 7]) == []
# kontrak tanpa baris barang: tidak ada yang dapat diisi
assert _gv.tambal_item({"item": []}, _v(("A B C D E F", 1.0))) == []
assert _gv.tambal_item({}, _v(("A B C D E F", 1.0))) == []

# --- dua lubang sekaligus, dan penandanya terkumpul ---
_k = _it(("INDOMIE GORENG", None), ("CHN CARROT (KG)", None))
_t = _gv.tambal_item(_k, _v(("INDOMIE GORENG 85G", 4.6),
                            ("CHN CARROT @ 3.5000", 3.43)))
assert _t == ["i1", "i2"], _t
assert [x["harga"] for x in _k["item"]] == [4.6, 3.43]
assert _k["itemAsalVlm"] == ["i1", "i2"]

# --- gerbang baris barang TIDAK ikut campur ke medan ringkasan ---
_k = _it(("INDOMIE GORENG", None))
_k["toko"] = "ADA"
_k["ringkasan"] = {}
assert _gv.tambal_item(_k, _v(("INDOMIE GORENG 85G", 4.6))) == ["i1"]
assert _k["toko"] == "ADA"
assert "asalVlm" not in _k

print("26. gerbang baris barang hanya mengisi: lolos")


# ---------------------------------------------------------------------------
# 27. APA YANG KELUAR KE PIHAK KETIGA, UNTUK NFR2
#
# Bagian ini menjaga satu janji privasi yang dapat hilang TANPA SUARA. Citra
# yang dikirim ke penyedia model bahasa visual dibangun ulang sebagai JPEG
# oleh PIL, dan PIL membuang EXIF kecuali diminta mempertahankannya. Jadi
# penanda kamera, waktu, dan koordinat GPS tidak ikut terkirim. Cukup satu
# orang menambahkan exif=... pada panggilan save, dan koordinat GPS struk
# orang lain mulai mengalir ke pihak ketiga tanpa satu pun galat.
#
# Yang dijaga di sini juga muatannya: tidak ada identitas pengguna, nomor
# struk, nama berkas, maupun teks hasil pengenalan yang keluar. Hanya nama
# model, perintah, dan citranya.
# ---------------------------------------------------------------------------
os.environ["VLM"] = "hidup"
os.environ["KUNCI_VLM"] = "kunci-palsu-untuk-uji"

import base64 as _b64  # noqa: E402
import io  # noqa: E402

_ex = _Img.Exif()
_ex[271] = "ACME"                       # Make
_ex[272] = "Telepon Uji"                # Model
_ex[306] = "2026:11:22 11:50:00"        # DateTime
_ex[37510] = "catatan pribadi"          # UserComment
_buf = io.BytesIO()
_Img.new("RGB", (2400, 1000), (250, 250, 250)).save(_buf, format="JPEG",
                                                    exif=_ex)
_masuk = _Img.open(io.BytesIO(_buf.getvalue()))
assert len(_masuk.getexif()) == 4, len(_masuk.getexif())

_bita = _b64.b64decode(_vlm._citra_base64(_masuk))
_keluar = _Img.open(io.BytesIO(_bita))
# seluruh EXIF hilang, termasuk penanda kamera dan waktu
assert len(_keluar.getexif()) == 0, len(_keluar.getexif())
assert _keluar.getexif().get(271) is None
assert _keluar.getexif().get(306) is None
# bukan hanya tak terbaca PIL: teksnya memang tidak ada di dalam bitanya
assert b"ACME" not in _bita
assert b"catatan pribadi" not in _bita
assert b"Telepon Uji" not in _bita
# sisi terpanjang diperkecil ke SISI_MAKS, jadi yang keluar bukan citra penuh
assert max(_keluar.size) == _vlm.SISI_MAKS, _keluar.size

# muatan yang keluar TIDAK memuat apa pun selain model, perintah, dan citra
_k = _balas('{"toko":"A"}')
_vlm.tanya(_CITRA, [{"medan": "toko", "zona": "atas", "pita": (0.0, 0.45)}],
           kirim=_k)
assert sorted(_k.muatan) == ["max_tokens", "messages", "model", "temperature"], \
    sorted(_k.muatan)
assert len(_k.muatan["messages"]) == 1
_jenis = [c["type"] for c in _k.muatan["messages"][0]["content"]]
assert _jenis == ["text", "image_url"], _jenis
# citranya JPEG sebaris, bukan URL yang dapat diambil pihak lain
assert _k.muatan["messages"][0]["content"][1]["image_url"]["url"].startswith(
    "data:image/jpeg;base64,")
# perintahnya tidak memuat teks hasil pengenalan maupun koordinat kotak
_p = _k.muatan["messages"][0]["content"][0]["text"]
assert "bbox" not in _p and "koordinat" not in _p

for _n, _v in zip(("VLM", "KUNCI_VLM"), _simpan_vlm):
    if _v is None:
        os.environ.pop(_n, None)
    else:
        os.environ[_n] = _v

print("27. yang keluar ke pihak ketiga: lolos")


# --- 27b. orientasi EXIF dan pagar keamanan app.py ---
import ast as _ast
from PIL import ImageOps as _ImageOps

with open("app.py", "r", encoding="utf-8") as _f_app:
    _kode_app = _f_app.read()
_ast.parse(_kode_app)
assert "Image.MAX_IMAGE_PIXELS = 25_000_000" in _kode_app
assert "_ANTREAN_MAKS = 3" in _kode_app
assert "_SEMAPHORE_KONKURENSI.release()" in _kode_app

_ex_miring = _Img.Exif()
_ex_miring[0x0112] = 6  # Rotasi 90 derajat CW
_buf_miring = io.BytesIO()
_Img.new("RGB", (100, 200), (255, 255, 255)).save(_buf_miring, format="JPEG", exif=_ex_miring)
with _Img.open(io.BytesIO(_buf_miring.getvalue())) as _im_m:
    assert _im_m.getexif().get(0x0112, 1) == 6
    _im_tegak = _ImageOps.exif_transpose(_im_m)
    assert _im_tegak.size == (200, 100), _im_tegak.size

print("27b. orientasi EXIF dan pagar app.py: lolos")


# --- 28. pembaca DBNet: pecah_baris, kode pajak, siapkan_citra, adaptor kotak ---
import pembaca_dbnet as _PD  # noqa: E402

# Cadangan proporsional (tanpa kotak huruf): lebar dibagi menurut posisi huruf.
# "AB 1.80SR" = 9 huruf pada lebar 90, jadi 10 piksel per huruf.
_s = _PD.pecah_baris([{"teks": "AB 1.80SR", "bbox": [0, 0, 90, 10], "yakin": 0.8}])
assert [b["teks"] for b in _s] == ["AB", "1.80", "SR"], _s
assert [b["bbox"] for b in _s] == [[0, 0, 20, 10], [30, 0, 70, 10], [70, 0, 90, 10]], _s
assert all(b["yakin"] == 0.8 for b in _s)

# Kode pajak Malaysia menempel pada harga dua desimal: dipisah.
for _t, _harga, _kode in [("1.80SR", "1.80", "SR"), ("3.71ZRL", "3.71", "ZRL"),
                          ("1,234.50SR", "1,234.50", "SR"), ("12,50SR", "12,50", "SR")]:
    _s = _PD.pecah_baris([{"teks": _t, "bbox": [0, 0, 10 * len(_t), 10]}])
    assert [b["teks"] for b in _s] == [_harga, _kode], (_t, _s)
# Kurung penutup ikut dibuang pada jalur proporsional.
assert [b["teks"] for b in _PD.pecah_baris([{"teks": "(1.80SR)", "bbox": [0, 0, 80, 10]}])] == ["(1.80", "SR"]
# Yang BUKAN harga dua desimal tidak boleh dipecah: SKU, satu desimal, tiga huruf lebih.
for _t in ["AB12SR", "12.5SR", "1.80SRXY", "SR", "1.80", "TOTAL"]:
    _s = _PD.pecah_baris([{"teks": _t, "bbox": [0, 0, 60, 10]}])
    assert [b["teks"] for b in _s] == [_t], (_t, _s)
# Teks kosong atau None tidak menghasilkan kata.
assert _PD.pecah_baris([{"teks": "", "bbox": [0, 0, 10, 10]}, {"teks": None, "bbox": [0, 0, 10, 10]}]) == []


def _huruf(teks, x0, lebar=10, tinggi=20, celah=None):
    """Kotak per huruf; `celah` = {indeks: piksel ekstra sebelum huruf itu}."""
    out, x = [], x0
    for i, ch in enumerate(teks):
        x += (celah or {}).get(i, 0)
        out.append((ch, [x, 0, x + lebar, tinggi]))
        x += lebar
    return out


# Jalur kotak huruf: kata dipecah pada spasi dan pada celah >= 0,25 tinggi (5 piksel).
_h = _huruf("AB", 0) + _huruf("CD", 40)                      # celah 20 px di antara B dan C
_s = _PD.pecah_baris([{"teks": "ABCD", "bbox": [0, 0, 60, 20], "yakin": 0.9, "huruf": _h}])
assert [b["teks"] for b in _s] == ["AB", "CD"] and _s[0]["bbox"] == [0, 0, 20, 20] and _s[1]["bbox"] == [40, 0, 60, 20], _s
_h = _huruf("AB", 0) + [(" ", [20, 0, 25, 20])] + _huruf("CD", 25)  # spasi eksplisit
assert [b["teks"] for b in _PD.pecah_baris([{"teks": "AB CD", "bbox": [0, 0, 45, 20], "huruf": _h}])] == ["AB", "CD"]
_h = _huruf("AB", 0) + _huruf("CD", 22)                      # celah 2 px < 5 px: satu kata
assert [b["teks"] for b in _PD.pecah_baris([{"teks": "ABCD", "bbox": [0, 0, 42, 20], "huruf": _h}])] == ["ABCD"]
# Kode pajak pada jalur kotak huruf: koordinat kata = gabungan huruf, bukan hasil bagi rata.
_h = _huruf("1.80SR", 100)
_s = _PD.pecah_baris([{"teks": "1.80SR", "bbox": [100, 0, 160, 20], "huruf": _h}])
assert [b["teks"] for b in _s] == ["1.80", "SR"], _s
assert _s[0]["bbox"] == [100, 0, 140, 20] and _s[1]["bbox"] == [140, 0, 160, 20], _s
_h = _huruf("3.71ZRL", 0)
assert [b["teks"] for b in _PD.pecah_baris([{"teks": "3.71ZRL", "bbox": [0, 0, 70, 20], "huruf": _h}])] == ["3.71", "ZRL"]

# siapkan_citra: orientasi EXIF diluruskan, sisi terpanjang <= 1600, keluaran RGB.
from PIL import Image as _ImgD  # noqa: E402
_ex6 = _ImgD.Exif()
_ex6[0x0112] = 6
_b6 = io.BytesIO()
_ImgD.new("RGB", (100, 200), "white").save(_b6, format="JPEG", exif=_ex6)
_b6.seek(0)
_c = _PD.siapkan_citra(_b6)
assert _c.size == (200, 100) and _c.mode == "RGB", _c.size
_bb = io.BytesIO()
_ImgD.new("RGB", (3200, 1000), "white").save(_bb, format="JPEG")
_bb.seek(0)
assert _PD.siapkan_citra(_bb).size == (1600, 500)
_bk = io.BytesIO()
_ImgD.new("L", (300, 400), 128).save(_bk, format="PNG")
_bk.seek(0)
assert _PD.siapkan_citra(_bk).mode == "RGB"


class _MesinPalsu:
    """Meniru keluaran rapidocr: [quad, teks, skor, kotak_huruf, huruf]; mencatat bentuk masukannya."""

    def __init__(self, hasil):
        self.hasil, self.masukan = hasil, None

    def __call__(self, bgr, **kw):
        self.masukan = (bgr, kw)
        return self.hasil, None


def _quad(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


_teks = "TOTAL 16,75SR"
_hq = [_quad(i * 10, 0, i * 10 + 10, 20) for i in range(len(_teks))]
_m = _MesinPalsu([[_quad(0, 0, 130, 20), _teks, 0.9, _hq, list(_teks)],
                  [_quad(0, 30, 50, 50), "品", 0.5, None, None],     # hanya huruf Han: dibuang
                  [_quad(0, 60, 40, 80), "RM1", 0.7, None, None]])        # tanpa kotak huruf: jalur cadangan
_cit = _ImgD.new("RGB", (400, 100), (10, 20, 30))
_k = _PD.kotak_dbnet(_cit, normalisasi_angka, mesin=_m)
assert [b["teks"] for b in _k] == ["TOTAL", "16.75", "SR", "RM1"], _k      # koma desimal dinormalisasi
assert [b["teks_asli"] for b in _k][1] == "16,75"
assert _m.masukan[1] == dict(use_det=True, use_cls=False, use_rec=True, return_word_box=True)
assert tuple(_m.masukan[0][0, 0]) == (30, 20, 10), _m.masukan[0][0, 0]       # BGR, bukan RGB
assert all(0 <= b["bbox"][0] < b["bbox"][2] <= 400 and 0 <= b["bbox"][1] < b["bbox"][3] <= 100 for b in _k)
assert [b["bbox"][1] for b in _k] == sorted(b["bbox"][1] for b in _k)         # urut atas ke bawah

print("28. pembaca DBNet, pecah_baris, kode pajak: lolos")


# --- 29. app.py: tanpa EasyOCR saat startup, inferensi dan VLM tidak memblokir event loop ---
_pohon = _ast.parse(_kode_app)
_ekstrak = next(n for n in _pohon.body if isinstance(n, _ast.AsyncFunctionDef) and n.name == "ekstrak")


def _panggilan(simpul):
    for n in _ast.walk(simpul):
        if isinstance(n, _ast.Call):
            yield n


def _nama_atribut(c):
    return c.func.attr if isinstance(c.func, _ast.Attribute) else getattr(c.func, "id", "")


# 1. Setiap `await ...run_in_executor(None, fungsi)` memanggil fungsi bersarang, bukan hasil panggilannya.
_menunggu = [n for n in _ast.walk(_ekstrak) if isinstance(n, _ast.Await)
             and isinstance(n.value, _ast.Call) and _nama_atribut(n.value) == "run_in_executor"]
assert len(_menunggu) == 2, len(_menunggu)           # inferensi dan VLM
_fungsi_bersarang = {n.name for n in _ast.walk(_ekstrak) if isinstance(n, _ast.FunctionDef)}
assert {a.value.args[1].id for a in _menunggu} == {"_eksekusi_inferensi", "_eksekusi_vlm"}
assert all(a.value.args[0].value is None for a in _menunggu)
assert {"_eksekusi_inferensi", "_eksekusi_vlm"} <= _fungsi_bersarang

# 2. Kerja berat (.result(), .submit(), jalankan(), ThreadPoolExecutor) hanya di dalam fungsi bersarang
#    yang dijalankan utas pekerja, tidak pernah langsung di badan coroutine.
_berat = {"result", "submit", "jalankan", "ThreadPoolExecutor", "tanya", "tanya_item", "sleep"}
_di_coroutine = []
for _n in _ekstrak.body:
    for _stmt in [_n]:
        _lewat = [x for x in _ast.walk(_stmt) if isinstance(x, _ast.FunctionDef)]
        _dalam = {id(y) for f in _lewat for y in _ast.walk(f)}
        for _c in _panggilan(_stmt):
            if id(_c) not in _dalam and _nama_atribut(_c) in _berat:
                _di_coroutine.append(_nama_atribut(_c))
assert not _di_coroutine, "memblokir event loop: %s" % _di_coroutine
assert "asyncio.to_thread" not in _kode_app

# 3. Perilaku polanya: fungsi pemblokir lewat run_in_executor tidak menahan event loop.
import asyncio as _asyncio  # noqa: E402
import time as _waktu  # noqa: E402


async def _uji_loop():
    _denyut = []

    async def _pendetak():
        for _ in range(8):
            _denyut.append(_waktu.perf_counter())
            await _asyncio.sleep(0.02)

    _t = _asyncio.ensure_future(_pendetak())
    await _asyncio.get_running_loop().run_in_executor(None, _waktu.sleep, 0.2)
    await _t
    return _denyut


_d = _asyncio.run(_uji_loop())
assert len(_d) == 8 and max(b - a for a, b in zip(_d, _d[1:])) < 0.1, "event loop terblokir"

# 4. Startup produksi tidak memuat EasyOCR/CRAFT kecuali PEMBACA=craft (jalur Model v1).
_impor_atas = [n for n in _pohon.body if isinstance(n, (_ast.Import, _ast.ImportFrom))]
assert not any(a.name.split(".")[0] == "easyocr" for n in _impor_atas if isinstance(n, _ast.Import) for a in n.names)
_blok_craft = next(n for n in _pohon.body if isinstance(n, _ast.If) and "PEMBACA" in _ast.dump(n.test)
                   and any(isinstance(m, _ast.Import) and m.names[0].name == "easyocr" for m in n.body))
assert any(isinstance(m, _ast.Import) and m.names[0].name == "easyocr" for m in _blok_craft.body)
assert not any(isinstance(m, _ast.Import) and m.names[0].name == "easyocr" for m in _blok_craft.orelse)
assert 'os.environ.get("MODEL_ID", "Klissh/layoutlmv3-struk-summa-v2-dbnet")' in _kode_app
assert 'os.environ.get("PEMBACA", "dbnet")' in _kode_app

print("29. app.py: tanpa EasyOCR saat startup, event loop tidak terblokir: lolos")

print("semua uji lolos")
