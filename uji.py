"""Uji mandiri untuk rantai.py. Jalankan: python uji.py

Sengaja tidak memuat model, supaya bisa dijalankan di laptop tanpa torch.
Yang diuji adalah bagian yang paling mudah rusak diam-diam: normalisasi teks,
urutan baca, dan bentuk JSON yang dikirim ke aplikasi.
"""

from rantai import (
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
assert set(app["ringkasan"]) == {"subtotal", "pajak", "total", "tunai", "kembalian"}
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

print("semua uji lolos")
