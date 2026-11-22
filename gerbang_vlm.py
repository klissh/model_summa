"""Gerbang VLM versi aman: siapa yang boleh diserahkan ke model bahasa visual.

Modul ini TIDAK memanggil VLM apa pun. Ia hanya memutuskan medan mana yang
boleh ditanyakan, dan menulis jawabannya tanpa pernah menimpa. Pemisahan itu
sengaja: keputusannya bisa diukur pada data yang sudah ada tanpa kuota, kunci
API, maupun mengirim foto orang lain ke pihak ketiga.

Mengapa gerbangnya sesempit ini, semuanya terukur pada 11 struk uji3 versi E:

  Usulan lima kondisi (kosong, nol, token nol, persamaan tak seimbang, jam di
  luar rentang) menandai 22 medan yang memang salah, TETAPI juga 12 dari 39
  medan yang sudah benar, yaitu 31 persen. Gerbang yang menyerahkan jawaban
  benar ke VLM menukar kepastian dengan kemungkinan, dan itu merugi.

  Tiga sebab terbesarnya:
    - nilai 0 dianggap kosong, padahal 4 medan di jawaban benar memang nol
    - blok persamaan subtotal + pajak = total menandai subtotal yang sudah
      benar, sebab struk Malaysia memuat service charge tanpa kolom sendiri
      (uji3_01 selisih 8,06 dan uji3_04 selisih 21,78)
    - uji kewajaran jam menuntut tepat HH:MM, padahal 9 dari 11 jawaban benar
      berbentuk 19:55:44, 01:28:33PM, atau 1:20:59 PM

Jadi di sini pemicunya SATU: nilainya belum ada sama sekali. Dengan itu, medan
benar yang dipertaruhkan nol, dan batas atas yang bisa dicapai 49 dari 66 atau
74,2 persen, naik dari 39 dari 66 atau 59,1 persen.

Penjaganya juga satu, dan objektif: harus ada kotak OCR di zona medan itu.
Kalau pengenal tidak menemukan teks apa pun di sana, tidak ada yang bisa
diperiksa terhadap jawaban VLM, jadi menanyakannya hanya mengundang karangan.
Penjaga ini TIDAK memakai laporan VLM tentang dirinya sendiri, sebab penilaian
diri bukan bukti.
"""

import re as _re

# Medan teks di kepala struk, dan medan uang di kakinya. Pembagian ini yang
# menentukan zona mana yang harus berisi kotak OCR.
MEDAN_ATAS = ("toko", "tanggal", "jam")
MEDAN_UANG = ("subtotal", "pajak", "total", "tunai", "kembalian")

# Satu struk dibayar dengan SATU cara. Kalau kartu atau dompet digital sudah
# berisi, `tunai` yang kosong itu benar kosong, bukan medan yang terlewat.
#
# Terlihat pada struk ST Rosyam Mart di layar aplikasi: model mengenali
# AMBANK TNGWALLET sebagai emoney 61,20, lalu gerbang tetap menanyakan tunai,
# lalu VLM membaca baris yang sama dan menjawab 61,20. Akibatnya aplikasi
# menampilkan pembayaran 61,20 DUA KALI, yaitu tunai dan dompet digital
# sekaligus, untuk belanja yang totalnya 61,20.
_PEMBAYARAN_LAIN = {"tunai": ("kartu", "emoney")}

# Pita zona sebagai bagian dari TINGGI CITRA, bukan dari rentang tegak kotak
# yang ditemukan. Bedanya menentukan: kalau pengenal hanya menemukan teks di
# kepala struk, rentang kotaknya pun hanya kepala, sehingga zona bawah ikut
# menyusut ke kepala dan penjaganya kehilangan gunanya. Keduanya bertumpang
# tindih sebab panjang struk sangat beragam.
ZONA = {"atas": (0.00, 0.45), "bawah": (0.40, 1.00)}

# shortcut: 2 kotak, yaitu label dan nilainya, belum disetel pada data mana
# pun. Naikkan kalau terukur ada zona berisi 2 kotak yang tetap tidak memuat
# medan yang dicari.
KOTAK_MIN = 2

# Pita tarif pajak Malaysia, satu-satunya pemeriksaan angka atas jawaban VLM.
#
# Alasannya terukur. Dua kesalahan VLM pada 11 struk uji3 keduanya di medan
# pajak, dan keduanya mengambil angka dari baris lain: uji3_05 mengambil
# "Others: 9.20", yang sebenarnya sisa bayaran sebab Total 57,80 = Mykasih
# 48,60 + Others 9,20, dan uji3_11 mengambil "Rounding Adj. RM0.02" padahal
# pajaknya "Service Tax@6% RM0.48". Memperketat perintah TIDAK menolong
# sedikit pun, terukur pada commit 29239be: kedua nilai itu keluar identik,
# sementara dua jawaban yang benar justru hilang.
#
# Pitanya dilebarkan supaya sah dipakai terhadap subtotal MAUPUN terhadap
# total, sebab subtotal sering belum ada saat pemeriksaan: 6 persen atas
# subtotal sama dengan 5,66 persen atas total, dan 8 persen sama dengan 7,41
# persen. Dasarnya rantai.py yang memakai pita sempit 6 dan 8 persen, dan di
# sana keenam nilai pajak uji3 terukur berada di 5,98 sampai 6,01 persen.
#
# shortcut: tarif Malaysia saja, yaitu 6 dan 8 persen. Struk dari negara lain
# atau pajak yang dibulatkan akan ditolak. Naikkan jadi daftar tarif per
# negara kalau aplikasi dipakai di luar Malaysia.
_PITA_PAJAK = ((0.0545, 0.0625), (0.0715, 0.0825))


def _kosong(nilai):
    """Hanya None dan teks kosong yang dianggap belum ada.

    Nilai 0 BUKAN kosong. Empat medan di jawaban benar uji3 memang nol, dan
    menganggapnya kosong menyerahkan medan yang sudah benar ke VLM.
    """
    if nilai is None:
        return True
    if isinstance(nilai, str):
        return nilai.strip() == ""
    return False


def _nilai(kontrak, medan):
    if medan in MEDAN_ATAS:
        return kontrak.get(medan)
    r = (kontrak.get("ringkasan") or {}).get(medan)
    return r.get("nilai") if isinstance(r, dict) else None


def _cukup_kotak(kotak, tinggi, zona):
    if not tinggi or tinggi <= 0:
        return False
    a, b = ZONA[zona]
    y0, y1 = a * tinggi, b * tinggi
    n = 0
    for k in kotak:
        bb = k.get("bbox")
        if not bb:
            continue
        tengah = (bb[1] + bb[3]) / 2.0
        if y0 <= tengah <= y1:
            n += 1
            if n >= KOTAK_MIN:
                return True
    return False


def _wajar_pajak(nilai, dasar):
    """False hanya kalau ADA dasar dan nisbahnya di luar pita tarif.

    Tanpa dasar, jawabannya diloloskan: tidak ada bukti untuk menolak, dan
    menolak tanpa bukti itu persis kesalahan yang membuat veto matematika
    usulan sebelumnya membuang jawaban benar.
    """
    if not isinstance(nilai, (int, float)) or not dasar or dasar <= 0:
        return True
    r = float(nilai) / float(dasar)
    return any(a <= r <= b for a, b in _PITA_PAJAK)


def _dasar_pajak(kontrak, jawaban):
    """Subtotal kalau ada, kalau tidak total. Jawaban VLM ikut diperhitungkan
    supaya hasilnya tidak bergantung pada urutan penulisan."""
    for m in ("subtotal", "total"):
        for v in (jawaban.get(m), _nilai(kontrak, m)):
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    return None


def perlu_tambal(kontrak, kotak, tinggi):
    """Daftar medan yang boleh ditanyakan ke VLM, beserta zona pemotongnya.

    Urutannya tetap, yaitu kepala dahulu lalu kaki, supaya keluarannya bisa
    dibandingkan antarjalan.
    """
    keluar = []
    for medan in MEDAN_ATAS + MEDAN_UANG:
        if not _kosong(_nilai(kontrak, medan)):
            continue
        if any(not _kosong(_nilai(kontrak, lain))
               for lain in _PEMBAYARAN_LAIN.get(medan, ())):
            continue
        zona = "atas" if medan in MEDAN_ATAS else "bawah"
        if not _cukup_kotak(kotak, tinggi, zona):
            continue
        keluar.append({"medan": medan, "zona": zona, "pita": ZONA[zona]})
    return keluar


def tambal(kontrak, jawaban, kotak, tinggi):
    """Tulis jawaban VLM, dan HANYA ke medan yang gerbangnya meloloskan.

    Larangan menimpa ditegakkan di sini, bukan hanya di komentar: medan yang
    sudah berisi tidak akan berubah walaupun jawaban memuatnya. Setiap medan
    yang tertulis ditandai asal `vlm`, supaya antarmuka bisa memberi tahu
    pengguna bahwa angka itu bukan dari pengenal huruf.

    Mengubah `kontrak` di tempat, dan mengembalikan daftar nama medan yang
    benar-benar tertulis.
    """
    if not jawaban:
        return []
    izin = {m["medan"] for m in perlu_tambal(kontrak, kotak, tinggi)}
    dasar = _dasar_pajak(kontrak, jawaban)
    tertulis = []
    for medan, nilai in jawaban.items():
        if medan not in izin or _kosong(nilai):
            continue
        if medan == "pajak" and not _wajar_pajak(nilai, dasar):
            continue
        if medan in MEDAN_ATAS:
            kontrak[medan] = nilai
        else:
            r = (kontrak.get("ringkasan") or {}).get(medan)
            if not isinstance(r, dict):
                continue
            r["nilai"] = nilai
            r["asal"] = "vlm"
            r["keyakinan"] = None
        tertulis.append(medan)
    # toko, tanggal dan jam di kontrak berupa teks polos tanpa medan `asal`,
    # jadi penandanya ditaruh sebagai satu daftar di akar. Antarmuka memakai
    # daftar ini, bukan hanya `asal` per medan ringkasan.
    if tertulis:
        kontrak["asalVlm"] = sorted(set(kontrak.get("asalVlm") or []) | set(tertulis))
    return tertulis


# ---------------------------------------------------------------------------
# BARIS BARANG. Aturannya SAMA dengan medan ringkasan di atas: hanya mengisi
# yang kosong, dan tidak pernah menimpa yang sudah ada.
#
# Rancangan serah blok, yang mengganti seluruh daftar baris barang ketika satu
# baris saja berlubang, pernah dipasang pada commit 28daed2 lalu DITARIK. Ia
# mencapai 96,3 persen harga benar, tetapi ia membuang 55 harga keluaran model
# pada 14 dari 16 struk, dan 24 di antaranya sudah benar. Larangan menimpa
# adalah aturan rancangan proyek ini, bukan sekadar tuning, jadi angka 96,3
# persen tidak cukup untuk membatalkannya.
#
# Empat rancangan yang hanya MENGISI diukur atas 107 harga pada 16 struk
# beranotasi, tiga lari, pada dataset/uji3/ukur_isi_kosong.py:
#
#   sistem saja, tanpa VLM             35, yaitu 32,7 persen
#   D1 sejajar dengan jangkar           35, yaitu 32,7 persen, nol lubang terisi
#   D0 sejajar tanpa jangkar            36, yaitu 33,6 persen
#   A COCOK NAMA, yang dipasang di sini 47, yaitu 43,9 persen
#
# Rancangan D menganggap urutan baris sistem sejajar dengan urutan baris VLM
# bila jumlah barisnya sama, lalu MEMERIKSA keandaian itu memakai baris yang
# harganya sudah ada. Gagasannya masuk akal dan hasilnya nol: kesejajarannya
# hanya terbukti pada 2 dari 16 struk, sebab jumlah baris sistem jarang sama
# dengan jumlah baris VLM. Dicatat di sini supaya tidak diusulkan lagi.
#
# Yang dipasang karena itu A, yaitu lubang diisi hanya bila namanya cocok
# dengan TEPAT SATU baris pada jawaban VLM. Pencocokan nama adalah buktinya,
# dan bukti itulah yang membedakannya dari rancangan sisip sisa yang ditolak
# pada 22 November: sisip sisa memasukkan harga tanpa bukti nama sama sekali,
# sehingga ia memproduksi harga benar yang menempel pada barang yang keliru.
#
# Harga yang dibayar A, dinyatakan terbuka: dari 14 lubang yang terisi, 12
# benar dan 2 SALAH. Nisbahnya 6 banding 1 dan itu menguntungkan, tetapi dua
# harga salah itu nyata dan pengguna yang harus menangkapnya di layar koreksi.
#
# Ambang panjang nama tidak disetel pada himpunan uji. Pemeriksaan kepekaan
# memberi hasil yang persis sama untuk ambang 5, 6, 7, dan 8, yaitu 47 harga
# benar dengan 14 isian dan 2 isian salah. Ambang 4 menambah satu isian yang
# salah, dan ambang 10 kehilangan satu isian yang benar. Jadi nilai 6 berada
# di tengah dataran, bukan di puncak yang dicari-cari.
#
# NAMA BARANG TIDAK PERNAH DIUBAH, hanya harganya yang diisi. Nama keluaran
# pengenal sering rusak, misalnya "SOCKET OUTLET ? 6. .0000", dan menggantinya
# dengan nama VLM yang bersih memang menolong pengguna. Tetapi itu menimpa,
# dan selain itu nilainya kecil: lubang hanya terisi ketika namanya sudah
# cocok, sehingga nama sistem pada baris itu memang sudah terbaca.

# shortcut: ambang 6 aksara pada pencocokan nama, berlaku bagi nama barang
# ritel Malaysia yang dicetak kasir. Naikkan kalau terukur ada domain yang
# nama barangnya lazim lebih pendek dari 6 aksara, misalnya kode SKU.
MIN_NAMA_ITEM = 6


def _samakan_nama(s):
    """Normalkan nama barang untuk dibandingkan.

    Imbuhan harga satuan dibuang, sebab VLM lazim menuliskannya menempel pada
    nama, misalnya "CHN CARROT @ 3.5000", sedangkan pengenal memisahkannya.
    """
    s = (s or "").upper()
    s = _re.sub(r"@.*$", "", s)
    s = _re.sub(r"[^A-Z0-9]+", " ", s)
    return " ".join(s.split())


def cocok_nama_item(a, b, k=MIN_NAMA_ITEM):
    """Satu nama termuat di dalam yang lain DAN yang terpendek cukup panjang.

    Syarat panjang itu wajib. Tanpa itu nama sependek CHN cocok dengan apa pun
    yang memuatnya, dan setiap lubang akan terisi oleh baris yang salah.
    """
    a, b = _samakan_nama(a), _samakan_nama(b)
    if not a or not b:
        return False
    pendek, panjang = sorted((a, b), key=len)
    return len(pendek) >= k and pendek in panjang


def perlu_tambal_item(kontrak):
    """True kalau ada baris barang yang harganya kosong.

    Pemicunya dapat diamati saat jalan dan tidak memakai jawaban benar. Nilai
    0 BUKAN kosong, aturan yang sama dengan _kosong di atas.
    """
    item = kontrak.get("item") or []
    if not item:
        return False
    return any(it.get("harga") is None for it in item)


def tambal_item(kontrak, item_vlm):
    """Isi harga baris barang yang kosong, dan HANYA yang kosong.

    Mengubah `kontrak` di tempat, dan mengembalikan daftar id baris yang
    terisi. Jumlah baris tidak pernah berubah, nama tidak pernah berubah, dan
    harga yang sudah ada tidak pernah berubah.

    Satu lubang diisi hanya bila namanya cocok dengan TEPAT SATU baris pada
    jawaban VLM. Dua kecocokan berarti rancu, dan yang rancu dibiarkan kosong:
    medan kosong yang disadari pengguna lebih aman daripada angka salah yang
    tidak disadarinya, sebagaimana dinyatakan Subbab 4.8.3.
    """
    item = kontrak.get("item") or []
    if not item or not item_vlm:
        return []
    calon = [v for v in item_vlm
             if isinstance(v, dict)
             and isinstance(v.get("harga"), (int, float))
             and (v.get("nama") or "").strip()]
    if not calon:
        return []
    terisi = []
    for baris in item:
        if baris.get("harga") is not None:
            continue
        kena = [v for v in calon if cocok_nama_item(baris.get("nama"),
                                                    v.get("nama"))]
        if len(kena) != 1:
            continue
        baris["harga"] = kena[0]["harga"]
        baris["asal"] = "vlm"
        baris["keyakinan"] = None
        terisi.append(baris.get("id"))
    # Penandanya juga sebagai daftar di akar, mengikuti pola `asalVlm`:
    # antarmuka wajib dapat menyorot baris mana saja yang harganya datang dari
    # model bahasa visual, sebab dua dari 14 isian terukur salah dan pengguna
    # satu-satunya yang dapat menangkapnya.
    if terisi:
        kontrak["itemAsalVlm"] = sorted(
            set(kontrak.get("itemAsalVlm") or []) | set(terisi))
    return terisi
