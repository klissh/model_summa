"""Fungsi pra-pemrosesan dan kontrak JSON.

Bagian atas berkas ini disalin APA ADANYA dari Notebook E. Jangan dirapikan,
jangan diganti namanya. Kalau isinya berbeda sedikit saja, model menerima
masukan yang tidak pernah dia lihat saat latih, dan angka di BAB IV tidak
lagi berlaku untuk server ini.

Berkas ini sengaja tidak mengimpor torch maupun easyocr, supaya bisa diuji
tanpa memuat model. Lihat uji.py.
"""

import collections
import contextlib
import contextvars
import statistics
import re as _re

# ---------------------------------------------------------------------------
# Notebook E sel 4. Sumber: Notebook A, PARAM_OCR = {"width_ths": 0.0}
# Kalibrasi Notebook A1: width_ths 0.0 memberi 97,4 kotak per struk (sasaran
# gold 97,6), sedangkan nilai bawaan 0,5 hanya 53,5 kotak.
# ---------------------------------------------------------------------------
BAHASA_OCR = ["en"]
PARAM_READTEXT = dict(detail=1, paragraph=False, width_ths=0.0)

_DESIMAL = _re.compile(r"(?<=\d),(?=\d{1,2}(?!\d))")


def normalisasi_angka(teks, label=None):
    if not teks:
        return teks
    if label in {"info.time", "info.date"}:
        return teks
    if ":" in teks or "/" in teks:
        return teks
    if _re.search(r"\d-\d", teks):
        return teks
    if not _re.search(r"\d", teks):
        return teks
    return _DESIMAL.sub(".", teks)


# ---------------------------------------------------------------------------
# Notebook E sel 7, yang sendiri disalin dari Notebook C sel 2.
# ---------------------------------------------------------------------------
def urut_baca(kotak, toleransi=0.6):
    sisa = sorted(kotak, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    baris, dipakai = [], set()
    for b in sisa:
        if id(b) in dipakai: continue
        y0, y1 = b["bbox"][1], b["bbox"][3]
        tinggi = max(1, y1 - y0)
        kel = [b]; dipakai.add(id(b))
        for c in sisa:
            if id(c) in dipakai: continue
            cy0, cy1 = c["bbox"][1], c["bbox"][3]
            tumpang = min(y1, cy1) - max(y0, cy0)
            if tumpang > toleransi * min(tinggi, max(1, cy1 - cy0)):
                kel.append(c); dipakai.add(id(c))
        kel.sort(key=lambda z: z["bbox"][0])
        baris.append(kel)
    baris.sort(key=lambda k: min(z["bbox"][1] for z in k))
    hasil, id_baris = [], []
    for i, k in enumerate(baris):
        for b in k:
            hasil.append(b); id_baris.append(i)
    return hasil, id_baris


def dari_bio(tag):
    return "O" if tag == "O" else tag[2:]


def norm_bbox(bbox, lebar, tinggi):
    x0, y0, x1, y1 = bbox
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    f = lambda v, d: min(1000, max(0, int(round(1000.0 * v / max(1, d)))))
    return [f(x0, lebar), f(y0, tinggi), f(x1, lebar), f(y1, tinggi)]


# ---------------------------------------------------------------------------
# Notebook E sel 9. Kontrak JSON seperti yang diukur.
# ---------------------------------------------------------------------------
def _uang(t):
    """Ubah teks jadi angka. Tahan spasi liar, minus di belakang, huruf tertukar."""
    if t is None: return None
    s = str(t).strip().upper().replace("RM", "").replace("MYR", "").replace("$", "").replace(" ", "")
    # Garis bawah itu titik desimal yang salah dibaca pengenal huruf: "10.99"
    # keluar sebagai "10 _" lalu "99". Tanpa pemetaan ini keduanya disaring
    # habis dan harga barangnya jadi 99,0.
    s = s.replace("_", ".")
    s = s.replace("O", "0").replace("I", "1").replace("L", "1")
    # Koma dengan TEPAT dua angka di ujung itu titik desimal yang salah
    # terbaca: pengenal huruf memberi "8, 00" untuk "8.00". Pemisah ribuan
    # selalu mengelompokkan tiga angka, jadi "1,234" dan "1,250" tidak
    # tersentuh, sedangkan "8,00" dan "1,234,50" dipulihkan.
    s = _re.sub(r",(\d{2})$", r".\1", s)
    s = s.replace(",", "")
    s = _re.sub(r"[^0-9.\-]", "", s)
    neg = s.endswith("-") or s.startswith("-")
    s = s.strip("-")
    # Titik di ujung itu sisa tanda baca OCR, bukan pemisah desimal. Tanpa ini
    # "04.36.;" jadi 436,0 dan "27.25." jadi 2725,0, yaitu nilai salah yang
    # percaya diri. 2725,0 memang tercatat sebagai kegagalan di G7.
    s = s.rstrip(".")
    if s.count(".") > 1:
        i = s.rfind("."); s = s[:i].replace(".", "") + s[i:]
    if s in ("", "."): return None
    if s.startswith("."): s = "0" + s
    try: v = float(s)
    except ValueError: return None
    if abs(v) > 100000: return None
    return -v if neg else v


def _taksir_kemiringan(pusat, tinggi_med):
    """Taksir kemiringan dominan struk dari sebaran titik tengah kotak.

    Profil proyeksi: untuk sejumlah kemiringan calon, titik tengah tegak
    diluruskan lalu dimasukkan ke laci. Kemiringan yang benar membuat titik
    menggerombol di sedikit laci, jadi jumlah kuadrat isi laci paling besar.
    Calon diurut dari yang paling dekat nol, supaya seri dimenangkan nol.
    """
    lebar_laci = max(1.0, tinggi_med / 2.0)

    def nilai(m):
        laci = collections.Counter(int((y - m * x) // lebar_laci) for x, y in pusat)
        return sum(c * c for c in laci.values())

    terbaik, nilai_terbaik = 0.0, nilai(0.0)
    for k in sorted(range(-70, 71), key=lambda z: (abs(z), z)):
        m = k * 0.005
        v = nilai(m)
        if v > nilai_terbaik:
            nilai_terbaik, terbaik = v, m
    # Kemiringan sungguhan menggerombolkan jauh lebih rapat daripada nol.
    # Tanpa ambang ini, sebaran acak pun menang tipis dan baris yang benar
    # malah dibelah.
    return terbaik if nilai_terbaik > 1.05 * nilai(0.0) else 0.0


def kelompok_baris(kot):
    """Nomor baris tiap kotak, tahan terhadap struk yang terfoto miring.

    BUKAN salinan Notebook E. Di sana penyusunan item memakai urut_baca(),
    yang syarat tumpang tindih tegaknya pecah di atas kira-kira 3 derajat:
    satu baris struk terbelah dua, sehingga nama barang lepas dari harganya.
    Pindaian SROIE hampir datar jadi tidak pernah kena, foto dari tangan
    selalu kena.

    Hanya dipakai untuk menyusun daftar item. Urutan token yang masuk ke
    LayoutLMv3 tetap dari urut_baca() dan tidak boleh ikut diubah, karena
    harus sama persis dengan saat model dilatih.
    """
    if not kot:
        return []
    tinggi = sorted(max(1, b["bbox"][3] - b["bbox"][1]) for b in kot)
    tinggi_med = tinggi[len(tinggi) // 2]
    pusat = [((b["bbox"][0] + b["bbox"][2]) / 2.0,
              (b["bbox"][1] + b["bbox"][3]) / 2.0) for b in kot]
    # Profil proyeksi butuh kotak yang banyak. Struk sungguhan punya sekitar
    # 97 kotak (kalibrasi Notebook A1), jadi ambang ini longgar di praktik.
    m = _taksir_kemiringan(pusat, tinggi_med) if len(kot) >= 20 else 0.0

    datar = sorted((y - m * x, i) for i, (x, y) in enumerate(pusat))
    ambang = 0.6 * tinggi_med
    id_baris = [0] * len(kot)
    r, jumlah, n = 0, 0.0, 0
    for y, i in datar:
        # Dibandingkan dengan rata-rata baris berjalan, bukan dengan anggota
        # terakhir, supaya kotak tidak merembet menyambung baris demi baris.
        if n and y - jumlah / n > ambang:
            r += 1
            jumlah, n = 0.0, 0
        jumlah += y
        n += 1
        id_baris[i] = r
    return id_baris


def _gabung(kot, label):
    """Gabungkan kotak berlabel sama yang bersebelahan jadi satu entitas."""
    out, kini = [], None
    for b in kot:
        if b["_label"] != label:
            if kini: out.append(kini); kini = None
            continue
        if kini is None:
            kini = dict(teks=[b["teks"]], yakin=[b["_yakin"]], bbox=list(b["bbox"]))
        else:
            kini["teks"].append(b["teks"]); kini["yakin"].append(b["_yakin"])
            kini["bbox"] = [min(kini["bbox"][0], b["bbox"][0]), min(kini["bbox"][1], b["bbox"][1]),
                            max(kini["bbox"][2], b["bbox"][2]), max(kini["bbox"][3], b["bbox"][3])]
    if kini: out.append(kini)
    for e in out:
        e["teks"] = " ".join(x for x in e["teks"] if x)
        e["yakin"] = round(sum(e["yakin"]) / max(1, len(e["yakin"])), 4)
    return out


def _angka_baris(bar, label):
    """Gabungkan dulu semua kotak berlabel sama dalam satu baris, baru diangkakan.

    CRAFT kerap memotong "10.90" menjadi dua kotak di titik desimalnya, dan
    keduanya dilabeli menu.price. Mengambil satu kotak saja memberi 90,0.
    Medan ringkasan sudah digabung lewat _gabung(); ini menyamakan
    perlakuannya untuk daftar item, yang sebelumnya terlewat.

    bar harus sudah urut kiri ke kanan.
    """
    pot = [b["teks"] for b in bar if b["_label"] == label]
    # _uang_potongan, bukan _uang: potongannya satu bilangan yang CRAFT pecah,
    # jadi titik desimal yang hilang atau terbaca koma masih bisa dipulihkan.
    return _uang_potongan(pot, kanan=True) if any((x or "").strip() for x in pot) else None


# Garis bawah ikut, sebab pengenal huruf memberi "10 _" untuk "10.". Tanpa itu
# potongan kiri sebuah harga ditolak di sini dan tidak pernah disatukan lagi.
_TANDA_ANGKA = set("0123456789.,-_")


_HOMOGLIF = (("O", "0"), ("I", "1"), ("L", "1"), ("|", "1"))


def _mirip_uang(teks):
    """Kotak yang isinya hanya angka dan tanda baca angka, boleh berlambang RM.

    Huruf yang `_uang` SUDAH petakan ke angka diperlakukan sebagai angka di
    sini juga. Tanpa ini kotak "RMO" (yaitu RM0) dan "RM3O" (yaitu RM30)
    disaring habis sebelum `_uang` berkesempatan memulihkannya, lalu nilainya
    hilang atau jadi ekor desimalnya saja. Terukur di uji3: "Service Taxe6%
    RMO 48" untuk 0,48 dan "TAXABLE AMT RM3O . 80" untuk 30,80.

    Digit sungguhan tetap diwajibkan, KECUALI bila ada lambang mata uang.
    Jadi "RMO" sah sebagai uang sedangkan huruf "O" yang berdiri sendiri
    tidak, dan kata seperti "LOT" atau "NO" tetap tersaring.
    """
    s = (teks or "").strip().upper()
    ada_mata = any(p in s for p in ("RM", "MYR", "$"))
    for buang in ("RM", "MYR", "$", " "):
        s = s.replace(buang, "")
    if not s:
        return False
    s2 = s
    for a, b in _HOMOGLIF:
        s2 = s2.replace(a, b)
    if not all(c in _TANDA_ANGKA for c in s2):
        return False
    return any(c.isdigit() for c in s) or ada_mata


def _harga_baris(bar):
    """Harga satu baris item, diambil dari kotak angka di kanan nama barang.

    Ini aturan LETAK, bukan keluaran model. Alasannya terukur: pada struk 99
    Speed Mart, CRAFT memotong "10.90" di titik desimalnya, dan model menolak
    melabeli potongan "10 ." karena bentuk begitu tidak pernah dilihatnya saat
    latih. Dari enam harga di struk itu hanya satu yang tidak terpotong, dan
    hanya yang satu itu yang berhasil dilabeli menu.price.

    Empat pengaman, karena aturan letak gampang salah tangkap:

    1. Hanya kotak di sebelah KANAN nama barang. Kolom kode barang (8049,
       5108) juga angka dan sebaris, tetapi letaknya di kiri nama.
    2. Potongan disatukan berdasar kedekatan mendatar, bukan asal gabung.
       Kolom kuantitas letaknya jauh, jadi tidak ikut tertelan. Tanpa ini,
       "2" dan "10.90" menyatu jadi 210,90.
    3. Kotak yang tingginya lebih dari enam kali teks sebaris dibuang. Cacat
       deteksi di tepi kertas bisa setinggi 7,2 baris dan terbaca "0".
       Batasnya sengaja longgar: 6,0 adalah nilai terlonggar yang masih
       membuang cacat itu, dan atas 43 struk ia tidak mengubah satu pun
       harga item lain.
       Letaknya paling kanan, jadi tanpa saringan ini ia memenangkan
       pengaman nomor 4 dan harga Colgate jadi 0,0.
    4. Gugus yang memuat kotak menu.price dimenangkan. Model tetap didahulukan
       kalau ia bersuara; letak hanya dipakai saat model diam.

    Hanya menyentuh daftar item. Medan ringkasan tidak lewat sini, jadi angka
    ketepatan di BAB IV tidak berubah.

    bar harus sudah urut kiri ke kanan.
    """
    nama = [b for b in bar if b["_label"] == "menu.nm"]
    if not nama:
        return _angka_baris(bar, "menu.price")
    batas = max(b["bbox"][2] for b in nama)
    tin = sorted(max(1, b["bbox"][3] - b["bbox"][1]) for b in bar)
    tinggi_med = tin[len(tin) // 2]
    calon = [b for b in bar if b["bbox"][0] >= batas
             and b["_label"] in ("O", "menu.price") and _mirip_uang(b["teks"])
             # shortcut: nisbah kasar, cuma menangkap cacat yang sangat besar;
             # naikkan ke aturan tumpang tindih pita baris kalau nanti sudah
             # ada jawaban benar tingkat item untuk mengukurnya.
             and 0.2 * tinggi_med <= max(1, b["bbox"][3] - b["bbox"][1]) <= 6.0 * tinggi_med]
    if not calon:
        return _angka_baris(bar, "menu.price")

    ambang = 0.6 * tinggi_med
    gugus, kini = [], [calon[0]]
    for b in calon[1:]:
        if b["bbox"][0] - kini[-1]["bbox"][2] <= ambang:
            kini.append(b)
        else:
            gugus.append(kini)
            kini = [b]
    gugus.append(kini)

    pilih = next((g for g in gugus if any(b["_label"] == "menu.price" for b in g)), gugus[-1])
    return _uang_potongan([b["teks"] for b in pilih], kanan=True)


def _buang_barcode(nm):
    """Buang barcode yang menempel di depan nama barang.

    Pengenal huruf kadang menyatukan kolom kode dengan kolom nama jadi satu
    kotak, dan kotak gabungan cuma bisa dapat satu label. Pada struk Lotus's
    13 dari 14 kode terlabeli menu.num dan terbuang benar; yang satu bocor
    karena tergabung, lalu muncul sebagai "08993175566438 NABAT I" di nama.

    Dua huruf salah baca dimaafkan, sebab "48a555149276688" juga barcode.
    Barcode EAN panjangnya 12 sampai 14 angka, dan nama barang tidak pernah
    diawali kata sepanjang itu yang isinya hampir semua angka.
    """
    pot = nm.split()
    if not pot:
        return ""
    kepala = pot[0].strip("\"'`~_-.,")
    if len(kepala) >= 12 and sum(c.isdigit() for c in kepala) >= len(kepala) - 2:
        return " ".join(pot[1:]).strip()
    return nm.strip()


# Ambang 15 kali median: disapu atas 213 harga item dari 41 struk nyata, dan
# 15 adalah nilai terketat yang masih nol korban. Pada 8 kali tangkapannya naik
# dari 11 jadi 16 dari 19 harga mustahil, tetapi satu harga wajar ikut terbuang.
_LIPAT_MUSTAHIL = 15


def _buang_harga_mustahil(item):
    """Kosongkan harga item yang mustahil. Mengosongkan, bukan menebak ulang.

    Dua tanda, keduanya disapu atas 213 harga item dari 41 struk nyata:

    1. Bilangan bulat empat angka atau lebih. Harga ritel Malaysia selalu
       dicetak dua desimal, jadi "72780" di kolom kanan itu kode barang yang
       salah tertangkap sebagai harga. Membuang 5 dari 19, nol korban.
    2. Lebih dari 15 kali median harga item LAIN di struk yang sama. Barang di
       satu struk tidak berjarak sejauh itu. Membuang 11 dari 19, nol korban.

    Harganya dikosongkan supaya pengguna melihat kolom kosong lalu mengisinya.
    Di aplikasi uang, angka salah yang percaya diri jauh lebih berbahaya
    daripada angka yang hilang: 72.780 untuk belanja 21,30 akan tertagih.
    """
    def _angka(h):
        return isinstance(h, (int, float)) and not isinstance(h, bool)

    punya = [(i, x["harga"]) for i, x in enumerate(item) if _angka(x["harga"])]
    for i, h in punya:
        if h >= 1000 and float(h).is_integer():
            item[i]["harga"] = None
            continue
        lain = sorted(v for j, v in punya if j != i)
        if not lain:
            continue
        med = statistics.median(lain)
        if med > 0 and h > _LIPAT_MUSTAHIL * med:
            item[i]["harga"] = None

    # Harga satuan yang jauh melebihi harga barisnya. Dari uji3: 15900,0
    # untuk baris berharga 15,90 dan 50,0 untuk baris berharga 3,50, yaitu
    # potongan angka yang tertukar kolom. Dikosongkan, bukan ditebak ulang.
    #
    # Syarat jumlah minimal satu menjaga barang timbangan: 0,15 kg seharga
    # RM 20 per kg memang berharga satuan jauh di atas harga barisnya.
    #
    # shortcut: ambang 5 kali dipilih dari 11 struk uji3 saja, bukan dari
    # sapuan besar seperti _LIPAT_MUSTAHIL; naikkan kalau ada korban nyata.
    for x in item:
        hs, hr, jm = x.get("harga_satuan"), x.get("harga"), x.get("jumlah")
        if (_angka(hs) and _angka(hr) and hr > 0
                and _angka(jm) and jm >= 1 and hs > 5 * hr):
            x["harga_satuan"] = None
    return item


def _satukan_dua_baris(item):
    """Satukan baris nama yang kosong angka dengan baris angka yang kosong nama.

    Sebagian struk, seperti LOGAM M&A Enterprise, mencetak nama barang di
    baris sendiri lalu kode, kuantitas dan harganya di baris berikutnya.
    Pengelompokan per baris memisahkan keduanya jadi dua item setengah jadi.

    Syaratnya sengaja ketat: baris atas harus punya nama DAN tidak punya satu
    pun angka uang, baris bawah harus punya harga DAN tidak punya nama. Struk
    satu baris seperti Lotus's dan 99 Speed Mart tidak pernah memenuhi ini.
    """
    out = []
    for it in item:
        atas = out[-1] if out else None
        if (atas is not None and it["nama"] is None and it["harga"] is not None
                and atas["nama"] is not None and atas["harga"] is None
                and atas["harga_satuan"] is None and atas["diskon"] is None):
            for k in ("jumlah", "harga_satuan", "harga", "diskon"):
                atas[k] = it[k]
            atas["yakin"] = min(atas["yakin"], it["yakin"])
            continue
        out.append(it)
    return out


def ke_json(kot, lebar, tinggi):
    """Kontrak JSON untuk aplikasi. Setiap nilai membawa keyakinannya sendiri."""
    def satu(label, angka=False):
        e = _gabung(kot, label)
        if not e: return None
        e = max(e, key=lambda z: z["yakin"])
        nilai = _uang(e["teks"]) if angka else e["teks"]
        return {"teks": e["teks"], "nilai": nilai, "yakin": e["yakin"], "bbox": e["bbox"]}

    # item disusun per baris, karena satu baris struk adalah satu item.
    # Barisnya ditaksir ulang di sini, tidak memakai urut_baca(), supaya
    # struk yang terfoto miring tidak terbelah. Lihat kelompok_baris().
    # Ini satu-satunya penyimpangan dari Notebook E, dan hanya menyentuh
    # daftar item. Medan ringkasan di bawah memakai _gabung() atas urutan
    # kot yang tidak berubah, jadi angka ketepatan di BAB IV tetap berlaku.
    id_baris = kelompok_baris(kot)
    per_baris = collections.defaultdict(list)
    for b, r in zip(kot, id_baris):
        per_baris[r].append(b)
    item = []
    for r in sorted(per_baris):
        bar = sorted(per_baris[r], key=lambda z: z["bbox"][0])
        nm = _buang_barcode(" ".join(b["teks"] for b in bar if b["_label"] == "menu.nm"))
        if not (nm or any(b["_label"] == "menu.price" for b in bar)): continue
        # Kolom "satuan * kuantitas" didahulukan: ia memuat kuantitas yang
        # dicetak apa adanya, sedangkan menu.cnt sering hanya potongannya.
        jml = _angka_bintang(bar, _ambang_baris(bar)) or _angka_baris(bar, "menu.cnt")
        item.append({
            "nama": nm or None,
            # Struk tanpa kolom kuantitas, seperti 99 Speed Mart, berarti satu
            # per barang. None membuat aplikasi menampilkan 0, yang salah.
            #
            # Nol diperlakukan sama dengan kosong: barang berkuantitas nol
            # tidak pernah tercetak di struk, jadi 0 selalu salah tangkap,
            # biasanya dari kolom diskon 0.00 di sebelahnya.
            "jumlah": 1.0 if not jml else jml,
            "harga_satuan": _angka_baris(bar, "menu.unitprice"),
            "harga": _harga_baris(bar),
            "diskon": _angka_baris(bar, "menu.discountprice"),
            "yakin": round(min([b["_yakin"] for b in bar if b["_label"] != "O"] or [0.0]), 4),
        })

    item = _buang_harga_mustahil(_satukan_dua_baris(item))

    toko = _gabung(kot, "store.nm")
    return {
        "toko": " ".join(e["teks"] for e in toko) if toko else None,
        "tanggal": satu("info.date"),
        "jam": satu("info.time"),
        "item": item,
        "subtotal": satu("sub_total.subtotal_price", angka=True),
        "pajak": satu("sub_total.tax_price", angka=True),
        "servis": satu("sub_total.service_price", angka=True),
        "pembulatan": satu("sub_total.etc", angka=True),
        "total": satu("total.total_price", angka=True),
        "tunai": satu("total.cashprice", angka=True),
        "kartu": satu("total.creditcardprice", angka=True),
        "emoney": satu("total.emoneyprice", angka=True),
        "kembalian": satu("total.changeprice", angka=True),
        "ukuran_gambar": {"lebar": lebar, "tinggi": tinggi},
    }


# ---------------------------------------------------------------------------
# Di bawah ini BUKAN salinan. Penyesuai bentuk JSON ke src/fitur/struk/tipe.ts
# di repo aplikasi. Hanya mengganti nama medan dan menyusun ulang, tidak ada
# nilai yang dihitung ulang, sehingga tidak mempengaruhi angka ketepatan.
# ---------------------------------------------------------------------------
# Kuantitas di atas ini tidak pernah ada di struk ritel. Batasnya longgar
# dengan sengaja: yang dibuang hanya angka yang jelas mustahil.
JUMLAH_MAKS = 100


def jumlah_tampil(jml):
    """Kuantitas untuk aplikasi. Pecahan DIPERTAHANKAN, yang mustahil dibuang.

    Dua kerusakan nyata yang ditangani, keduanya terlihat pada struk ST Rosyam
    Mart di layar aplikasi:

      int(0.404) jadi 0   Struk Malaysia menjual sayur per kilogram, sehingga
                          0,404 kg itu kuantitas yang SAH. Memangkasnya ke
                          bilangan bulat membuat aplikasi menampilkan 0.
      404                 OCR memecah "3.29*0.404" sehingga "404" tertangkap
                          sebagai kuantitas. Kuantitas 404, 503, dan 991 tidak
                          pernah ada di struk ritel.

    Yang mustahil dikembalikan None, bukan ditebak ulang, supaya pengguna
    melihat kolom kosong lalu mengisinya sendiri. Sama dengan alasan
    _buang_harga_mustahil: angka salah yang percaya diri lebih berbahaya
    daripada angka yang hilang.
    """
    if not isinstance(jml, (int, float)) or isinstance(jml, bool):
        return None
    j = float(jml)
    if j <= 0 or j > JUMLAH_MAKS:
        return None
    return int(j) if j.is_integer() else round(j, 3)


EKSTENSI_GAMBAR = {"jpeg": ".jpg", "png": ".png", "webp": ".webp",
                   "bmp": ".bmp", "tiff": ".tif"}


def ekstensi_gambar(format_pillow):
    """Nama format dari Pillow jadi akhiran berkas. None kalau tidak didukung.

    EasyOCR membaca lewat skimage, dan skimage memilih pembaca berdasarkan
    AKHIRAN berkas, bukan isinya. Akhiran yang salah membuatnya memilih
    pembaca citra medis SimpleITK lalu gagal.
    """
    return EKSTENSI_GAMBAR.get((format_pillow or "").lower())


MEDAN_RINGKASAN = {
    "subtotal": "sub_total.subtotal_price",
    "pajak": "sub_total.tax_price",
    "total": "total.total_price",
    "tunai": "total.cashprice",
    "kembalian": "total.changeprice",
    # Skema anotasi punya dua cara bayar nontunai, tetapi sebelumnya keduanya
    # dibuang di sini sehingga tidak pernah sampai ke aplikasi. Akibatnya
    # struk DuitNow QR terlihat seperti dibayar tunai.
    # Jarang di data: emoney 7 struk, kartu 13 struk, dari 162. Jadi model
    # memang jarang meramalkannya, tetapi membuangnya membuat yang sedikit
    # itu pun pasti hilang.
    # Tidak mengubah angka ketepatan: ukur_server.py mengukur delapan medan
    # yang didaftarnya sendiri, dan tunai tetap dipetakan ke cashprice.
    "kartu": "total.creditcardprice",
    "emoney": "total.emoneyprice",
}


def asal_per_label(kot):
    """Label mana yang jawabannya datang dari aturan, bukan dari model.

    Dipakai medan `asal` di tipe.ts, supaya antarmuka bisa memberi tahu
    pengguna mesin mana yang menjawab.
    """
    out = {}
    for b in kot:
        L = b.get("_label", "O")
        if L == "O":
            continue
        a = "aturan" if b.get("_dari_aturan") else "model"
        # kalau satu entitas tersusun dari beberapa kotak, cukup satu kotak
        # berasal dari aturan untuk menandai medan itu sebagai dari aturan
        if out.get(L) != "aturan":
            out[L] = a
    return out


def ke_kontrak_aplikasi(hasil, kot, id_struk, versi_model):
    """Dari keluaran ke_json jadi bentuk HasilEkstraksi di tipe.ts."""
    asal = asal_per_label(kot)

    def medan(kunci):
        d = hasil.get(kunci)
        label = MEDAN_RINGKASAN[kunci]
        if d is None:
            return {"nilai": None, "teks": None, "keyakinan": None, "asal": None}
        return {
            "nilai": d["nilai"],
            "teks": d["teks"],
            "keyakinan": d["yakin"],
            "asal": asal.get(label, "model"),
        }

    def teks_saja(kunci):
        d = hasil.get(kunci)
        return d["teks"] if d else None

    item = []
    for n, it in enumerate(hasil["item"], start=1):
        item.append({
            "id": f"i{n}",
            "nama": it["nama"] or "",
            "jumlah": jumlah_tampil(it["jumlah"]),
            "hargaSatuan": it["harga_satuan"],
            "harga": it["harga"],
            "keyakinan": it["yakin"],
        })

    return {
        "id": id_struk,
        "versiModel": versi_model,
        "toko": hasil["toko"],
        "tanggal": teks_saja("tanggal"),
        "jam": teks_saja("jam"),
        "item": item,
        "ringkasan": {k: medan(k) for k in MEDAN_RINGKASAN},
    }


# ---------------------------------------------------------------------------
# PENAMBALAN AGREGASI (varian V6). BUKAN salinan Notebook E.
#
# Sengaja TIDAK dimasukkan ke dalam ke_json(), karena ke_json() juga dipakai
# menyusun jawaban gold di ukur_server.py. Kalau gold ikut ditambal,
# pengukurannya membandingkan sistem dengan dirinya sendiri.
#
# Disetel pada 12 struk pilot, lalu diuji SEKALI pada 16 struk uji: ketepatan
# gabungan 30,7 persen menjadi 43,6 persen. Enam varian dicoba. Rinciannya di
# aplikasi/percobaan_agregasi/pra_registrasi.md dan catatan BAB IV bagian G10.
#
# Seluruhnya bekerja di hilir model. Kotak, label, dan masukan model tidak
# disentuh, jadi angka tingkat kotak di bagian A sampai G8 tetap berlaku.
# ---------------------------------------------------------------------------
import pembanding_aturan as _PA

# Titik TIDAK boleh jadi pemisah jam: "7.95" itu harga, bukan jam.
_POLA_JAM = _re.compile(r"^\d{1,2}[:;]\d{2}(?:[:;]\d{2})?\s*[AP]?\.?M?\.?$", _re.I)
_POLA_TANGGAL = _re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$")

# "SUBT" bukan "SUB", supaya nama toko seperti SUBWAY tidak ikut tertangkap.
# medan -> (kata kunci, kata pembatal, ambil baris paling bawah)
#
# `total` ditambahkan setelah uji3: model hanya melabeli total.total_price
# pada 3 dari 11 struk, padahal 9 dari 11 punya kotak bertuliskan TOTAL.
# Kata TOTAL juga muncul di SUB TOTAL, TOTAL ITEM, TOTAL QTY, TOTAL SAVINGS
# dan TOTAL SALES EXCLUSIVE SST, yang semuanya BUKAN jumlah akhir. Karena itu
# pembatalnya panjang, dan yang diambil baris paling bawah: jumlah akhir
# selalu tercetak di bawah rinciannya.
_KATA_MEDAN = {"subtotal": (("SUBT", "AMOUNT"), (), False),
               "pajak": (("GST", "SST", "TAX", "CUKAI"), (), False),
               "total": (("TOTAL", "AMOUNTDUE", "JUMLAHBESAR"),
                         ("SUBTOTAL", "SUBTTL", "SAVING", "QTY", "ITEM",
                          "DISCOUNT", "EXCLUSIVE", "EXCTAX", "CHANGE",
                          "TENDER", "POINT"), True)}


_SAMBUNG = ",'`\u2019"


# Satu bilangan uang yang sudah lengkap: dua desimal, boleh ribuan dan bintang.
_UANG_UTUH = _re.compile(r"(?i)(?:RM)?-?\d{1,3}(?:,\d{3})*\.\d{2}\*?")
# "61.200.00": dua bilangan yang menempel, batasnya tidak dapat dipastikan.
_TITIK_GANDA = _re.compile(r"\d\.\d{2,3}\.\d{2}(?!\d)")

# Dua aturan di atas hanya hidup di jalur prediksi. ke_json() dan nilai_item.py
# memakai _uang_potongan untuk menyusun jawaban rujukan, jadi bawaannya harus
# tetap perilaku lama (anti-pencemaran). ContextVar, bukan peubah global, sebab
# peladen melayani beberapa utas sekaligus.
_PREDIKSI = contextvars.ContextVar("prediksi", default=False)


@contextlib.contextmanager
def _jalur_prediksi():
    tanda = _PREDIKSI.set(True)
    try:
        yield
    finally:
        _PREDIKSI.reset(tanda)


def _uang_potongan(potong, kanan=False):
    """Angkakan potongan satu bilangan, tahan titik desimal yang tidak terbaca.

    "2 ." + "05" jadi 2.05, titiknya terbaca dan _uang sudah cukup.
    "56" + "95"  jadi 56.95, titiknya hilang lalu dipulihkan dari angka ekor.
    "2,"  + "40" jadi 2.40, komanya di sambungan jadi itu titik desimal.

    Pemulihan ekor hanya saat ada dua potongan atau lebih dan tidak satu pun
    memuat titik atau koma, supaya bilangan utuh "170" tidak jadi 1,70.
    """
    p = [x for x in ((q or "").strip() for q in potong) if x]
    # Dua uang utuh itu dua nilai, bukan potongan satu bilangan: menyambungnya
    # memberi 61200,0 dari "61.20" + "0.00". Baris barang (kanan=True) mengambil
    # kolom harga paling kanan; ringkasan menolak.
    if _PREDIKSI.get():
        utuh = [x for x in p if _UANG_UTUH.fullmatch(x.replace(" ", ""))]
        if len(utuh) >= 2:
            return _uang(utuh[-1]) if kanan else None
        if len(p) == 1 and _TITIK_GANDA.search(p[0]):
            return None
    # Koma atau petik di SAMBUNGAN antarpotongan itu titik desimal yang salah
    # terbaca. Pemisah ribuan tidak pernah di sambungan, ia selalu di dalam satu
    # potongan, jadi "1,234" + "56" tidak tersentuh.
    for i in range(len(p) - 1):
        if p[i][-1] in _SAMBUNG:
            p[i] = p[i][:-1] + "."
        if p[i + 1][0] in _SAMBUNG:
            p[i + 1] = "." + p[i + 1][1:]
    teks = " ".join(p)
    # Spasi dengan TEPAT dua angka di ujung itu titik desimal yang terbaca
    # spasi: satu kotak berbunyi "RM76 90" untuk 76,90, dan tanpa ini jadi
    # 7690,0 di struk yang totalnya 99,75. Pemisah ribuan mengelompokkan tiga
    # angka, jadi "1 234" tidak tersentuh.
    #
    # SENGAJA di sini, bukan di _uang. _uang dipakai ke_json untuk menyusun
    # jawaban gold saat mengukur, dan kalau ia ikut ditambal maka gold ikut
    # ditambal. Lihat uji.py bagian e.
    teks = _re.sub(r"\s(\d{2})$", r".\1", teks)
    v = _uang(teks)
    if len(p) >= 2 and not any(c in ".," for c in teks):
        ekor = p[-1]
        if len(ekor) == 2 and ekor.isdigit():
            kepala = _uang(" ".join(p[:-1]))
            if kepala is not None:
                return kepala + int(ekor) / 100.0
    return v


def _angka_bintang(bar, ambang):
    """Kuantitas dari kolom "satuan * kuantitas", atau None.

    Struk pasar swalayan Malaysia mencetak barisnya begini:

        025102000133004043   3.29*0.404   1.33

    yaitu barcode, lalu harga satuan kali kuantitas, lalu harga baris. Yang
    penting di sini: ketika pengenal kehilangan angka depan harga satuan,
    KUANTITAS DI KANAN BINTANG SERING SELAMAT. Pada struk ST Rosyam Mart
    kotaknya berbunyi "29*0." lalu "404", sehingga harga satuannya rusak
    tetapi 0,404 masih utuh.

    Tanpa ini, menu.cnt menangkap potongannya saja dan aplikasi menampilkan
    kuantitas 404, 503, atau 480 untuk sayur yang dijual per kilogram.

    Dua penjaga:

    1. Harus ada ANGKA tepat sebelum bintang di kotak yang sama. Itu yang
       membedakan kolom harga dari nama barang: "85GM *5'S" dan "RING*KG"
       bukan kolom kuantitas, dan keduanya ada di struk yang sama.
    2. Perluasan ke kanan berhenti pada jarak antarkotak, sama seperti
       _gugus_semua, supaya harga baris di kolom kanan tidak ikut tertelan.
    """
    for i, b in enumerate(bar):
        t = b["teks"] or ""
        if "*" not in t:
            continue
        kepala, _, ekor = t.rpartition("*")
        if not (kepala and kepala.rstrip()[-1:].isdigit()):
            continue
        potong, kanan = [ekor], b["bbox"][2]
        for c in bar[i + 1:]:
            if c["bbox"][0] - kanan > ambang:
                break
            potong.append(c["teks"] or "")
            kanan = c["bbox"][2]
        s = _re.sub(r"[^0-9.]", "", "".join(potong))
        # Titik ganda muncul di sambungan potongan: "0." + ".420" jadi "0..420"
        if s.count(".") > 1:
            k = s.find(".")
            s = s[:k + 1] + s[k + 1:].replace(".", "")
        if not s or s == ".":
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        if 0 < v <= JUMLAH_MAKS:
            return v
    return None


def _baris_kotak(kot, idb):
    baris = {}
    for b, r in zip(kot, idb):
        baris.setdefault(r, []).append(b)
    for v in baris.values():
        v.sort(key=lambda z: z["bbox"][0])
    return baris


def _ambang_baris(sebaris):
    tin = sorted(max(1, b["bbox"][3] - b["bbox"][1]) for b in sebaris)
    return 0.6 * tin[len(tin) // 2]


def _gugus_semua(sebaris, ambang):
    """Seluruh gugus angka pada satu baris, urut dari kiri ke kanan."""
    calon = [b for b in sebaris if _mirip_uang(b["teks"])]
    if not calon:
        return []
    gugus, kini = [], [calon[0]]
    for b in calon[1:]:
        if b["bbox"][0] - kini[-1]["bbox"][2] <= ambang:
            kini.append(b)
        else:
            gugus.append(kini)
            kini = [b]
    gugus.append(kini)
    return gugus


def _gugus_kanan(sebaris, ambang):
    """Gugus angka paling kanan pada satu baris, potongannya sudah disatukan."""
    gugus = _gugus_semua(sebaris, ambang)
    return gugus[-1] if gugus else None


def _jendela_kiri(sebaris, gugus):
    """Kata yang menempel di kiri gugus angka, tanpa menyeberangi angka lain.

    Satu baris struk sering memuat DUA pasangan kata dan angka sekaligus:

        Total Saving 5. 32 Total 39, 25

    Menilai seluruh baris membuat kata pembatal "SAVING" membatalkan 39,25
    yang justru jawaban benarnya, padahal kata yang menerangkan angka itu
    hanyalah "Total" tepat di kirinya. Jendela berhenti pada angka pertama
    yang ditemui ke kiri, jadi "Sub Total 39, 23" tetap terbaca SUBTOTAL dan
    tetap dibatalkan.
    """
    x = min(b["bbox"][0] for b in gugus)
    kiri = sorted((b for b in sebaris if b["bbox"][2] <= x),
                  key=lambda b: b["bbox"][0], reverse=True)
    kata = []
    for b in kiri:
        t = b["teks"] or ""
        if _mirip_uang(t):
            break
        if any(c.isalpha() for c in t):
            kata.append(t)
    return "".join(c for c in "".join(reversed(kata)).upper() if c.isalpha())


def _isi(h, medan, kotak2):
    h[medan] = {"teks": " ".join(b["teks"] for b in kotak2),
                "nilai": _uang_potongan([b["teks"] for b in kotak2]),
                "yakin": round(sum(b["_yakin"] for b in kotak2) / len(kotak2), 4),
                "bbox": [min(b["bbox"][0] for b in kotak2), min(b["bbox"][1] for b in kotak2),
                         max(b["bbox"][2] for b in kotak2), max(b["bbox"][3] for b in kotak2)]}


def _satukan_potongan(kot, idb, h):
    """V1. Satukan potongan bilangan yang CRAFT pecah di titik desimalnya.

    Perluasan mengabaikan label tetangga, karena justru itu masalahnya: kedua
    potongan sering mendapat label berbeda. Dibatasi tiga hal, yaitu harus
    sebaris, harus berisi angka saja, dan jaraknya di bawah 0,6 tinggi huruf.

    Ini satu-satunya komponen yang mengubah nilai yang sudah ada, jadi
    satu-satunya yang bisa menurunkan ketepatan.
    """
    for medan, label in MEDAN_RINGKASAN.items():
        e = h.get(medan)
        if not e:
            continue
        x0, y0, x1, y1 = e["bbox"]
        inti = [i for i, b in enumerate(kot)
                if b["_label"] == label and b["bbox"][0] >= x0 and b["bbox"][2] <= x1
                and b["bbox"][1] >= y0 and b["bbox"][3] <= y1]
        if not inti:
            continue
        baris = {idb[i] for i in inti}
        sebaris = sorted((i for i, r in enumerate(idb) if r in baris),
                         key=lambda i: kot[i]["bbox"][0])
        ambang = _ambang_baris([kot[i] for i in sebaris])
        pakai, tumbuh = set(inti), True
        while tumbuh:
            tumbuh = False
            kiri = min(kot[i]["bbox"][0] for i in pakai)
            kanan = max(kot[i]["bbox"][2] for i in pakai)
            for i in sebaris:
                if i in pakai or not _mirip_uang(kot[i]["teks"]):
                    continue
                if kot[i]["bbox"][0] - kanan <= ambang and kiri - kot[i]["bbox"][2] <= ambang:
                    pakai.add(i)
                    tumbuh = True
        potong = [kot[i]["teks"] for i in sorted(pakai, key=lambda i: kot[i]["bbox"][0])]
        if len(potong) == 1:
            continue
        # Dua kotak yang SUDAH berlabel sama pun tetap diproses di sini,
        # karena "11" + "10" perlu pemulihan titik desimal walau tidak ada
        # tetangga baru yang diserap.
        v = _uang_potongan(potong)
        if v is None:
            # Gabungan yang tidak masuk akal, misalnya kemasukan "Q0",
            # dibiarkan. Nilai semula lebih baik daripada None.
            continue
        e["nilai"] = v
        e["teks"] = " ".join(potong)
    return h


def _cadangan_aturan(kot, lebar, tinggi, h):
    """V2. Jawaban pembanding aturan, hanya untuk medan yang masih kosong.

    hibrida_penengah.SET_ATURAN_MENANG tidak memuat sub_total.tax_price
    maupun total.changeprice, jadi jawaban aturan untuk keduanya selalu
    dibuang penengah walau sering benar. Penengah itu beku dan tidak
    disentuh, cadangan ini dipasang sesudahnya.
    """
    aturan = _PA.prediksi({"kotak": kot, "lebar": lebar, "tinggi": tinggi})
    for medan, label in MEDAN_RINGKASAN.items():
        if (h.get(medan) or {}).get("nilai") is not None:
            continue
        ambil = [b for b, a in zip(kot, aturan) if a == label]
        if not ambil:
            continue
        ambil.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        if _uang_potongan([b["teks"] for b in ambil]) is not None:
            _isi(h, medan, ambil)
    return h


# "2024.09.1819:55:44": DBNet membaca tanggal dan jam sebagai satu kotak.
_TANGGAL_JAM = _re.compile(
    r"(?<!\d)((?:20\d{2}([./-])\d{2}\2\d{2})|(?:\d{2}([./-])\d{2}\3 *20\d{2}))"
    r"((?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?: ?[AP]M)?)(?!\d)", _re.I)


def _cadangan_bentuk(kot, h):
    """V4. Tanggal dan jam lewat bentuk tulisannya, hanya saat medan kosong.

    Tidak disetel pada data sama sekali. Pada 16 struk uji justru komponen
    inilah yang paling banyak menolong, tanggal naik 6, karena ia berasal
    dari pengetahuan bentuk dan bukan dari 12 struk pilot.
    """
    for medan, pola in (("tanggal", _POLA_TANGGAL), ("jam", _POLA_JAM)):
        if h.get(medan):
            continue
        for b in kot:  # urutan baca, jadi yang paling atas menang
            t = (b["teks"] or "").strip()
            if pola.match(t):
                h[medan] = {"teks": t, "nilai": t, "yakin": round(b["_yakin"], 4),
                            "bbox": list(b["bbox"])}
                break
    # Hanya mengisi yang kosong, atau merapikan jawaban yang kotaknya memang
    # kotak gabungan ini. Jawaban dari kotak lain tidak disentuh.
    for b in kot:
        m = _TANGGAL_JAM.search(b["teks"] or "")
        if not m:
            continue
        for medan, nilai in (("tanggal", m.group(1)), ("jam", m.group(4))):
            if not h.get(medan) or h[medan]["teks"] == b["teks"]:
                h[medan] = {"teks": nilai, "nilai": nilai, "yakin": round(b["_yakin"], 4),
                            "bbox": list(b["bbox"])}
        break
    return h


def _cadangan_kata(kot, idb, h):
    """V6. Kata kunci sebaris, hanya saat medan kosong.

    Model tidak pernah sekali pun melabeli subtotal maupun pajak pada 12
    struk pilot, padahal teks kotaknya sering sempurna. Pembanding aturan
    juga lewat, karena kata kuncinya rusak dibaca OCR: "Sub Tatal",
    "SubTota|", "Tota Amount:". Pencocokan atas HURUF saja membuatnya tahan
    terhadap kotak terpotong dan tanda baca liar.
    """
    baris = _baris_kotak(kot, idb)
    huruf = {r: "".join(c for c in "".join(b["teks"] or "" for b in v).upper() if c.isalpha())
             for r, v in baris.items()}
    for medan, (kata, batal, dari_bawah) in _KATA_MEDAN.items():
        if (h.get(medan) or {}).get("nilai") is not None:
            continue
        urut = sorted(baris, reverse=dari_bawah)
        for r in urut:
            if not any(w in huruf[r] for w in kata):
                continue
            pilih = _gugus_kanan(baris[r], _ambang_baris(baris[r]))
            if not pilih:
                continue
            # Kata kunci dan pembatal dinilai pada jendela di kiri angka yang
            # dipilih, bukan pada seluruh baris. Lihat _jendela_kiri.
            jendela = _jendela_kiri(baris[r], pilih)
            if not any(w in jendela for w in kata):
                continue
            if any(w in jendela for w in batal):
                continue
            if _uang_potongan([b["teks"] for b in pilih]) is not None:
                _isi(h, medan, pilih)
                break
    return h


# Tarif SST Malaysia. Pitanya sempit dengan sengaja: pada 11 struk uji3
# keenam nilai pajak berada di 5,98 sampai 6,01 persen dari dasarnya, jadi
# pasangan palsu hampir mustahil. Service charge 10 persen TIDAK dimasukkan,
# sebab ia medan `servis` yang terpisah dan memasukkannya membuat caj
# perkhidmatan terambil sebagai pajak.
#
# shortcut: pita ini dari tarif 6 dan 8 persen saja, bukan dari sapuan besar.
# Tambahkan tarif baru kalau Malaysia mengubahnya lagi.
_PITA_SST = ((0.0580, 0.0620), (0.0780, 0.0820))
_KATA_SST = ("GST", "SST", "TAX", "CUKAI")
_BATAL_SST = ("INVOICE", "EXEMPT", "ZERO", "NOTAPPLICABLE")


def _angka_tarif(gugus, nilai):
    """Benar kalau gugus ini angka TARIF, bukan nilai uang.

    "SST 6%" dan "6 Qo SST" memberi gugus bernilai 6 yang bukan dasar
    pengenaan pajak. Tanpa penolakan ini, pasangan (6, 0.48) pada uji3_09
    lolos pita 8 persen dan menetapkan dasar yang salah.
    """
    teks = " ".join(b["teks"] or "" for b in gugus)
    if "%" in teks:
        return True
    return nilai in (6.0, 8.0, 10.0) and not any(c in ".," for c in teks)


def _cadangan_pajak_sebaris(kot, idb, h):
    """V7. Pajak dari pasangan angka SEBARIS pada baris ringkasan SST.

    Struk Malaysia mencetak ringkasan pajak sebagai satu baris bertingkat:

        SST 6%   81.00   4.84

    yaitu tarif, dasar pengenaan pajak, lalu nilai pajaknya. Mengambil gugus
    paling kanan tanpa memeriksa apa pun memberi jawaban yang percaya diri
    tetapi salah: pada uji3 ia menghasilkan 0,04 (nilai pembulatan di baris
    lain) dan 2,55 (nilai diskon).

    Yang dicari di sini pasangan bersebelahan (dasar, pajak) yang rasionya
    jatuh pada tarif SST. Dasarnya sekalian dipakai mengisi `subtotal` kalau
    medan itu masih kosong, sebab pada tata letak ini dasar pengenaan pajak
    memang subtotalnya.
    """
    baris = _baris_kotak(kot, idb)
    huruf = {r: "".join(c for c in "".join(b["teks"] or "" for b in v).upper() if c.isalpha())
             for r, v in baris.items()}
    for r in sorted(baris):
        if not any(w in huruf[r] for w in _KATA_SST):
            continue
        if any(w in huruf[r] for w in _BATAL_SST):
            continue
        sebaris = baris[r]
        gugus = _gugus_semua(sebaris, _ambang_baris(sebaris))
        nilai = [(g, _uang_potongan([b["teks"] for b in g])) for g in gugus]
        nilai = [(g, v) for g, v in nilai if v is not None and v > 0]
        for i in range(len(nilai) - 1):
            (gd, dasar), (gp, pajak) = nilai[i], nilai[i + 1]
            if _angka_tarif(gd, dasar):
                continue
            if not any(a <= pajak / dasar <= b for a, b in _PITA_SST):
                continue
            # Nilai yang sudah ada hanya ditimpa kalau ia TIDAK ikut memenuhi
            # tarif terhadap dasar yang sama. Jadi pajak yang sudah benar
            # tidak pernah diganti, sedangkan 0,04 (pembulatan) dan 0,0 pada
            # uji3 diganti oleh pasangan yang terbukti aritmetikanya.
            kini = (h.get("pajak") or {}).get("nilai")
            if kini is not None and any(a <= kini / dasar <= b for a, b in _PITA_SST):
                return h
            _isi(h, "pajak", gp)
            if (h.get("subtotal") or {}).get("nilai") is None:
                _isi(h, "subtotal", gd)
            return h
    return h


_KATA_EWALLET = ("WALLET", "TNG", "EWALLET", "DUITNOW", "BOOST", "GRABPAY",
                 "SHOPEEPAY", "QRPAY")
_KATA_KARTU = ("VISA", "MASTERCARD", "MYDEBIT", "DEBITCARD", "CREDITCARD")


def _pindah_nontunai(kot, idb, h):
    """Pindahkan nilai `tunai` ke emoney atau kartu kalau barisnya menyebut
    alat bayar nontunai.

    Alasannya sudah tercatat di MEDAN_RINGKASAN: model jarang meramalkan medan
    nontunai sebab di data latih hanya 7 struk emoney dan 13 kartu dari 162,
    sehingga struk DuitNow QR terlihat seperti dibayar tunai. Baris alat bayar
    di struk Malaysia memuat namanya, misalnya "AMBANK TNGWALLET 61.20", jadi
    kata itulah buktinya, bukan tebakan.
    """
    e = h.get("tunai")
    if not e or e.get("nilai") is None:
        return h
    x0, y0, x1, y1 = e["bbox"]
    baris = set()
    for b, r in zip(kot, idb):
        bx = b["bbox"]
        if bx[0] >= x0 and bx[2] <= x1 and bx[1] >= y0 and bx[3] <= y1:
            baris.add(r)
    if not baris:
        return h
    huruf = "".join(c for c in "".join(
        b["teks"] or "" for b, r in zip(kot, idb) if r in baris).upper()
        if c.isalpha())
    for tujuan, kata in (("emoney", _KATA_EWALLET), ("kartu", _KATA_KARTU)):
        if not any(w in huruf for w in kata):
            continue
        if (h.get(tujuan) or {}).get("nilai") is not None:
            return h
        h[tujuan] = h.pop("tunai")
        return h
    return h


def _jaga_wajar(h):
    """Kosongkan pajak dan subtotal yang mustahil terhadap total. Tidak menebak.

    Pajak setengah total atau lebih, dan subtotal di atas dua kali total, tidak
    pernah sah. Penjaga diam bila total kosong, nol, atau di bawah separuh
    jumlah harga barang, sebab saat itu total sendiri yang patut dicurigai.
    """
    nil = lambda m: (h.get(m) or {}).get("nilai")
    total = nil("total")
    barang = sum(x["harga"] for x in h["item"] if isinstance(x["harga"], (int, float)))
    if total is None or total <= 0 or total < 0.5 * barang:
        return h
    if (nil("pajak") or 0) >= 0.5 * total:
        h["pajak"] = None
    if (nil("subtotal") or 0) > 2 * total:
        h["subtotal"] = None
    return h


@_jalur_prediksi()
def ke_json_lengkap(kot, lebar, tinggi):
    """ke_json ditambah penambalan agregasi V6. Inilah yang dipakai server.

    ke_json() yang polos tetap dipertahankan, karena dipakai menyusun jawaban
    gold saat mengukur. Jangan disatukan.
    """
    h = ke_json(kot, lebar, tinggi)
    idb = kelompok_baris(kot)
    h = _satukan_potongan(kot, idb, h)
    h = _cadangan_aturan(kot, lebar, tinggi, h)
    h = _cadangan_bentuk(kot, h)
    h = _cadangan_kata(kot, idb, h)
    h = _cadangan_pajak_sebaris(kot, idb, h)
    return _jaga_wajar(_pindah_nontunai(kot, idb, h))
