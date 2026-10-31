"""Fungsi pra-pemrosesan dan kontrak JSON.

Bagian atas berkas ini disalin APA ADANYA dari Notebook E. Jangan dirapikan,
jangan diganti namanya. Kalau isinya berbeda sedikit saja, model menerima
masukan yang tidak pernah dia lihat saat latih, dan angka di BAB IV tidak
lagi berlaku untuk server ini.

Berkas ini sengaja tidak mengimpor torch maupun easyocr, supaya bisa diuji
tanpa memuat model. Lihat uji.py.
"""

import collections
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
    s = s.replace("O", "0").replace("I", "1").replace("L", "1").replace(",", "")
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
    t = " ".join(b["teks"] for b in bar if b["_label"] == label)
    return _uang(t) if t.strip() else None


# Garis bawah ikut, sebab pengenal huruf memberi "10 _" untuk "10.". Tanpa itu
# potongan kiri sebuah harga ditolak di sini dan tidak pernah disatukan lagi.
_TANDA_ANGKA = set("0123456789.,-_")


def _mirip_uang(teks):
    """Kotak yang isinya hanya angka dan tanda baca angka, boleh berlambang RM."""
    s = (teks or "").strip().upper()
    for buang in ("RM", "MYR", "$", " "):
        s = s.replace(buang, "")
    return bool(s) and any(c.isdigit() for c in s) and all(c in _TANDA_ANGKA for c in s)


def _harga_baris(bar):
    """Harga satu baris item, diambil dari kotak angka di kanan nama barang.

    Ini aturan LETAK, bukan keluaran model. Alasannya terukur: pada struk 99
    Speed Mart, CRAFT memotong "10.90" di titik desimalnya, dan model menolak
    melabeli potongan "10 ." karena bentuk begitu tidak pernah dilihatnya saat
    latih. Dari enam harga di struk itu hanya satu yang tidak terpotong, dan
    hanya yang satu itu yang berhasil dilabeli menu.price.

    Tiga pengaman, karena aturan letak gampang salah tangkap:

    1. Hanya kotak di sebelah KANAN nama barang. Kolom kode barang (8049,
       5108) juga angka dan sebaris, tetapi letaknya di kiri nama.
    2. Potongan disatukan berdasar kedekatan mendatar, bukan asal gabung.
       Kolom kuantitas letaknya jauh, jadi tidak ikut tertelan. Tanpa ini,
       "2" dan "10.90" menyatu jadi 210,90.
    3. Gugus yang memuat kotak menu.price dimenangkan. Model tetap didahulukan
       kalau ia bersuara; letak hanya dipakai saat model diam.

    Hanya menyentuh daftar item. Medan ringkasan tidak lewat sini, jadi angka
    ketepatan di BAB IV tidak berubah.

    bar harus sudah urut kiri ke kanan.
    """
    nama = [b for b in bar if b["_label"] == "menu.nm"]
    if not nama:
        return _angka_baris(bar, "menu.price")
    batas = max(b["bbox"][2] for b in nama)
    calon = [b for b in bar if b["bbox"][0] >= batas
             and b["_label"] in ("O", "menu.price") and _mirip_uang(b["teks"])]
    if not calon:
        return _angka_baris(bar, "menu.price")

    tin = sorted(max(1, b["bbox"][3] - b["bbox"][1]) for b in bar)
    ambang = 0.6 * tin[len(tin) // 2]
    gugus, kini = [], [calon[0]]
    for b in calon[1:]:
        if b["bbox"][0] - kini[-1]["bbox"][2] <= ambang:
            kini.append(b)
        else:
            gugus.append(kini)
            kini = [b]
    gugus.append(kini)

    pilih = next((g for g in gugus if any(b["_label"] == "menu.price" for b in g)), gugus[-1])
    return _uang(" ".join(b["teks"] for b in pilih))


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
        nm = " ".join(b["teks"] for b in bar if b["_label"] == "menu.nm")
        if not (nm or any(b["_label"] == "menu.price" for b in bar)): continue
        jml = _angka_baris(bar, "menu.cnt")
        item.append({
            "nama": nm or None,
            # Struk tanpa kolom kuantitas, seperti 99 Speed Mart, berarti satu
            # per barang. None membuat aplikasi menampilkan 0, yang salah.
            "jumlah": 1.0 if jml is None else jml,
            "harga_satuan": _angka_baris(bar, "menu.unitprice"),
            "harga": _harga_baris(bar),
            "diskon": _angka_baris(bar, "menu.discountprice"),
            "yakin": round(min([b["_yakin"] for b in bar if b["_label"] != "O"] or [0.0]), 4),
        })

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
        jml = it["jumlah"]
        item.append({
            "id": f"i{n}",
            "nama": it["nama"] or "",
            "jumlah": int(jml) if jml is not None else None,
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
_KATA_MEDAN = {"subtotal": ("SUBT", "AMOUNT"),
               "pajak": ("GST", "SST", "TAX", "CUKAI")}


def _uang_potongan(potong):
    """Angkakan potongan satu bilangan, tahan titik desimal yang tidak terbaca.

    "2 ." + "05" jadi 2.05, titiknya terbaca dan _uang sudah cukup.
    "56" + "95"  jadi 56.95, titiknya hilang lalu dipulihkan dari angka ekor.

    Pemulihan ekor hanya saat ada dua potongan atau lebih dan tidak satu pun
    memuat titik atau koma, supaya bilangan utuh "170" tidak jadi 1,70.
    """
    teks = " ".join(p for p in potong if p)
    v = _uang(teks)
    if len(potong) >= 2 and not any(c in ".," for c in teks):
        ekor = (potong[-1] or "").strip()
        if len(ekor) == 2 and ekor.isdigit():
            kepala = _uang(" ".join(potong[:-1]))
            if kepala is not None:
                return kepala + int(ekor) / 100.0
    return v


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


def _gugus_kanan(sebaris, ambang):
    """Gugus angka paling kanan pada satu baris, potongannya sudah disatukan."""
    calon = [b for b in sebaris if _mirip_uang(b["teks"])]
    if not calon:
        return None
    gugus, kini = [], [calon[0]]
    for b in calon[1:]:
        if b["bbox"][0] - kini[-1]["bbox"][2] <= ambang:
            kini.append(b)
        else:
            gugus.append(kini)
            kini = [b]
    gugus.append(kini)
    return gugus[-1]


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
    for medan, kata in _KATA_MEDAN.items():
        if (h.get(medan) or {}).get("nilai") is not None:
            continue
        for r in sorted(baris):
            if not any(w in huruf[r] for w in kata):
                continue
            pilih = _gugus_kanan(baris[r], _ambang_baris(baris[r]))
            if pilih and _uang_potongan([b["teks"] for b in pilih]) is not None:
                _isi(h, medan, pilih)
                break
    return h


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
    return _cadangan_kata(kot, idb, h)
