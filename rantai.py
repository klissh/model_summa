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
    s = s.replace("O", "0").replace("I", "1").replace("L", "1").replace(",", "")
    s = _re.sub(r"[^0-9.\-]", "", s)
    neg = s.endswith("-") or s.startswith("-")
    s = s.strip("-")
    if s.count(".") > 1:
        i = s.rfind("."); s = s[:i].replace(".", "") + s[i:]
    if s in ("", "."): return None
    if s.startswith("."): s = "0" + s
    try: v = float(s)
    except ValueError: return None
    if abs(v) > 100000: return None
    return -v if neg else v


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


def ke_json(kot, lebar, tinggi):
    """Kontrak JSON untuk aplikasi. Setiap nilai membawa keyakinannya sendiri."""
    def satu(label, angka=False):
        e = _gabung(kot, label)
        if not e: return None
        e = max(e, key=lambda z: z["yakin"])
        nilai = _uang(e["teks"]) if angka else e["teks"]
        return {"teks": e["teks"], "nilai": nilai, "yakin": e["yakin"], "bbox": e["bbox"]}

    # item disusun per baris, karena satu baris struk adalah satu item
    _, id_baris = urut_baca(kot)
    per_baris = collections.defaultdict(list)
    for b, r in zip(kot, id_baris):
        per_baris[r].append(b)
    item = []
    for r in sorted(per_baris):
        bar = per_baris[r]
        nm = " ".join(b["teks"] for b in bar if b["_label"] == "menu.nm")
        hr = [b for b in bar if b["_label"] == "menu.price"]
        if not (nm or hr): continue
        cn = [b for b in bar if b["_label"] == "menu.cnt"]
        up = [b for b in bar if b["_label"] == "menu.unitprice"]
        dk = [b for b in bar if b["_label"] == "menu.discountprice"]
        if not hr and not nm: continue
        item.append({
            "nama": nm or None,
            "jumlah": _uang(cn[0]["teks"]) if cn else None,
            "harga_satuan": _uang(up[0]["teks"]) if up else None,
            "harga": _uang(hr[-1]["teks"]) if hr else None,
            "diskon": _uang(dk[0]["teks"]) if dk else None,
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
