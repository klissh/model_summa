# -*- coding: utf-8 -*-
"""Pembanding berbasis aturan untuk ekstraksi informasi struk.
Masukan sama dengan model: kotak, teks, koordinat. Label diabaikan."""
import json, os, re, sys, collections, difflib

# ---------- utilitas teks ----------
def norm(t):
    return re.sub(r"[^A-Z0-9]", "", str(t).upper())

def mirip(teks, kata, amb=0.78):
    """cocok kalau mengandung, atau mirip secara fuzzy"""
    a, b = norm(teks), norm(kata)
    if not a or not b: return False
    if b in a: return True
    if len(a) < len(b) - 2: return False
    for i in range(0, max(1, len(a) - len(b) + 2)):
        pot = a[i:i + len(b) + 1]
        if difflib.SequenceMatcher(None, pot, b).ratio() >= amb: return True
    return False

def ada_kata(teks, daftar, amb=0.78):
    return any(mirip(teks, k, amb) for k in daftar)

UANG = re.compile(r"-?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}(?!\d)")
def uang(t):
    s = str(t).replace(" ", "")
    m = UANG.search(s)
    if not m:
        m2 = re.fullmatch(r"-?[.,](\d{2})", s.strip("()[]{}"))
        if m2: return float("0." + m2.group(1))
        return None
    v = m.group().replace(",", ".")
    if v.count(".") > 1:
        a = v.rsplit(".", 1)
        v = a[0].replace(".", "") + "." + a[1]
    try: return float(v)
    except: return None

POLA_TGL = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b|\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b")
POLA_JAM = re.compile(r"\b\d{1,2}[:.]\d{2}(?:[:.]\d{2})?\b")

# ---------- pengelompokan baris ----------
def ke_baris(kotak, tol=0.6):
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
            tump = min(y1, cy1) - max(y0, cy0)
            if tump > tol * min(tinggi, max(1, cy1 - cy0)):
                kel.append(c); dipakai.add(id(c))
        kel.sort(key=lambda z: z["bbox"][0])
        baris.append(kel)
    baris.sort(key=lambda k: min(z["bbox"][1] for z in k))
    return baris

def teks_baris(bar):
    return " ".join(str(b["teks"]) for b in bar)

def y_baris(bar):
    return sum(b["bbox"][1] for b in bar) / len(bar)

# ---------- kamus kata kunci ----------
K_TOKO    = ["SDNBHD", "SDN", "BHD", "ENTERPRISE", "TRADING", "RESTAURANT",
             "RESTORAN", "MARKETING", "SUPERMARKET", "MARTSDN", "HOLDINGS"]
K_BUANG   = ["TEL", "FAX", "EMAIL", "ROCNO", "GSTID", "COMPANYNO", "WWW",
             "JALAN", "NO:", "LEVEL", "LOT", "TAMAN", "BANDAR"]
K_KEPALA  = ["INVOICE", "RECEIPT", "CASHBILL", "BILL", "DATE", "CASHIER",
             "SALESPERSON", "DESCRIPTION", "QTY", "ITEM", "AMOUNT", "PRICE",
             "TEL", "EMAIL", "GSTID", "ORD", "REG", "TAXINVOICE"]
K_TOTAL   = ["TOTAL", "NETTOTAL", "GRANDTOTAL", "ROUNDED", "AMOUNTDUE",
             "NETT", "JUMLAH"]
K_SUBTOT  = ["SUBTOTAL", "TOTALSALES", "TOTALEXCLUDE", "TOTALEXCL",
             "TOTALAMOUNT", "TOTALITEM"]
K_PAJAK   = ["GST", "SST", "TAX", "CUKAI"]
K_BULAT   = ["ROUNDAMT", "ROUNDING", "ROUNDADJ", "PEMBULATAN", "ROUNDINGADJ",
             "ADJUSTMENT", "ROUNDOFF", "RNDADJ"]
K_DISKON  = ["DISCOUNT", "DISC", "REBATE", "POTONGAN"]
K_LAYANAN = ["SERVICECHARGE", "SERVICECHG", "SVRCHG", "SERVIS"]
K_TUNAI   = ["CASH", "TUNAI", "TENDERED"]
K_KEMBALI = ["CHANGE", "BAKI", "KEMBALI"]
K_KARTU   = ["VISA", "MASTER", "CARD", "DEBIT", "CREDIT", "ONLINEBANKING",
             "FPX", "DUITNOW", "MAYBANK", "CIMB"]
K_EMONEY  = ["TNG", "TOUCHNGO", "TOUCH", "BOOST", "GRABPAY", "SHOPEEPAY",
             "EWALLET", "WALLET"]
K_QTY     = ["TOTALQTY", "QTY:", "TOTALITEMQTY", "ITEMQTY"]
K_JENIS   = ["TOTALITEM", "ITEMCOUNT", "JENIS"]
K_BAYAR   = K_TUNAI + K_KEMBALI + K_KARTU + K_EMONEY

def punya_uang(bar):
    return any(uang(b["teks"]) is not None for b in bar)

def kotak_uang_kanan(bar):
    kand = [b for b in bar if uang(b["teks"]) is not None]
    if not kand: return None
    return max(kand, key=lambda b: b["bbox"][2])

# ---------- mesin aturan ----------
def prediksi(d):
    kotak = d["kotak"]
    H = max(1, d.get("tinggi") or max(b["bbox"][3] for b in kotak))
    hasil = {id(b): "O" for b in kotak}
    baris = ke_baris(kotak)

    # --- batas wilayah ---
    i_ringkas = None
    for i, bar in enumerate(baris):
        t = teks_baris(bar)
        if not punya_uang(bar): continue
        if ada_kata(t, K_TOTAL + K_SUBTOT + K_PAJAK + K_BULAT + K_DISKON
                       + K_LAYANAN + K_BAYAR):
            if y_baris(bar) > 0.25 * H:
                i_ringkas = i; break
    if i_ringkas is None: i_ringkas = len(baris)

    i_bayar = None
    for i, bar in enumerate(baris):
        if y_baris(bar) < 0.25 * H: continue
        t = teks_baris(bar)
        if punya_uang(bar) and ada_kata(t, K_BAYAR):
            i_bayar = i; break
    if i_bayar is None: i_bayar = len(baris)
    i_ringkas = min(i_ringkas, i_bayar)      # barang tidak boleh melewati blok bayar

    i_item = 0
    for i, bar in enumerate(baris[:i_ringkas]):
        if ada_kata(teks_baris(bar), K_KEPALA): i_item = i + 1

    # --- kepala struk ---
    ket = None
    for bar in baris:
        if y_baris(bar) > 0.35 * H: break
        t = teks_baris(bar)
        if ada_kata(t, K_TOKO) and not ada_kata(t, ["ROCNO", "GSTID"]):
            ket = bar; break
    if ket is None:                       # cadangan: baris teks pertama yang wajar
        for bar in baris:
            if y_baris(bar) > 0.30 * H: break
            t = teks_baris(bar)
            if ada_kata(t, K_BUANG) or uang(t) is not None: continue
            hur = sum(len(re.findall(r"[A-Za-z]", str(b["teks"]))) for b in bar)
            ang = sum(len(re.findall(r"\d", str(b["teks"]))) for b in bar)
            if hur >= 6 and hur > ang:
                ket = bar; break
    if ket is not None:
        for b in ket:
            tb = str(b["teks"])
            if re.search(r"[A-Za-z]", tb) and not tb.strip().startswith("("):
                hasil[id(b)] = "store.nm"

    for b in kotak:
        t = str(b["teks"])
        if hasil[id(b)] != "O": continue
        if ada_kata(t, ["TEL", "FAX", "WWW", "HTTP"]): continue
        if POLA_TGL.search(t) and len(norm(t)) <= 12:
            hasil[id(b)] = "info.date"
        else:
            tt = t.strip().strip("()[],;")
            tt = re.sub(r"(?i)\s*[AP]\.?M\.?$", "", tt).strip()
            m = re.fullmatch(r"(\d{1,2})[:.](\d{2})(?:[:.](\d{2}))?", tt)
            if m and (":" in tt or m.group(3)):
                j = int(m.group(1)); mn = int(m.group(2))
                if j <= 23 and mn <= 59:
                    hasil[id(b)] = "info.time"

    # --- baris barang ---
    STOP = {"RM", "MYR", "X", "@", "SR", "ZR", "S/8", "#"}
    idx_item = [i for i in range(i_item, i_ringkas)
                if y_baris(baris[i]) >= 0.2 * H]
    berharga = {}
    for i in idx_item:
        duit = sorted([b for b in baris[i] if uang(b["teks"]) is not None],
                      key=lambda b: b["bbox"][2])
        if duit: berharga[i] = duit

    for i in idx_item:
        bar = baris[i]
        duit = berharga.get(i)
        if not duit: continue
        hasil[id(duit[-1])] = "menu.price"
        if len(duit) >= 2: hasil[id(duit[-2])] = "menu.unitprice"
        batas = duit[-1]["bbox"][2]
        sisa = [b for b in bar if hasil[id(b)] == "O" and b["bbox"][0] <= batas]
        # kode barang: token paling kiri yang seluruhnya angka
        if sisa:
            kiri = sisa[0]; tk = str(kiri["teks"]).strip()
            if re.fullmatch(r"\d{3,13}", tk):
                hasil[id(kiri)] = "menu.num"
        for b in sisa:
            if hasil[id(b)] != "O": continue
            tb = str(b["teks"]).strip()
            if tb.upper() in STOP: continue
            if re.fullmatch(r"@?\d{1,3}", tb):
                hasil[id(b)] = "menu.cnt"
            else:
                hasil[id(b)] = "menu.nm"

    # baris tanpa harga hanya jadi nama kalau baris berharga di dekatnya tanpa nama
    for i in idx_item:
        if i in berharga: continue
        tetangga = [j for j in (i - 1, i + 1) if j in berharga]
        if not tetangga: continue
        j = tetangga[0]
        punya_nama = any(hasil[id(b)] == "menu.nm" for b in baris[j])
        if punya_nama: continue
        for b in baris[i]:
            if hasil[id(b)] == "O" and str(b["teks"]).strip().upper() not in STOP:
                hasil[id(b)] = "menu.nm"

    # --- blok ringkasan ---
    cal_total, cal_sub, cal_pajak = [], [], []
    for i in range(i_ringkas, i_bayar):
        bar = baris[i]
        if not punya_uang(bar): continue
        t = teks_baris(bar)
        kb = kotak_uang_kanan(bar)
        if kb is None: continue
        v = uang(kb["teks"])
        pajak_ish = ada_kata(t, K_PAJAK) and (
            "@" in t or ada_kata(t, ["INCLUDES", "TOTALGST", "GSTAMOUNT"]))
        if ada_kata(t, K_BULAT) and not ada_kata(t, ["TOTALROUNDED", "ROUNDEDTOTAL"]):
            hasil[id(kb)] = "sub_total.etc"
        elif ada_kata(t, K_LAYANAN):
            hasil[id(kb)] = "sub_total.service_price"
        elif ada_kata(t, K_DISKON):
            hasil[id(kb)] = "sub_total.discount_price"
        elif pajak_ish:
            cal_pajak.append((i, kb, v))
        elif ada_kata(t, K_SUBTOT):
            cal_sub.append((i, kb, v))
        elif ada_kata(t, K_TOTAL):
            cal_total.append((i, kb, v))
        elif ada_kata(t, K_PAJAK):
            cal_pajak.append((i, kb, v))

    total_v = None
    if cal_total:
        i, kb, v = cal_total[-1]            # R1: paling bawah di atas blok bayar
        hasil[id(kb)] = "total.total_price"; total_v = v
    for i, kb, v in cal_pajak:
        hasil[id(kb)] = "sub_total.tax_price"
    pajak_v = sum(v for _, _, v in cal_pajak if v is not None)

    if cal_sub:
        pilih = None
        if total_v is not None:
            for i, kb, v in cal_sub:
                if v is not None and abs(v + pajak_v - total_v) <= 0.11:
                    pilih = (i, kb, v); break
        if pilih is None: pilih = cal_sub[0]
        hasil[id(pilih[1])] = "sub_total.subtotal_price"
        for i, kb, v in cal_sub:
            if kb is not pilih[1] and hasil[id(kb)] == "O":
                hasil[id(kb)] = "O"

    # --- blok pembayaran ---
    for i in range(i_bayar, len(baris)):
        bar = baris[i]
        if not punya_uang(bar): continue
        t = teks_baris(bar)
        kb = kotak_uang_kanan(bar)
        if kb is None or hasil[id(kb)] != "O": continue
        if   ada_kata(t, K_KEMBALI): hasil[id(kb)] = "total.changeprice"
        elif ada_kata(t, K_EMONEY):  hasil[id(kb)] = "total.emoneyprice"
        elif ada_kata(t, K_KARTU):   hasil[id(kb)] = "total.creditcardprice"
        elif ada_kata(t, K_TUNAI):   hasil[id(kb)] = "total.cashprice"

    # --- pajak yang tercetak di bawah blok pembayaran ---
    if not cal_pajak:
        for i in range(i_bayar, len(baris)):
            bar = baris[i]
            if not punya_uang(bar): continue
            t = teks_baris(bar)
            if not ada_kata(t, K_PAJAK): continue
            if ada_kata(t, K_BAYAR): continue
            kb = kotak_uang_kanan(bar)
            if kb is not None and hasil[id(kb)] == "O":
                hasil[id(kb)] = "sub_total.tax_price"
            break

    # --- jumlah unit dan jenis ---
    for i in range(max(0, i_ringkas - 3), i_bayar):
        if i >= len(baris): break
        t = teks_baris(baris[i])
        if ada_kata(t, K_QTY):
            for b in baris[i]:
                if hasil[id(b)] == "O" and re.fullmatch(r"\d{1,3}", str(b["teks"]).strip()):
                    hasil[id(b)] = "total.menuqty_cnt"; break
    return [hasil[id(b)] for b in kotak]

# =====================================================================
#  EVALUASI. Metrik identik dengan sel 9 dan 12 Notebook C.
#  Pemakaian:
#     python pembanding_aturan.py <folder_labels>            -> data DEV
#     python pembanding_aturan.py <folder_labels> --uji-final -> data UJI
#  Aturan di atas HANYA boleh disetel memakai hasil DEV.
# =====================================================================
BLOK = {"kepala":    ["store.nm", "info.date", "info.time"],
        "menu":      "menu.",
        "sub_total": "sub_total.",
        "total":     "total."}

def evaluasi(dirlabel, peran):
    berkas = sorted(f for f in os.listdir(dirlabel) if f.endswith(".json"))
    data = []
    for f in berkas:
        d = json.load(open(os.path.join(dirlabel, f), encoding="utf-8"))
        if d.get("peran") == peran: data.append(d)
    if not data:
        print("tidak ada struk berperan", peran); return

    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    benar = total = 0
    per_struk = []
    for d in data:
        pred = prediksi(d)
        emas = [b["label"] for b in d["kotak"]]
        bs = 0
        for a, b in zip(emas, pred):
            total += 1
            if a == b:
                benar += 1; bs += 1
                if a != "O": tp[a] += 1
            else:
                if a != "O": fn[a] += 1
                if b != "O": fp[b] += 1
        per_struk.append((d["id"], 100.0 * bs / max(1, len(emas))))

    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    P = T / (T + F) if (T + F) else 0.0
    R = T / (T + N) if (T + N) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0

    print("=" * 64)
    print("PEMBANDING BERBASIS ATURAN   peran = %s   struk = %d" % (peran, len(data)))
    print("=" * 64)
    print("  f1_mikro      %.4f" % F1)
    print("  presisi       %.4f" % P)
    print("  recall        %.4f" % R)
    print("  akurasi_kotak %.4f" % (benar / max(1, total)))
    print("  perlu_koreksi %d dari %d" % (total - benar, total))

    label = sorted(set(list(tp) + list(fp) + list(fn)))
    print("\n%-28s %6s %6s %6s %6s" % ("label", "P", "R", "F1", "n"))
    f1s = []
    for L in label:
        n = tp[L] + fn[L]
        p = tp[L] / (tp[L] + fp[L]) if (tp[L] + fp[L]) else 0.0
        r = tp[L] / n if n else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        print("%-28s %6.3f %6.3f %6.3f %6d%s" %
              (L, p, r, f, n, "  <- terlalu sedikit" if n < 5 else ""))
        if n > 0: f1s.append((f, n))

    print("\n%-12s %6s %6s %6s %6s" % ("blok", "P", "R", "F1", "n"))
    for nama, pola in BLOK.items():
        Ls = pola if isinstance(pola, list) else [L for L in label if L.startswith(pola)]
        t_ = sum(tp[L] for L in Ls); f_ = sum(fp[L] for L in Ls); n_ = sum(fn[L] for L in Ls)
        p = t_ / (t_ + f_) if (t_ + f_) else 0.0
        r = t_ / (t_ + n_) if (t_ + n_) else 0.0
        print("%-12s %6.3f %6.3f %6.3f %6d" %
              (nama, p, r, 2 * p * r / (p + r) if (p + r) else 0.0, t_ + n_))

    ada = [f for f, n in f1s]
    cuk = [f for f, n in f1s if n >= 5]
    print("\nF1 makro %2d label : %.4f" % (len(ada), sum(ada) / max(1, len(ada))))
    print("F1 makro %2d label : %.4f   (hanya n >= 5)" % (len(cuk), sum(cuk) / max(1, len(cuk))))

    per_struk.sort(key=lambda x: x[1])
    print("\n5 struk terburuk (persen kotak benar):")
    for i, a in per_struk[:5]: print("   %-26s %.1f%%" % (i, a))

def diagnosa(dirlabel, peran, batas=60):
    berkas = sorted(f for f in os.listdir(dirlabel) if f.endswith(".json"))
    salah = collections.Counter()
    contoh = collections.defaultdict(list)
    for f in berkas:
        d = json.load(open(os.path.join(dirlabel, f), encoding="utf-8"))
        if d.get("peran") != peran: continue
        pred = prediksi(d)
        pet = {}
        for bar in ke_baris(d["kotak"]):
            t = teks_baris(bar)
            for b in bar: pet[id(b)] = t
        for b, p in zip(d["kotak"], pred):
            a = b["label"]
            if a == p: continue
            salah[(a, p)] += 1
            if len(contoh[(a, p)]) < 3:
                contoh[(a, p)].append((d["id"], str(b["teks"])[:18],
                                       pet.get(id(b), "")[:58]))
    print("=" * 78)
    print("DIAGNOSIS kesalahan pada peran =", peran)
    print("=" * 78)
    print("%-26s -> %-26s %5s" % ("emas", "tebakan aturan", "jml"))
    for (a, p), n in salah.most_common(18):
        print("\n%-26s -> %-26s %5d" % (a, p, n))
        for sid, teks, bar in contoh[(a, p)]:
            print("      %-24s [%-18s]  baris: %s" % (sid, teks, bar))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("pakai: python pembanding_aturan.py <folder_labels> [--uji-final]")
        sys.exit(1)
    dirlabel = sys.argv[1]
    if "--salah" in sys.argv:
        diagnosa(dirlabel, "pilot"); sys.exit(0)
    if "--uji-final" in sys.argv:
        print("\n*** MODE UJI FINAL. Setelah ini aturan TIDAK BOLEH diubah lagi. ***\n")
        evaluasi(dirlabel, "uji")
    else:
        evaluasi(dirlabel, "pilot")
