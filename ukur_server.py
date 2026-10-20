"""Ukur ketepatan server sungguhan pada 16 struk uji.

Bedanya dengan Notebook E sel 12: rantai di server memakai penengah hibrida,
sedangkan notebook hanya memakai model. Jadi skrip ini menjawab apakah
penengah menaikkan ketepatan, dan sekaligus menguji server di bawah beban
16 struk berturut-turut, termasuk struk terbesar yang 150 kotak.

Jalankan:
    python ukur_server.py
    python ukur_server.py --server https://... --kunci RAHASIA
"""

import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rantai  # noqa: E402  (ke_json dan urut_baca yang sama dengan notebook)

AKAR = r"C:/Users/user/Documents/muk/documents/Kuliah/SKRIPSI/dataset/skripsi-struk/upload_gold_v15"
SERVER = "https://modelsumma-production.up.railway.app"

MEDAN_ANGKA = ["subtotal", "pajak", "total", "tunai", "kembalian"]
MEDAN_TEKS = ["toko", "tanggal", "jam"]


def json_dari_gold(d):
    """Jawaban benar, disusun dari label gold. Sama dengan Notebook E sel 12."""
    kot, _ = rantai.urut_baca([dict(b) for b in d["kotak"]])
    for b in kot:
        b["_label"] = b["label"]
        b["_yakin"] = 1.0
    return rantai.ke_json(kot, d["lebar"], d["tinggi"])


def samakan_teks(s):
    if s is None:
        return None
    return re.sub(r"[^A-Z0-9]", "", str(s).upper()) or None


def kirim(server, kunci, jalur):
    """Kirim satu foto sebagai multipart, kembalikan JSON balasan."""
    batas = "----batas" + os.urandom(8).hex()
    with open(jalur, "rb") as f:
        isi = f.read()
    badan = (
        f"--{batas}\r\n"
        f'Content-Disposition: form-data; name="berkas"; filename="{os.path.basename(jalur)}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + isi + f"\r\n--{batas}--\r\n".encode()

    req = urllib.request.Request(server.rstrip("/") + "/ekstrak", data=badan)
    req.add_header("Content-Type", f"multipart/form-data; boundary={batas}")
    if kunci:
        req.add_header("X-Kunci", kunci)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default=SERVER)
    p.add_argument("--kunci", default=os.environ.get("KUNCI", ""))
    p.add_argument("--akar", default=AKAR)
    a = p.parse_args()

    gold = {}
    dir_label = os.path.join(a.akar, "labels")
    for f in sorted(os.listdir(dir_label)):
        if f.endswith(".json"):
            d = json.load(open(os.path.join(dir_label, f), encoding="utf-8"))
            gold[d["id"]] = d
    UJI = sorted(k for k, d in gold.items() if d["peran"] == "uji")
    assert len(UJI) == 16, f"struk uji harusnya 16, terbaca {len(UJI)}"

    benar = collections.Counter()
    ada = collections.Counter()
    meleset = collections.defaultdict(list)
    waktu = []

    print(f"{'id':26s} {'detik':>7} {'kotak':>6} {'total sistem':>14}")
    for k in UJI:
        d = gold[k]
        jalur = os.path.join(a.akar, d.get("nama_file_disk") or d["nama_file"])
        t0 = time.perf_counter()
        try:
            r = kirim(a.server, a.kunci, jalur)
        except Exception as e:
            print(f"{k:26s}   GAGAL  {type(e).__name__}: {str(e)[:60]}")
            continue
        dtk = time.perf_counter() - t0
        waktu.append(dtk)
        pred = r["mentah"]
        tot = (pred.get("total") or {}).get("nilai")
        print(f"{k:26s} {dtk:7.2f} {r['waktu']['n_kotak']:6d} {str(tot):>14}")

        emas = json_dari_gold(d)
        for m in MEDAN_ANGKA:
            ge = (emas.get(m) or {}).get("nilai")
            pe = (pred.get(m) or {}).get("nilai")
            if ge is None:
                continue
            ada[m] += 1
            if pe is not None and abs(ge - pe) <= 0.01:
                benar[m] += 1
            else:
                meleset[m].append((k, ge, pe))
        for m in MEDAN_TEKS:
            g = emas.get(m)
            ge = samakan_teks(g if isinstance(g, str) else (g or {}).get("teks"))
            q = pred.get(m)
            pe = samakan_teks(q if isinstance(q, str) else (q or {}).get("teks"))
            if ge is None:
                continue
            ada[m] += 1
            if pe is not None and (ge == pe or ge in pe or pe in ge):
                benar[m] += 1
            else:
                meleset[m].append((k, ge, pe))

    print("\n" + "=" * 62)
    print("KETEPATAN SERVER, 16 STRUK UJI, RANTAI LENGKAP DENGAN PENENGAH")
    print("=" * 62)
    print("%-14s %8s %8s %11s" % ("medan", "ada", "benar", "ketepatan"))
    for m in MEDAN_TEKS + MEDAN_ANGKA:
        if not ada[m]:
            print("%-14s %8d %8s %11s" % (m, 0, "-", "tidak ada"))
            continue
        print("%-14s %8d %8d %10.1f%%" % (m, ada[m], benar[m], 100 * benar[m] / ada[m]))
    tb, ta = sum(benar.values()), sum(ada.values())
    print("%-14s %8d %8d %10.1f%%" % ("GABUNGAN", ta, tb, 100 * tb / max(1, ta)))
    print("\nPembanding Notebook E sel 12, tanpa penengah : 30.7%")

    if waktu:
        w = sorted(waktu)
        print("\nwaktu bolak-balik termasuk jaringan:")
        print("  median %.2f dtk | min %.2f | maks %.2f" % (w[len(w) // 2], w[0], w[-1]))


if __name__ == "__main__":
    main()
