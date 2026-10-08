"""
eksperimen.py - Eksperimen untuk Bab 4 (Matrix Chain Multiplication)

Memanggil fungsi dari tugas.py:
    matrix_chain_order(p) -> (m, s)
    print_parens(s, i, j) -> string pengurungan
    brute_force(p, i, j)  -> biaya minimum (rekursi eksponensial)

Menghasilkan (di folder hasil/):
    1. eksp1_waktu_dp.png / .csv          : waktu DP vs n, dibandingkan kurva c*n^3
    2. eksp2_dp_vs_bruteforce.png / .csv  : DP vs brute force untuk n kecil
    3. eksp3_studi_kasus.png / .csv       : perkalian matriks nyata (NumPy),
                                            urutan naif vs urutan optimal

Cara menjalankan:
    python eksperimen.py          # eksperimen lengkap (beberapa menit)
    python eksperimen.py --cepat  # versi singkat untuk mencoba

Pustaka: numpy (alat ukur perkalian matriks nyata), matplotlib (grafik).
Keduanya bukan inti algoritma; inti algoritma ada di tugas.py.
"""

import csv
import math
import os
import statistics
import sys
import textwrap
import time

# Pasang numpy & matplotlib otomatis jika belum ada, ke Python yang sedang
# menjalankan skrip ini (menghindari masalah beda interpreter di VS Code).
try:
    import matplotlib, numpy  # noqa: F401
except ModuleNotFoundError:
    import subprocess
    print("Memasang numpy dan matplotlib ke:", sys.executable)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib"])

import matplotlib
matplotlib.use("Agg")  # simpan grafik ke file tanpa membuka jendela
import matplotlib.pyplot as plt
import numpy as np

from tugas import matrix_chain_order, print_parens, brute_force

# ---------------------------------------------------------------------------
# Pengaturan eksperimen
# ---------------------------------------------------------------------------
CEPAT = "--cepat" in sys.argv
SEED = 42                 # seed tetap agar hasil bisa diulang
ULANGAN = 5               # pengulangan tiap percobaan (syarat >= 5)
UKURAN_DP = [25, 50, 100, 200] if CEPAT else [50, 100, 200, 400, 800]
UKURAN_BF = list(range(2, 12)) if CEPAT else list(range(2, 16))
DIM_MIN, DIM_MAX = 10, 100
ULANGAN_NUMPY = 20        # eksperimen 3: lebih banyak ulangan karena waktu NumPy berisik
PEMANASAN_NUMPY = 3       # run awal yang dibuang  # rentang dimensi acak untuk eksperimen 1 & 2

FOLDER = "hasil"
os.makedirs(FOLDER, exist_ok=True)


def acak_dimensi(n, rng):
    """Membuat vektor dimensi p (panjang n+1) secara acak."""
    return [int(x) for x in rng.integers(DIM_MIN, DIM_MAX + 1, size=n + 1)]


def ukur(fungsi, ulangan=ULANGAN, pemanasan=0, median=False):
    """Menjalankan fungsi beberapa kali; mengembalikan (rata-rata, simpangan baku) detik.
    pemanasan: jumlah run awal yang tidak diukur (menghindari efek cache/thread BLAS).
    median=True: juga mengembalikan median (tahan terhadap lonjakan sesekali)."""
    for _ in range(pemanasan):
        fungsi()
    waktu = []
    for _ in range(ulangan):
        t0 = time.perf_counter()
        fungsi()
        waktu.append(time.perf_counter() - t0)
    sd = statistics.stdev(waktu) if len(waktu) > 1 else 0.0
    if median:
        return statistics.mean(waktu), sd, statistics.median(waktu)
    return statistics.mean(waktu), sd


def simpan_csv(nama, header, baris):
    with open(os.path.join(FOLDER, nama), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(baris)


def catalan(k):
    """Bilangan Catalan C_k; banyak pengurungan n matriks = C_{n-1}."""
    return math.comb(2 * k, k) // (k + 1)


# ===========================================================================
# EKSPERIMEN 1: waktu DP terhadap n, dibandingkan kurva teoretis c * n^3
# ===========================================================================
def eksperimen_1():
    print("=" * 70)
    print("EKSPERIMEN 1: Waktu DP vs ukuran input n")
    print("=" * 70)
    rng = np.random.default_rng(SEED)
    baris, ns, rata, sd = [], [], [], []

    for n in UKURAN_DP:
        p = acak_dimensi(n, rng)
        mean, std = ukur(lambda: matrix_chain_order(p))
        iterasi = (n ** 3 - n) // 6  # banyak evaluasi q, hasil penurunan formal
        ns.append(n); rata.append(mean); sd.append(std)
        baris.append([n, iterasi, f"{mean:.6f}", f"{std:.6f}", f"{mean / n**3 * 1e9:.3f}"])
        print(f"n = {n:4d} | iterasi (n^3-n)/6 = {iterasi:>12,} | "
              f"waktu = {mean:9.4f} s +/- {std:.4f} | t/n^3 = {mean / n**3 * 1e9:.2f} ns")

    simpan_csv("eksp1_waktu_dp.csv",
               ["n", "iterasi_(n^3-n)/6", "rata2_detik", "simpangan_baku", "t/n^3_(ns)"], baris)

    # Konstanta c untuk kurva c*n^3, dicari dengan kuadrat terkecil
    n_arr, t_arr = np.array(ns, float), np.array(rata)
    c = np.sum(t_arr * n_arr ** 3) / np.sum(n_arr ** 6)
    n_halus = np.linspace(min(ns), max(ns), 200)

    # Kemiringan log-log: untuk Theta(n^3) mendekati 3
    slope = np.polyfit(np.log(n_arr), np.log(t_arr), 1)[0]
    print(f"\nKemiringan log-log = {slope:.3f} (teori: 3)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].errorbar(ns, rata, yerr=sd, fmt="o", capsize=4, label="Pengukuran (rata-rata ± SD)")
    ax[0].plot(n_halus, c * n_halus ** 3, "--", label=f"Teoretis c·n³ (c = {c:.2e})")
    ax[0].set_xlabel("Banyak matriks n"); ax[0].set_ylabel("Waktu (detik)")
    ax[0].set_title("Waktu DP vs kurva n³ (skala linear)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].loglog(ns, rata, "o", label="Pengukuran")
    ax[1].loglog(n_halus, c * n_halus ** 3, "--", label="c·n³")
    ax[1].set_xlabel("n (log)"); ax[1].set_ylabel("Waktu (log)")
    ax[1].set_title(f"Skala log-log, kemiringan = {slope:.2f} (teori 3)")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(FOLDER, "eksp1_waktu_dp.png"), dpi=150)
    plt.close(fig)


# ===========================================================================
# EKSPERIMEN 2: DP vs brute force untuk n kecil
# ===========================================================================
def eksperimen_2():
    print("\n" + "=" * 70)
    print("EKSPERIMEN 2: DP vs brute force (n kecil)")
    print("=" * 70)
    rng = np.random.default_rng(SEED)
    baris, ns, t_dp, t_bf = [], [], [], []

    for n in UKURAN_BF:
        p = acak_dimensi(n, rng)
        m, s = matrix_chain_order(p)
        biaya_bf = brute_force(p, 1, n)
        assert m[1][n] == biaya_bf, "Hasil DP berbeda dengan brute force!"

        mean_dp, sd_dp = ukur(lambda: matrix_chain_order(p))
        mean_bf, sd_bf = ukur(lambda: brute_force(p, 1, n))
        ns.append(n); t_dp.append(mean_dp); t_bf.append(mean_bf)
        baris.append([n, catalan(n - 1), m[1][n], "sama",
                      f"{mean_dp:.6f}", f"{sd_dp:.6f}", f"{mean_bf:.6f}", f"{sd_bf:.6f}"])
        print(f"n = {n:2d} | pengurungan C(n-1) = {catalan(n - 1):>10,} | "
              f"DP = {mean_dp * 1e3:8.3f} ms | brute force = {mean_bf * 1e3:11.3f} ms")

    simpan_csv("eksp2_dp_vs_bruteforce.csv",
               ["n", "banyak_pengurungan_Catalan", "biaya_optimal", "DP_vs_BF",
                "DP_rata2_detik", "DP_SD", "BF_rata2_detik", "BF_SD"], baris)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(ns, t_dp, "o-", label="DP bottom-up Θ(n³)")
    ax.semilogy(ns, t_bf, "s-", label="Brute force (eksponensial)")
    ax.set_xlabel("Banyak matriks n"); ax.set_ylabel("Waktu (detik, skala log)")
    ax.set_title("DP vs brute force")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FOLDER, "eksp2_dp_vs_bruteforce.png"), dpi=150)
    plt.close(fig)


# ===========================================================================
# EKSPERIMEN 3: studi kasus perkalian matriks nyata dengan NumPy
# ===========================================================================
def split_kiri_ke_kanan(n):
    """Tabel s untuk urutan naif kiri-ke-kanan: (((A1A2)A3)...An)."""
    s = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s[i][j] = j - 1
    return s


def split_kanan_ke_kiri(n):
    """Tabel s untuk urutan naif kanan-ke-kiri: A1(A2(...(An-1An)))."""
    s = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s[i][j] = i
    return s


def biaya_urutan(p, s, i, j):
    """Jumlah perkalian skalar untuk pengurungan yang dinyatakan tabel s."""
    if i == j:
        return 0
    k = s[i][j]
    return biaya_urutan(p, s, i, k) + biaya_urutan(p, s, k + 1, j) + p[i - 1] * p[k] * p[j]


def kalikan(mats, s, i, j):
    """Mengalikan matriks sungguhan mengikuti pengurungan pada tabel s."""
    if i == j:
        return mats[i - 1]
    k = s[i][j]
    return kalikan(mats, s, i, k) @ kalikan(mats, s, k + 1, j)


# Skenario studi kasus (p = vektor dimensi, A_i berukuran p[i-1] x p[i])
SKENARIO = [
    ("MLP linear 784-512-256-128-10, batch 1024",
     # W4 (10x128) W3 (128x256) W2 (256x512) W1 (512x784) X (784x1024)
     [10, 128, 256, 512, 784, 1024]),
    ("Rantai Jacobian x vektor (forward-mode)",
     # J1 (1500x1500) J2 (1500x1500) J3 (1500x1500) v (1500x1)
     [1500, 1500, 1500, 1500, 1]),
    ("Rantai heterogen 8 matriks",
     # dimensi bervariasi; urutan optimal berbeda dari kedua urutan naif
     [766, 257, 1284, 20, 362, 340, 823, 564, 251]),
]


def eksperimen_3():
    print("\n" + "=" * 70)
    print("EKSPERIMEN 3: Studi kasus perkalian matriks nyata (NumPy)")
    print("=" * 70)
    rng = np.random.default_rng(SEED)
    baris, label_grafik, data_grafik = [], [], []

    for nama, p in SKENARIO:
        if p is None:
            p = [int(x) for x in rng.integers(20, 1500, size=9)]
        n = len(p) - 1
        if CEPAT:  # perkecil dimensi agar cepat
            p = [max(1, d // 4) for d in p]

        mats = [rng.standard_normal((p[i], p[i + 1])) for i in range(n)]
        _, s_opt = matrix_chain_order(p)
        urutan = {
            "Kiri-ke-kanan": split_kiri_ke_kanan(n),
            "Kanan-ke-kiri": split_kanan_ke_kiri(n),
            "Optimal (DP)": s_opt,
        }

        print(f"\n{nama}\n  p = {p}")
        acuan = kalikan(mats, s_opt, 1, n)
        hasil_skenario = {}
        for label, s in urutan.items():
            flop = biaya_urutan(p, s, 1, n)
            hasil = kalikan(mats, s, 1, n)
            sama = np.allclose(hasil, acuan, rtol=1e-6, atol=1e-6)
            mean, std, med = ukur(lambda: kalikan(mats, s, 1, n), ulangan=ULANGAN_NUMPY,
                                  pemanasan=PEMANASAN_NUMPY, median=True)
            hasil_skenario[label] = (flop, med, mean, std)
            print(f"  {label:14s} {print_parens(s, 1, n):28s} perkalian = {flop:>15,} | "
                  f"median = {med * 1e3:8.2f} ms | rata2 = {mean * 1e3:8.2f} +/- {std * 1e3:.2f} | "
                  f"hasil sama: {sama}")

        f_opt, t_opt = hasil_skenario["Optimal (DP)"][0], hasil_skenario["Optimal (DP)"][1]
        for label, (flop, med, mean, std) in hasil_skenario.items():
            baris.append([nama, label, print_parens(urutan[label], 1, n), flop,
                          f"{flop / f_opt:.2f}", f"{med:.6f}", f"{mean:.6f}", f"{std:.6f}",
                          f"{med / t_opt:.2f}"])
        label_grafik.append(nama)
        data_grafik.append(hasil_skenario)

    simpan_csv("eksp3_studi_kasus.csv",
               ["skenario", "urutan", "pengurungan", "perkalian_skalar",
                "rasio_perkalian_vs_optimal", "median_detik", "rata2_detik", "SD",
                "rasio_median_vs_optimal"], baris)

    # Grafik: perkalian skalar (teori) dan waktu nyata, per skenario
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    labels_urutan = ["Kiri-ke-kanan", "Kanan-ke-kiri", "Optimal (DP)"]
    x = np.arange(len(label_grafik)); lebar = 0.26
    for idx, lu in enumerate(labels_urutan):
        flops = [d[lu][0] for d in data_grafik]
        waktu = [d[lu][1] for d in data_grafik]
        ax[0].bar(x + (idx - 1) * lebar, flops, lebar, label=lu)
        ax[1].bar(x + (idx - 1) * lebar, waktu, lebar, label=lu)
    for a, judul, ylabel in [(ax[0], "Jumlah perkalian skalar (teori)", "Perkalian skalar (log)"),
                             (ax[1], f"Waktu NumPy, median {ULANGAN_NUMPY} ulangan", "Waktu detik (log)")]:
        a.set_yscale("log"); a.set_title(judul); a.set_ylabel(ylabel)
        a.set_xticks(x); a.set_xticklabels([textwrap.fill(l, 20) for l in label_grafik], fontsize=8)
        a.legend(fontsize=8); a.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FOLDER, "eksp3_studi_kasus.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"Mode: {'CEPAT' if CEPAT else 'LENGKAP'} | seed = {SEED} | ulangan = {ULANGAN}\n")
    eksperimen_1()
    eksperimen_2()
    eksperimen_3()
    print(f"\nSelesai. Grafik dan tabel CSV tersimpan di folder '{FOLDER}/'.")