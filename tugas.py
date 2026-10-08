"""
Matrix Chain Multiplication (MCM) - Dynamic Programming bottom-up
Referensi pseudocode: Cormen dkk., Introduction to Algorithms (CLRS), Bab 15.

Input  : p = list dimensi, panjang n+1. Matriks A_i berukuran p[i-1] x p[i].
Output : m[i][j] = biaya minimum (jumlah perkalian skalar) untuk A_i..A_j
         s[i][j] = titik belah k optimal untuk A_i..A_j
Kompleksitas: waktu Theta(n^3), ruang Theta(n^2)
"""

import math


def matrix_chain_order(p):
    """Mengisi tabel m dan s secara bottom-up (berdasarkan panjang rantai)."""
    n = len(p) - 1  # banyaknya matriks

    # Tabel berukuran (n+1) x (n+1) agar indeks 1..n sama dengan pseudocode;
    # baris/kolom 0 tidak dipakai.
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]

    # Kasus dasar: m[i][i] = 0 (satu matriks tidak butuh perkalian).
    # Sudah terpenuhi karena tabel diinisialisasi dengan 0.

    # l = panjang rantai yang sedang dihitung, dari 2 sampai n
    for l in range(2, n + 1):
        # i = indeks awal rantai; rantai A_i..A_j dengan j = i + l - 1
        for i in range(1, n - l + 2):
            j = i + l - 1
            m[i][j] = math.inf  # nilai awal: belum ada solusi
            # k = posisi belah: (A_i..A_k)(A_{k+1}..A_j)
            for k in range(i, j):
                # biaya = biaya kiri + biaya kanan + biaya mengalikan dua hasilnya
                # hasil kiri berukuran p[i-1] x p[k], hasil kanan p[k] x p[j]
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:   # simpan jika lebih murah
                    m[i][j] = q
                    s[i][j] = k
    return m, s


def print_parens(s, i, j):
    """Merekonstruksi pengurungan optimal dari tabel s (rekursif)."""
    if i == j:
        return f"A{i}"
    k = s[i][j]
    return "(" + print_parens(s, i, k) + print_parens(s, k + 1, j) + ")"


def brute_force(p, i, j):
    """Rekursi tanpa memoization: mencoba semua pengurungan (eksponensial).
    Dipakai sebagai pembanding dan untuk memverifikasi hasil DP pada n kecil."""
    if i == j:
        return 0
    best = math.inf
    for k in range(i, j):
        q = brute_force(p, i, k) + brute_force(p, k + 1, j) + p[i - 1] * p[k] * p[j]
        best = min(best, q)
    return best


# ---------------------------------------------------------------------------
# Kasus uji kecil yang bisa dicek manual (syarat minimal 3 kasus)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    kasus_uji = [
        # (nama, p, biaya yang diharapkan)
        ("Dua matriks 10x20, 20x30",          [10, 20, 30],                  6000),
        ("Tiga matriks 10x100, 100x5, 5x50",  [10, 100, 5, 50],              7500),
        ("Contoh CLRS, enam matriks",         [30, 35, 15, 5, 10, 20, 25],   15125),
        ("Satu matriks (kasus tepi)",         [5, 10],                       0),
    ]

    for nama, p, harapan in kasus_uji:
        n = len(p) - 1
        m, s = matrix_chain_order(p)
        biaya = m[1][n]
        bf = brute_force(p, 1, n)
        status = "OK" if biaya == harapan == bf else "SALAH"
        print(f"[{status}] {nama}")
        print(f"    p = {p}")
        print(f"    biaya DP = {biaya}, brute force = {bf}, harapan = {harapan}")
        print(f"    pengurungan optimal: {print_parens(s, 1, n)}\n")