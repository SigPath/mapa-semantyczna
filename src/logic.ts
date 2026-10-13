/**
 * Mapa semantyczna — czysta logika: wektory, PCA, k-means, wyszukiwanie.
 * Zero DOM — wszystko testowalne w Node.
 */

export type Vec = number[];

/** Iloczyn skalarny. */
export function dot(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Podobieństwo kosinusowe (wektory niezerowe). */
export function cosine(a: Vec, b: Vec): number {
  const na = Math.sqrt(dot(a, a));
  const nb = Math.sqrt(dot(b, b));
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

/** Standaryzacja długości (normalizacja L2) — po niej iloczyn skalarny = kosinus. */
export function normalize(v: Vec): Vec {
  const n = Math.sqrt(dot(v, v));
  return n === 0 ? v.slice() : v.map((x) => x / n);
}

/**
 * PCA do 2 wymiarów: iteracyjna potęga (svd-lite bez zależności).
 * Zwraca rzut danych (centrowanych) na dwie główne składowe.
 */
export function pca2(data: Vec[]): Array<[number, number]> {
  if (data.length === 0) return [];
  const dims = data[0].length;
  const mean = new Array(dims).fill(0);
  for (const row of data) for (let i = 0; i < dims; i++) mean[i] += row[i];
  for (let i = 0; i < dims; i++) mean[i] /= data.length;

  const centered = data.map((row) => row.map((x, i) => x - mean[i]));

  const powerComponent = (): Vec => {
    let v = new Array(dims).fill(0).map((_, i) => Math.sin(i + 1) + 0.1);
    for (let iter = 0; iter < 64; iter++) {
      const next = new Array(dims).fill(0);
      for (const row of centered) {
        const proj = dot(row, v);
        for (let i = 0; i < dims; i++) next[i] += proj * row[i];
      }
      const norm = Math.sqrt(dot(next, next));
      if (norm < 1e-12) break;
      v = next.map((x) => x / norm);
    }
    return v;
  };

  const c1 = powerComponent();
  // druga składowa: ortogonalizacja Grama–Schmidta względem pierwszej
  const c2raw = powerComponent();
  const proj = dot(c2raw, c1);
  const c2 = c2raw.map((x, i) => x - proj * c1[i]);
  const n2 = Math.sqrt(dot(c2, c2));
  const c2orth = n2 < 1e-9 ? c2 : c2.map((x) => x / n2);

  return centered.map((row) => [dot(row, c1), dot(row, c2orth)] as [number, number]);
}

/** k-means (dane już znormalizowane/2D); zwraca indeks klastra per punkt. */
export function kmeans(points: Array<[number, number]>, k: number, iterations = 32): number[] {
  if (points.length === 0) return [];
  const kk = Math.max(1, Math.min(k, points.length));
  // inicjalizacja: rozrzut po średnich kwartylach pierwszej osi (deterministycznie)
  const sortedX = [...points].map((p) => p[0]).sort((a, b) => a - b);
  let centroids: Array<[number, number]> = [];
  for (let i = 0; i < kk; i++) {
    const x = sortedX[Math.floor(((i + 0.5) / kk) * sortedX.length)] ?? 0;
    centroids.push([x, 0]);
  }
  let labels = new Array(points.length).fill(0);

  for (let it = 0; it < iterations; it++) {
    labels = points.map((p) => {
      let best = 0;
      let bestD = Infinity;
      centroids.forEach((c, ci) => {
        const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2;
        if (d < bestD) { bestD = d; best = ci; }
      });
      return best;
    });
    const sums = centroids.map(() => [0, 0, 0]);
    points.forEach((p, i) => {
      sums[labels[i]][0] += p[0]; sums[labels[i]][1] += p[1]; sums[labels[i]][2] += 1;
    });
    centroids = sums.map(([sx, sy, n]) => (n ? [sx / n, sy / n] : centroids[0]));
  }
  return labels;
}

/** Indeksy najbardziej podobnych zdań do zapytania (bez samego zapytania). */
export function topSimilar(query: Vec, corpus: Vec[], topK: number): Array<{ index: number; score: number }> {
  return corpus
    .map((v, index) => ({ index, score: cosine(query, v) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

/** Kolory klastrów — kontrastowe, czytelne na jasnym tle. */
export const CLUSTER_COLORS = ['#E8490C', '#2E45D8', '#1E7A3C', '#C7820A', '#7A2ED6', '#0E7C86'];
