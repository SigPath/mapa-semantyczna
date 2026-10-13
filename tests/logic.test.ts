import test from 'node:test';
import assert from 'node:assert/strict';
import { dot, cosine, normalize, pca2, kmeans, topSimilar } from '../src/logic.ts';

test('dot i cosine: ortogonalne vs równe kierunki', () => {
  const a = [1, 0], b = [0, 1], c = [2, 0];
  assert.equal(dot(a, b), 0);
  assert.ok(Math.abs(cosine(a, c) - 1) < 1e-12);
  assert.equal(cosine(a, b), 0);
});

test('normalize: długość 1 po normalizacji', () => {
  const v = normalize([3, 4]);
  assert.ok(Math.abs(Math.sqrt(dot(v, v)) - 1) < 1e-12);
});

test('pca2: dane wzdłuż jednej osi mapują się na oś X', () => {
  const data = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [-1, 0, 0]];
  const pts = pca2(data);
  // Y bliskie zeru, X rosnący z danymi
  for (const [x, y] of pts) assert.ok(Math.abs(y) < 1e-6, `y=${y}`);
  assert.ok(pts[2][0] > pts[0][0]);
});

test('pca2: ortogonalność składowych na danych 2D', () => {
  const data = [[2, 1], [3, 0.5], [1, 1.5], [2.5, 0.8], [1.8, 1.2]];
  const pts = pca2(data);
  assert.equal(pts.length, data.length);
  for (const p of pts) assert.ok(Number.isFinite(p[0]) && Number.isFinite(p[1]));
});

test('kmeans: dwa oddalone klastry są rozdzielone', () => {
  const points: Array<[number, number]> = [
    [0, 0], [0.2, 0.1], [0.1, 0.2],
    [10, 10], [10.2, 10.1], [10.1, 10.2],
  ];
  const labels = kmeans(points, 2);
  const first = labels[0];
  for (let i = 0; i < 3; i++) assert.equal(labels[i], first);
  for (let i = 3; i < 6; i++) assert.notEqual(labels[i], first);
});

test('topSimilar: najbliższy wektor na pierwszym miejscu, self nie liczy się', () => {
  const query = [1, 0];
  const corpus = [[0, 1], [0.9, 0.1], [1, 0.01]];
  const top = topSimilar(query, corpus, 2);
  assert.equal(top.length, 2);
  assert.equal(top[0].index, 2);
  assert.ok(top[0].score > top[1].score);
});
