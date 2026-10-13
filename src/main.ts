import type { FeatureExtractionPipeline } from '@huggingface/transformers';
import { pca2, kmeans, normalize, topSimilar, CLUSTER_COLORS, type Vec } from './logic.ts';
import './style.css';

const $ = <T extends HTMLElement>(s: string) => document.querySelector(s) as T;
const canvas = $('#map') as HTMLCanvasElement;
const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

let extractor: FeatureExtractionPipeline | null = null;
let state: { texts: string[]; vectors: Vec[]; points: Array<[number, number]>; labels: number[] } | null = null;

async function getExtractor(): Promise<FeatureExtractionPipeline> {
  if (!extractor) {
    $('#status').textContent = 'Ładuję transformers.js i model (~30 MB, jednorazowo)…';
    const { pipeline, env } = await import('@huggingface/transformers');
    env.allowLocalModels = false; // model z HF Hub bezpośrednio przez przeglądarkę
    extractor = await pipeline('feature-extraction', 'Xenova/multilingual-e5-small', { dtype: 'q8' });
  }
  return extractor;
}

async function embed(texts: string[]): Promise<Vec[]> {
  const ex = await getExtractor();
  const out = await ex(texts.map((t) => `query: ${t}`), { pooling: 'mean', normalize: true });
  const list = out.tolist() as number[][];
  return list.map(normalize);
}

function draw(): void {
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#ECE8DA';
  ctx.fillRect(0, 0, W, H);
  // siatka
  ctx.strokeStyle = 'rgba(24,22,16,0.07)';
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  if (!state) return;
  const st = state;

  // skala: margin 60 px, punkty z PCA mają rozrzut dowolny
  const xs = st.points.map((p) => p[0]);
  const ys = st.points.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (v: number) => 60 + ((v - minX) / (maxX - minX || 1)) * (W - 120);
  const sy = (v: number) => H - 60 - ((v - minY) / (maxY - minY || 1)) * (H - 120);

  st.points.forEach((p, i) => {
    const color = CLUSTER_COLORS[st.labels[i] % CLUSTER_COLORS.length];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(sx(p[0]), sy(p[1]), 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#181610';
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.fillStyle = '#181610';
    ctx.font = '11px ui-monospace, monospace';
    const label = st.texts[i].slice(0, 26) + (st.texts[i].length > 26 ? '…' : '');
    const tw = ctx.measureText(label).width;
    const px = Math.min(sx(p[0]) + 11, W - tw - 6);
    ctx.fillText(label, px, sy(p[1]) + 4);
  });
}

async function run(): Promise<void> {
  const btn = $('#run') as HTMLButtonElement;
  btn.disabled = true;
  try {
    const texts = ($('#input') as HTMLTextAreaElement).value
      .split('\n').map((s) => s.trim()).filter(Boolean).slice(0, 80);
    if (texts.length < 3) {
      $('#status').textContent = 'Podaj co najmniej 3 zdania.';
      return;
    }
    $('#status').textContent = 'Embeddingi liczą się w Twojej przeglądarce…';
    const vectors = await embed(texts);
    const points = pca2(vectors);
    const k = Math.max(2, Math.min(6, Number(($('#k') as HTMLInputElement).value) || 3));
    const labels = kmeans(points, k);

    state = { texts, vectors, points, labels };
    draw();

    const legend = $('#legend');
    legend.innerHTML = '';
    labels.forEach((lab) => {
      if (legend.querySelector(`[data-k="${lab}"]`)) return;
      const chip = document.createElement('span');
      chip.dataset.k = String(lab);
      chip.innerHTML = `<i style="background:${CLUSTER_COLORS[lab % CLUSTER_COLORS.length]}"></i> klaster ${lab + 1}`;
      legend.appendChild(chip);
    });
    const clusterCount = new Set(labels).size;
    $('#status').textContent = `Zmapowano ${texts.length} zdań w ${clusterCount} klastrach. Model liczył lokalnie — zero wysyłanych danych.`;
    ($('#search') as HTMLButtonElement).disabled = false;
  } catch (e) {
    $('#status').textContent = `Błąd: ${e instanceof Error ? e.message : String(e)}`;
  } finally {
    btn.disabled = false;
  }
}

async function search(): Promise<void> {
  if (!state) return;
  const q = ($('#query') as HTMLInputElement).value.trim();
  if (!q) return;
  const qv = (await embed([q]))[0];
  const hits = topSimilar(qv, state.vectors, 5);
  const ol = $('#hits');
  ol.innerHTML = '';
  for (const h of hits) {
    const li = document.createElement('li');
    li.textContent = `${state.texts[h.index]} — podobieństwo ${(h.score * 100).toFixed(1)}%`;
    ol.appendChild(li);
  }
}

$('#run').addEventListener('click', () => void run());
$('#search').addEventListener('click', () => void search());
draw();
