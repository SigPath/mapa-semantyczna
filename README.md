# Mapa semantyczna

**Embeddingi działające w 100% w przeglądarce.** Wpisujesz zdania po polsku → model
(multilingual-e5-small, ONNX) liczony jest lokalnie w Twojej przeglądarce → zdania
lądują na mapie 2D pogrupowane semantycznie. Zero serwera, zero kluczy, zero
wysyłanych danych.

→ **Demo:** https://sigpath.github.io/mapa-semantyczna/ (pierwsze uruchomienie pobiera model ~30 MB, potem cache)

## Co się dzieje pod maską

1. **Embeddingi**: transformers.js + `Xenova/multilingual-e5-small` (kwantyzacja q8),
   pooling mean, normalizacja L2 — biblioteka ładuje się leniwie, dopiero na kliknięcie,
2. **PCA → 2D**: własna implementacja (metoda potęgowa + ortogonalizacja
   Grama–Schmidta), bez bibliotek numerycznych — `src/logic.ts`,
3. **k-means**: własny, deterministyczna inicjalizacja kwartylami,
4. **szukaj po podobieństwie**: kosinus do wszystkich zdań, top-5 z procentami.

## Testy

```bash
npm test   # 6 testów: dot/cosine, normalizacja, PCA (oś + ortogonalność), k-means, topSimilar
```

Matematyka jest w pełni odizolowana od DOM (`src/logic.ts`) i testowana w Node
przez `node --test`.

## Uczciwe ograniczenia

- jakość mapy zależy od modelu — e5-small to najmniejszy sensowny model wielojęzyczny,
- PCA to przybliżenie: odległości na mapie są orientacyjne,
- pierwsze mapowanie liczy się kilka sekund (WASM); na kolejnych jest szybciej.

## Licencja

MIT
