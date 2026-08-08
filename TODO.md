# RetailRocket Integration Plan

- [x] 1. Create `src/retail_rocket.py` (loader: column mapping + session derivation + popularity ranking)
- [x] 2. Update `src/cold_start.py` to use real popularity ranking
- [x] 3. Update `src/serve.py` to use real popularity ranking + accept raw item IDs via vocab
- [x] 4. Add `--retail` flag to `src/train.py`
- [x] 5. Add `--retail` flag to `src/evaluate.py`
- [x] 6. Add `--retail` flag to `src/inference.py`
- [x] 7. Run RetailRocket pipeline on GPU (train → evaluate → build index → serve)
      - Train: loss 10.33 → 8.15 (3 epochs, RTX 4060)
      - Evaluate: Recall@10=0.3102, MRR@10=0.2133, NDCG@10=0.2365, +3374% vs popularity baseline
      - FAISS: index built, inference 6.89ms
      - Serve: /recommend works via faiss (cache hit 1.0ms), real popularity fallback, raw item-ID mapping via vocab

# Deep Experiment & Analysis Plan

- [ ] A. Verify data leakage (session-level split; no cross-split session overlap)
- [ ] B. Verify train/val/test split methodology (counts + proportions + seed stability)
- [ ] C. Compare additional baselines (random, popularity, item-KNN, MostPopLastItem)
- [ ] D. Run ablation experiments (GRU4REC vs AttentionRNN vs ContextAwareRNN vs TwoTower on subsampled data)
- [ ] E. Measure inference latency without cache (fresh embeddings, p50/p95/p99)
- [ ] F. Analyze Recall@K/MRR@K/NDCG@K by session (input) length
- [ ] G. Analyze cold-start performance (short sessions)
- [ ] H. Analyze popular vs rare product performance (popularity bias)
- [ ] I. Inspect actual recommendation examples (session + top-k + popularity + category)
- [ ] J. Document all results in experiments/report.md
