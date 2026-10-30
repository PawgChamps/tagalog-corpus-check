# Corpus check of the unverified words (cloud job)

Goal: find which of the dictionary's still-unverified words occur in a large Tagalog web corpus. A word seen in three or
more documents becomes "atestado" (attested: it exists, its meaning is not confirmed).

Files (in `Project TGL to PRT-BR/Scripts/unverified_shards/corpus_check/`):
- `words.tsv` — the list (key, lookup forms, sources), made by `make_list.py <published json>`.
- `count_corpus.py` — streams a Hugging Face corpus and counts whole-word matches; writes `counts.tsv`.
- `apply_counts.py` — folds `counts.tsv` into `shards/unverified_auto.json` (run locally after the job).

Run on a free runner (GitHub Actions, public repository) or any machine with network:
```bash
pip install datasets
python3 count_corpus.py --dataset allenai/c4 --config fil            # mC4 Tagalog, public, ~3 GB of text
# optional second pass, needs HF_TOKEN with access to the gated corpus:
python3 count_corpus.py --dataset oscar-corpus/OSCAR-2301 --config fil --out counts_oscar.tsv
```
Deliver `counts.tsv` (and `counts_oscar.tsv`) back into the same folder; nothing else changes. Do not install anything
on the dictionary machine; the counting runs only in the cloud.
