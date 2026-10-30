#!/usr/bin/env python3
"""Count the words of words.tsv in a large Tagalog web corpus, streamed from Hugging Face (no download to disk):
    python3 count_corpus.py --dataset allenai/c4 --config tl [--max-docs N] [--out counts.tsv]     (mC4 Tagalog, public)
    python3 count_corpus.py --dataset oscar-corpus/OSCAR-2301 --config tl                          (needs HF_TOKEN)
Whole-word, accent-blind, case-blind matches (hyphens kept: "sul-ol" is one token; a phrase of 2-3 words is matched as
a sequence). Writes counts.tsv: key, occurrences, documents, and the total documents and tokens seen. Resumable by
--skip-docs. Needs `pip install datasets`. On a GitHub Actions runner mC4 tl (about 3 GB of text) streams in roughly
an hour."""
import argparse, collections, csv, os, re, sys, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
TOK = re.compile(r"[a-zà-ÿñ'’][a-zà-ÿñ'’\-]*")

def strip(s): return "".join(c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn").lower()

def load_words(path):
    single, phrases = {}, {}
    for row in csv.reader(open(path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
        key, forms = row[0], row[1].split("|")
        for f in forms:
            f = strip(f)
            if " " in f:
                phrases.setdefault(tuple(f.split()), set()).add(key)
            else:
                single.setdefault(f, set()).add(key)
    return single, phrases

def count(docs, single, phrases, max_docs=0, skip=0):
    occ, dcount = collections.Counter(), collections.Counter(); ndocs = ntok = 0
    first = {t[0] for t in phrases}; maxlen = max((len(t) for t in phrases), default=1)
    for i, text in enumerate(docs):
        if i < skip: continue
        if max_docs and ndocs >= max_docs: break
        toks = [strip(t) for t in TOK.findall(text.lower())]
        ndocs += 1; ntok += len(toks); seen = set()
        for j, t in enumerate(toks):
            for key in single.get(t, ()):
                occ[key] += 1; seen.add(key)
            if t in first:
                for n in range(2, maxlen + 1):
                    for key in phrases.get(tuple(toks[j:j + n]), ()):
                        occ[key] += 1; seen.add(key)
        for key in seen: dcount[key] += 1
    return occ, dcount, ndocs, ntok

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="allenai/c4"); ap.add_argument("--config", default="tl")
    ap.add_argument("--split", default="train"); ap.add_argument("--max-docs", type=int, default=0); ap.add_argument("--skip-docs", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "counts.tsv")); ap.add_argument("--words", default=os.path.join(HERE, "words.tsv"))
    a = ap.parse_args()
    single, phrases = load_words(a.words)
    from datasets import load_dataset
    ds = load_dataset(a.dataset, a.config, split=a.split, streaming=True, token=os.environ.get("HF_TOKEN"))
    occ, dcount, ndocs, ntok = count((d.get("text") or "" for d in ds), single, phrases, a.max_docs, a.skip_docs)
    keys = sorted({k for ks in list(single.values()) + list(phrases.values()) for k in ks})
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(f"#dataset\t{a.dataset}/{a.config}\tdocuments\t{ndocs}\ttokens\t{ntok}\n")
        for k in keys:
            fh.write(f"{k}\t{occ[k]}\t{dcount[k]}\n")
    print(f"{ndocs:,} documents, {ntok:,} tokens; {sum(1 for k in keys if dcount[k])} of {len(keys)} words seen -> {a.out}")

if __name__ == "__main__":
    main()
