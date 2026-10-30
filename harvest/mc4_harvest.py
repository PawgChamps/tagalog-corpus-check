#!/usr/bin/env python3
"""Harvest candidate example sentences for the dictionary's target words from mC4 Tagalog (allenai/c4, config "fil"),
streamed from Hugging Face. Runs on a GitHub Actions runner (PawgChamps/tagalog-corpus-check); the inputs come from
examples/mc4.py prep: targets.tsv (form<TAB>key|key…), known.txt, natives.txt, english.txt.

    python3 mc4_harvest.py --targets targets.tsv --known known.txt --natives natives.txt --english english.txt \
        --out candidates.jsonl.gz [--max-docs N] [--per-key 5] [--split train]

A sentence is a candidate for a key when it passes the Leipzig filters (5-14 words, ends in . ! ?, no URL / @ / # / long
number, not shouted, at most two capitalised words after the first, 85 % of its words known to the dictionary, no
Taglish: at most one English dictionary word that is not a native headword, none in short sentences) and contains one
of the key's forms whole and not capitalised mid-sentence. Up to --per-key sentences per key are kept, the best by
score (closest to 8 words, fewer capitals, the headword itself before an aspect form). Output: one JSON line per kept
sentence {key, form, tl, score, doc}; a final line {"#stats": …}. Needs `pip install datasets`."""
import argparse
import collections
import gzip
import heapq
import json
import os
import re
import sys
import time
import unicodedata

WORD = re.compile(r"[A-Za-zÀ-ÿñÑ][A-Za-zÀ-ÿñÑ'’\-]*")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý¿¡\"“])")
BAD = re.compile(r"https?://|www\.|@|#|\d{3,}|[<>|_{}\[\]\\/]|[^\x00-\x7FÀ-ÿñÑ‘’“”„«»…–—]")


def strip(s):
    return "".join(c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn").lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True); ap.add_argument("--known", required=True); ap.add_argument("--natives", required=True)
    ap.add_argument("--english", required=True); ap.add_argument("--out", default="candidates.jsonl.gz")
    ap.add_argument("--dataset", default="allenai/c4"); ap.add_argument("--config", default="fil"); ap.add_argument("--split", default="train")
    ap.add_argument("--max-docs", type=int, default=0); ap.add_argument("--per-key", type=int, default=5)
    a = ap.parse_args()
    forms = {}
    for line in open(a.targets, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2:
            forms[p[0]] = p[1].split("|")
    allw = {l.strip() for l in open(a.known, encoding="utf-8") if l.strip()}
    natives = {l.strip() for l in open(a.natives, encoding="utf-8") if l.strip()}
    en_words = {l.strip() for l in open(a.english, encoding="utf-8") if l.strip()}
    keys = {k for v in forms.values() for k in v}
    print(f"targets: {len(keys):,} keys, {len(forms):,} forms; known {len(allw):,}, english {len(en_words):,}", flush=True)
    from datasets import load_dataset
    ds = load_dataset(a.dataset, a.config, split=a.split, streaming=True, token=os.environ.get("HF_TOKEN"))
    best = collections.defaultdict(list); n = 0; docs = sents = kept = 0; t0 = time.time()
    full = 0
    for doc in ds:
        docs += 1
        text = (doc.get("text") or "").replace("­", "")
        for s in SENT.split(text.replace("\n", " ")):
            s = re.sub(r"\s+", " ", s).strip()
            if not s or len(s) > 200:
                continue
            w = s.split()
            if not (5 <= len(w) <= 14) or s[-1] not in ".!?" or BAD.search(s):
                continue
            if sum(c.isupper() for c in s) > len(s) / 5 or s[0].islower() or re.search(r'["“”„«»…]', s) or re.search(r"\b[A-ZÀ-Ý]{3,}\b", s):
                continue
            sents += 1
            raw = WORD.findall(s)
            toks = [strip(x) for x in raw]
            if len(toks) < 5:
                continue
            hit = set()
            for pos_, x in enumerate(toks):
                if x in forms and not (pos_ and raw[pos_][:1].isupper()):
                    for k in forms[x]:
                        hit.add((k, x))
            if not hit:
                continue
            caps = sum(1 for x in w[1:] if x[:1].isupper())
            if caps > 2:
                continue
            known = sum(1 for x in toks if x in allw)
            if known < 0.85 * len(toks):
                continue
            en_t = sum(1 for x in toks if len(x) >= 3 and x in en_words and x not in natives)
            if en_t >= 2 or (en_t == 1 and len(toks) < 8):
                continue
            kept += 1
            for k, x in hit:
                score = abs(len(w) - 8) + caps * 2 + (0 if x == k else 1)
                h = best[k]; n += 1
                if len(h) < a.per_key:
                    heapq.heappush(h, (-score, n, s, x))
                    if len(h) == a.per_key:
                        full += 1
                elif -score > h[0][0]:
                    heapq.heapreplace(h, (-score, n, s, x))
        if docs % 20000 == 0:
            print(f"{docs:,} docs, {sents:,} sentences seen, {kept:,} usable, {len(best):,} keys with a candidate ({full:,} full), "
                  f"{time.time() - t0:.0f}s", flush=True)
        if a.max_docs and docs >= a.max_docs:
            break
    with gzip.open(a.out, "wt", encoding="utf-8") as fh:
        for k, h in best.items():
            for negs, _, s, x in sorted(h, key=lambda t: (-t[0], t[1])):
                fh.write(json.dumps({"key": k, "form": x, "tl": s, "score": -negs}, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"#stats": {"docs": docs, "sentences": sents, "usable": kept, "keys": len(best), "full": full,
                                        "seconds": round(time.time() - t0)}}) + "\n")
    print(f"done: {docs:,} docs, {sents:,} sentences, {kept:,} usable, {len(best):,} of {len(keys):,} keys got a candidate -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
