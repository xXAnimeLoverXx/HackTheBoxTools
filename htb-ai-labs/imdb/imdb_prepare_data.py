#!/usr/bin/env python3
import os, argparse, glob
from datasets import load_dataset

def ensure_dirs(base):
    for split in ("train","test"):
        for lab in ("pos","neg"):
            os.makedirs(f"{base}/{split}/{lab}", exist_ok=True)

def count(dirpath):
    return len(glob.glob(os.path.join(dirpath, "*.txt")))

def fill_split(base, split, target_pos, target_neg):
    pos_dir = f"{base}/{split}/pos"; neg_dir = f"{base}/{split}/neg"
    cur_pos, cur_neg = count(pos_dir), count(neg_dir)
    need_pos, need_neg = max(0, target_pos-cur_pos), max(0, target_neg-cur_neg)
    if need_pos==0 and need_neg==0:
        print(f"[{split}] ya completo: pos={cur_pos}, neg={cur_neg}"); return
    print(f"[{split}] actuales: pos={cur_pos}, neg={cur_neg} | creando: pos={need_pos}, neg={need_neg}")
    ds = load_dataset("imdb", split=split, streaming=True)
    got_pos = got_neg = 0
    for ex in ds:
        lab = "pos" if ex["label"]==1 else "neg"
        if lab=="pos" and got_pos>=need_pos: continue
        if lab=="neg" and got_neg>=need_neg: continue
        idx = (cur_pos+got_pos) if lab=="pos" else (cur_neg+got_neg)
        out = f'{base}/{split}/{lab}/{idx:06d}.txt'
        txt = ex["text"].replace("<br />"," ").replace("<br/>"," ")
        with open(out, "w", encoding="utf-8") as f: f.write(txt)
        if lab=="pos": got_pos+=1
        else: got_neg+=1
        if got_pos>=need_pos and got_neg>=need_neg: break
    print(f"[{split}] finales: pos={count(pos_dir)}, neg={count(neg_dir)}")

def main():
    ap = argparse.ArgumentParser(description="Crear estructura aclImdb desde HF por streaming")
    ap.add_argument("--base", default="aclImdb", help="Carpeta raíz")
    ap.add_argument("--train-pos", type=int, default=6000)
    ap.add_argument("--train-neg", type=int, default=6000)
    ap.add_argument("--test-pos",  type=int, default=2500)
    ap.add_argument("--test-neg",  type=int, default=2500)
    a = ap.parse_args()
    ensure_dirs(a.base)
    fill_split(a.base, "train", a.train_pos, a.train_neg)
    fill_split(a.base, "test",  a.test_pos,  a.test_neg)
    print(f"[✓] Listo en ./{a.base}")

if __name__ == "__main__":
    main()
