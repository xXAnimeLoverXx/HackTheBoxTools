#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, random, socket, json
from urllib.parse import urlparse
import joblib, numpy as np, requests
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

def check_url(url):
    u = urlparse(url)
    if not (u.scheme and u.netloc): raise ValueError(f"URL inválida: {url}")
    socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme=='https' else 80))

def set_seed(s=1337):
    random.seed(s); np.random.seed(s)

def read_split(path):
    data,y=[],[]
    for name,label in (("neg",0),("pos",1)):
        d=os.path.join(path,name)
        if not os.path.isdir(d): raise FileNotFoundError(f"No existe {d}")
        for fn in os.listdir(d):
            if not fn.endswith('.txt'): continue
            with open(os.path.join(d,fn),'r',encoding='utf-8',errors='ignore') as f:
                data.append(f.read().replace('<br />',' ').replace('<br/>',' ')); y.append(label)
    return data, np.array(y,dtype=np.int32)

def load_aclImdb(root):
    Xtr,ytr=read_split(os.path.join(root,'train'))
    Xte,yte=read_split(os.path.join(root,'test'))
    return Xtr,ytr,Xte,yte

def build_pipeline(max_features=300_000,C=2.0):
    vectorizer=TfidfVectorizer(lowercase=True, strip_accents='ascii', stop_words='english',
        ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True, max_features=max_features,
        token_pattern=r"(?u)\b\w\w+\b")
    classifier=LogisticRegression(solver='liblinear', C=C, class_weight='balanced', max_iter=2000)
    return Pipeline([("vectorizer",vectorizer),("classifier",classifier)])

def upload_model(url, model_path, connect_timeout=20, read_timeout=900):
    headers={"Expect":"100-continue"}
    with open(model_path,'rb') as f:
        r=requests.post(url, files={"model":f}, headers=headers, timeout=(connect_timeout,read_timeout))
    print(f"[*] HTTP {r.status_code}")
    try: print(json.dumps(r.json(), indent=4))
    except Exception: print(r.text[:2000])
    r.raise_for_status()

def main():
    ap=argparse.ArgumentParser(description='IMDB Sentiment — Portal compat')
    ap.add_argument('--data', required=True)
    ap.add_argument('--upload-url', required=True)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--max-features', type=int, default=300_000)
    ap.add_argument('--C', type=float, default=2.0)
    ap.add_argument('--model-out', default='skills_assessment.joblib')
    a=ap.parse_args(); set_seed(a.seed); check_url(a.upload_url)
    print('[*] Cargando IMDB...'); Xtr,ytr,Xte,yte=load_aclImdb(a.data); print(f"[*] Train: {len(Xtr)} | Test: {len(Xte)}")
    print("[*] Construyendo pipeline (LR con pasos 'vectorizer' y 'classifier')...")
    pipe=build_pipeline(a.max_features,a.C)
    print('[*] Entrenando...'); pipe.fit(Xtr,ytr)
    print('[*] Evaluando...'); yhat=pipe.predict(Xte); acc=accuracy_score(yte,yhat); print(f"[✓] Test Accuracy: {acc*100:.2f}%"); print(classification_report(yte,yhat,digits=4))
    print(f"[*] Guardando modelo en '{a.model_out}' ..."); joblib.dump(pipe,a.model_out,compress=3); print('[✓] Modelo guardado.')
    print(f"[*] Subiendo a '{a.upload_url}' ..."); upload_model(a.upload_url, a.model_out); print('[✓] Hecho.')

if __name__=='__main__':
    try: main()
    except Exception as e: print(f"[X] Error: {e}"); sys.exit(1)
