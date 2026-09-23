"""Avaliacao de re-ID por focinho (Modelo B, baseline com embedding congelado).

Duas metricas que medem coisas diferentes:

  IDENTIFICACAO (closed-set): enrola metade das fotos de cada animal na
  galeria; a outra metade vira probe. Para cada probe, vizinho mais proximo
  por cosseno -> Rank-1/Rank-5/mAP. Mede qualidade do espaco de embedding.

  VERIFICACAO (open-set): 'e o mesmo animal?'. Distribuicao de similaridade
  de pares GENUINOS (mesmo animal) vs IMPOSTORES (animais diferentes) ->
  EER e FAR@FRR=1%. Metrica critica para gado-como-garantia de credito.

Uso:
  python -m biometria.reid_eval --data ../data/interim/muzzle --limit 60
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def scan(root: pathlib.Path) -> tuple[list[pathlib.Path], np.ndarray]:
    """Retorna (paths, labels) — 1 pasta = 1 animal."""
    paths, labels = [], []
    folders = sorted(p for p in root.iterdir()
                     if p.is_dir() and not p.name.startswith("._"))
    for lab, folder in enumerate(folders):
        imgs = [q for q in sorted(folder.rglob("*"))
                if q.suffix.lower() in IMG_EXT and not q.name.startswith("._")]
        for q in imgs:
            paths.append(q); labels.append(lab)
    return paths, np.array(labels)


def identification(emb, labels, seed=42):
    """Split gallery/probe por foto (metade/metade dentro de cada animal)."""
    rng = np.random.default_rng(seed)
    gal, prb = [], []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < 2:
            continue
        rng.shuffle(idx)
        k = len(idx) // 2
        gal.extend(idx[:k]); prb.extend(idx[k:])
    gal, prb = np.array(gal), np.array(prb)
    G, P = emb[gal], emb[prb]
    gl, pl = labels[gal], labels[prb]

    sims = P @ G.T                              # cosseno (ja normalizado)
    order = np.argsort(-sims, axis=1)           # ranking da galeria por probe
    ranked = gl[order]
    correct = ranked == pl[:, None]

    rank1 = correct[:, 0].mean()
    rank5 = correct[:, :5].any(axis=1).mean()
    # mAP
    aps = []
    for i in range(len(prb)):
        rel = correct[i]
        n_rel = rel.sum()
        if n_rel == 0:
            aps.append(0.0); continue
        cum = np.cumsum(rel)
        prec = cum / (np.arange(len(rel)) + 1)
        aps.append((prec * rel).sum() / n_rel)
    return dict(n_gallery=len(gal), n_probe=len(prb),
                rank1=rank1, rank5=rank5, mAP=float(np.mean(aps)))


def verification(emb, labels, n_pairs=20000, seed=42):
    """Amostra pares genuinos/impostores -> EER, FAR@FRR=1%."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    by_lab = {l: np.where(labels == l)[0] for l in np.unique(labels)}
    multi = [l for l, ix in by_lab.items() if len(ix) >= 2]

    gen, imp = [], []
    for _ in range(n_pairs):
        l = multi[rng.integers(len(multi))]
        a, b = rng.choice(by_lab[l], 2, replace=False)
        gen.append(emb[a] @ emb[b])
    for _ in range(n_pairs):
        i, j = rng.integers(n), rng.integers(n)
        if labels[i] != labels[j]:
            imp.append(emb[i] @ emb[j])
    gen, imp = np.array(gen), np.array(imp)

    # varre limiar -> FAR (impostor aceito) e FRR (genuino rejeitado)
    ths = np.linspace(-1, 1, 400)
    far = np.array([(imp >= t).mean() for t in ths])
    frr = np.array([(gen < t).mean() for t in ths])
    eer = float((far[np.argmin(np.abs(far - frr))] +
                 frr[np.argmin(np.abs(far - frr))]) / 2)
    # FAR quando FRR ~ 1%
    j = np.argmin(np.abs(frr - 0.01))
    return dict(gen_mean=float(gen.mean()), imp_mean=float(imp.mean()),
                EER=eer, FAR_at_FRR1=float(far[j]), thr_FRR1=float(ths[j]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="usar so os N primeiros animais (rapido)")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    root = pathlib.Path(args.data)
    paths, labels = scan(root)
    if args.limit:
        keep = labels < args.limit
        paths = [p for p, k in zip(paths, keep) if k]
        labels = labels[keep]
    print(f"{len(paths)} imagens de {len(np.unique(labels))} animais")

    from biometria.embed import embed_paths
    cache = pathlib.Path(args.cache) if args.cache else None
    if cache and cache.exists():
        emb = np.load(cache)
        print(f"embeddings do cache {cache.name}")
    else:
        print("extraindo embeddings (ResNet50 ImageNet, CPU)...")
        emb = embed_paths(paths)
        if cache:
            np.save(cache, emb)

    idr = identification(emb, labels)
    ver = verification(emb, labels)
    print("\n=== IDENTIFICACAO (closed-set) ===")
    print(f"  galeria={idr['n_gallery']}  probe={idr['n_probe']}")
    print(f"  Rank-1={idr['rank1']*100:5.1f}%  Rank-5={idr['rank5']*100:5.1f}%"
          f"  mAP={idr['mAP']*100:5.1f}%")
    print("=== VERIFICACAO (open-set) ===")
    print(f"  sim genuino={ver['gen_mean']:.3f}  impostor={ver['imp_mean']:.3f}")
    print(f"  EER={ver['EER']*100:5.1f}%   FAR@FRR=1%={ver['FAR_at_FRR1']*100:5.1f}%")
    print("\n  (baseline sem treino; fine-tuning c/ ArcFace/triplet melhora)")


if __name__ == "__main__":
    main()
