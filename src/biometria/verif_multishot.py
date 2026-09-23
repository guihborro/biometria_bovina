"""Verificacao multi-foto: o cenario real de campo.

A avaliacao padrao compara UMA foto contra UMA foto — o cenario mais severo.
No brete voce cadastra 3-5 fotos por animal e compara a nova contra TODAS,
ficando com a melhor similaridade. Isto ataca diretamente a cauda de
genuinos ruins (foto borrada/angulo ruim) que segura o FAR la em cima.

Compara, no MESMO split de teste (animais nunca vistos):
  - 1 foto cadastrada (protocolo severo, o que medimos ate agora)
  - N fotos cadastradas (protocolo real de campo)

Uso: python -m biometria.verif_multishot
"""
from __future__ import annotations
import pathlib
import random
import numpy as np
import torch
from torchvision import transforms

from biometria.finetune_arcface import MuzzleNet, Frozen, DS, load_animals, DEVICE
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
CKPT = ROOT / "data" / "interim" / "muzzlenet_arcface.pt"

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def embed(model, items):
    model.eval()
    vs, ls = [], []
    for x, y in DataLoader(DS(items, EVAL_TF), batch_size=64):
        vs.append(model(x.to(DEVICE)).cpu().numpy()); ls.append(y.numpy())
    e = np.vstack(vs).astype("float32")
    e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    return e, np.concatenate(ls)


def verify(emb, labels, n_enroll, n_pairs=20000, seed=42):
    """n_enroll fotos cadastradas por animal; probe = outra foto.
    Similaridade = MELHOR match contra as cadastradas."""
    rng = np.random.default_rng(seed)
    by = {l: np.where(labels == l)[0] for l in np.unique(labels)}
    usable = [l for l, ix in by.items() if len(ix) >= n_enroll + 1]

    gen, imp = [], []
    for _ in range(n_pairs):
        l = usable[rng.integers(len(usable))]
        pick = rng.choice(by[l], n_enroll + 1, replace=False)
        enroll, probe = pick[:n_enroll], pick[n_enroll]
        gen.append(float(np.max(emb[enroll] @ emb[probe])))

        l2 = usable[rng.integers(len(usable))]
        while l2 == l:
            l2 = usable[rng.integers(len(usable))]
        enroll2 = rng.choice(by[l2], n_enroll, replace=False)
        imp.append(float(np.max(emb[enroll2] @ emb[probe])))

    gen, imp = np.array(gen), np.array(imp)
    ths = np.linspace(-1, 1, 600)
    far = np.array([(imp >= t).mean() for t in ths])
    frr = np.array([(gen < t).mean() for t in ths])
    eer = float((far[np.argmin(np.abs(far - frr))] + frr[np.argmin(np.abs(far - frr))]) / 2)
    return dict(eer=eer,
                far1=float(far[np.argmin(np.abs(frr - 0.01))]),
                far5=float(far[np.argmin(np.abs(frr - 0.05))]),
                gen=gen.mean(), imp=imp.mean())


def main() -> None:
    animals = load_animals()
    random.Random(42).shuffle(animals)          # MESMO split do treino
    test_a = animals[:max(1, int(0.25 * len(animals)))]
    items = [(p, i) for i, (_, ps) in enumerate(test_a) for p in ps]
    print(f"{len(test_a)} animais de teste (nunca vistos), {len(items)} imagens\n")

    net = MuzzleNet(512).to(DEVICE)
    net.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    print("embeddings fine-tuned...", flush=True)
    emb, lab = embed(net, items)
    print("embeddings congelados...", flush=True)
    embf, _ = embed(Frozen().to(DEVICE), items)

    print(f"\n{'protocolo':<26}{'EER':>8}{'FAR@FRR1%':>12}{'FAR@FRR5%':>12}")
    print("-" * 58)
    for tag, e in [("congelada", embf), ("fine-tuned", emb)]:
        for k in (1, 3, 5):
            r = verify(e, lab, k)
            print(f"{tag+' — '+str(k)+' foto(s)':<26}"
                  f"{r['eer']*100:7.1f}%{r['far1']*100:11.1f}%{r['far5']*100:11.1f}%")
    print("\n1 foto = protocolo severo | 3-5 fotos = cadastro real no brete")


if __name__ == "__main__":
    main()
