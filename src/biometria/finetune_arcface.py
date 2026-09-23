"""Fine-tuning do focinho com ArcFace — versao CPU do notebook.

Fecha o gap de VERIFICACAO da baseline congelada (FAR@FRR=1% de 54.2%).
A baseline usa embedding generico do ImageNet: ordena bem (Rank-1 98%) mas
nao AFASTA genuinos de impostores. Metric learning (ArcFace) treina isso ->
abre um vale entre as distribuicoes -> FAR (False Acceptance Rate) despenca.

PROTOCOLO HONESTO (regra de ouro do projeto):
  - split POR ANIMAL: os bois de teste NUNCA aparecem no treino (open-set real)
  - avalia fine-tuned E congelada no MESMO split de teste -> ganho comparavel

Uso:
  python -m biometria.finetune_arcface                 # 224px, 12 epocas (~2h CPU)
  python -m biometria.finetune_arcface --size 160 --epochs 8   # rapido (~40min)
"""
from __future__ import annotations
import argparse
import pathlib
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim" / "muzzle" / "BeefCattle_Muzzle_Individualized"
OUT = ROOT / "data" / "interim" / "muzzlenet_arcface.pt"
IMG_EXT = {".jpg", ".jpeg", ".png"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_animals(min_imgs=4):
    out = []
    for d in sorted(p for p in DATA.iterdir() if p.is_dir() and not p.name.startswith("._")):
        imgs = [q for q in sorted(d.glob("*.jpg")) if not q.name.startswith("._")]
        if len(imgs) >= min_imgs:
            out.append((d.name, imgs))
    return out


class DS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


class MuzzleNet(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        bb.fc = nn.Linear(2048, emb_dim)
        self.backbone, self.bn = bb, nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        return self.bn(self.backbone(x))


class ArcFace(nn.Module):
    def __init__(self, in_f, n_cls, s=30.0, m=0.50):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_cls, in_f))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def forward(self, emb, y):
        x, W = F.normalize(emb), F.normalize(self.W)
        cos = (x @ W.t()).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cos) + self.m)
        oh = F.one_hot(y, cos.size(1)).float()
        return self.s * (oh * target + (1 - oh) * cos)


class Frozen(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        bb.fc = nn.Identity()
        self.bb = bb

    def forward(self, x):
        return self.bb(x)


@torch.no_grad()
def embed_all(model, items, tf, bs=64):
    model.eval()
    vs, ls = [], []
    for x, y in DataLoader(DS(items, tf), batch_size=bs):
        vs.append(model(x.to(DEVICE)).cpu().numpy())
        ls.append(y.numpy())
    e = np.vstack(vs).astype("float32")
    e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    return e, np.concatenate(ls)


def identification(emb, labels, seed=42):
    rng = np.random.default_rng(seed)
    gal, prb = [], []
    for l in np.unique(labels):
        ix = np.where(labels == l)[0]
        rng.shuffle(ix)
        k = len(ix) // 2
        gal += list(ix[:k]); prb += list(ix[k:])
    gal, prb = np.array(gal), np.array(prb)
    sims = emb[prb] @ emb[gal].T
    corr = labels[gal][np.argsort(-sims, axis=1)] == labels[prb][:, None]
    aps = []
    for i in range(len(prb)):
        rel = corr[i]; n = rel.sum()
        if n == 0:
            aps.append(0.0); continue
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append((prec * rel).sum() / n)
    return corr[:, 0].mean(), corr[:, :5].any(1).mean(), float(np.mean(aps))


def verification(emb, labels, n=20000, seed=42):
    rng = np.random.default_rng(seed)
    by = {l: np.where(labels == l)[0] for l in np.unique(labels)}
    multi = [l for l, ix in by.items() if len(ix) >= 2]
    gen = []
    for _ in range(n):
        a, b = rng.choice(by[multi[rng.integers(len(multi))]], 2, replace=False)
        gen.append(emb[a] @ emb[b])
    imp = []
    while len(imp) < n:
        i, j = rng.integers(len(labels)), rng.integers(len(labels))
        if labels[i] != labels[j]:
            imp.append(emb[i] @ emb[j])
    gen, imp = np.array(gen), np.array(imp)
    ths = np.linspace(-1, 1, 400)
    far = np.array([(imp >= t).mean() for t in ths])
    frr = np.array([(gen < t).mean() for t in ths])
    eer = float((far[np.argmin(np.abs(far - frr))] + frr[np.argmin(np.abs(far - frr))]) / 2)
    return eer, float(far[np.argmin(np.abs(frr - 0.01))]), gen.mean(), imp.mean()


def report(tag, emb, labels):
    r1, r5, mAP = identification(emb, labels)
    eer, far, gm, im = verification(emb, labels)
    print(f"  [{tag:11s}] Rank-1={r1*100:5.1f}%  Rank-5={r5*100:5.1f}%  mAP={mAP*100:5.1f}%"
          f"  | EER={eer*100:4.1f}%  FAR@FRR1%={far*100:5.1f}%  (gen {gm:.2f}/imp {im:.2f})",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    animals = load_animals()
    random.Random(42).shuffle(animals)
    n_test = max(1, int(0.25 * len(animals)))
    test_a, train_a = animals[:n_test], animals[n_test:]
    print(f"dispositivo={DEVICE} | treino={len(train_a)} animais | "
          f"teste (nunca vistos)={len(test_a)} animais", flush=True)

    train_items = [(p, i) for i, (_, ps) in enumerate(train_a) for p in ps]
    test_items = [(p, i) for i, (_, ps) in enumerate(test_a) for p in ps]
    n_cls = len(train_a)
    print(f"{len(train_items)} imagens de treino | {len(test_items)} de teste", flush=True)

    train_tf = transforms.Compose([
        transforms.Resize((args.size + 32, args.size + 32)),
        transforms.RandomCrop(args.size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.02),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("\n=== ANTES: baseline congelada nos animais de teste ===", flush=True)
    fro = Frozen().to(DEVICE)
    e0, lab = embed_all(fro, test_items, eval_tf)
    report("congelada", e0, lab)
    del fro

    net, head = MuzzleNet(512).to(DEVICE), ArcFace(512, n_cls).to(DEVICE)
    opt = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": 1e-4},
        {"params": net.bn.parameters(), "lr": 1e-3},
        {"params": head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(DS(train_items, train_tf), batch_size=args.batch,
                        shuffle=True, drop_last=True)

    print(f"\n=== TREINO ArcFace ({args.epochs} epocas, {args.size}px) ===", flush=True)
    for ep in range(args.epochs):
        net.train(); head.train()
        t0, tot, ok, ls = time.time(), 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = head(net(x), y)
            loss = crit(logits, y)
            loss.backward(); opt.step()
            ls += loss.item() * len(y); tot += len(y)
            ok += (logits.argmax(1) == y).sum().item()
        sched.step()
        print(f"  epoca {ep+1}/{args.epochs}  loss={ls/tot:.3f}  "
              f"acc_treino={ok/tot:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        torch.save(net.state_dict(), OUT)

    print("\n=== DEPOIS: nos MESMOS animais de teste (nunca vistos) ===", flush=True)
    e1, lab1 = embed_all(net, test_items, eval_tf)
    report("congelada", e0, lab)
    report("fine-tuned", e1, lab1)
    print(f"\nmodelo salvo em {OUT}", flush=True)


if __name__ == "__main__":
    main()
