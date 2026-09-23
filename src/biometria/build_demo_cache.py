"""Gera o cache de embeddings do modelo TREINADO para a demo.

IMPORTANTE — honestidade da demo: o modelo foi treinado nas identidades de
201 animais. Mostrar esses animais na demo inflaria o resultado (o ArcFace
otimizou justamente para separa-los). A demo usa portanto SO os 67 animais
DE TESTE, que o modelo nunca viu — que e o cenario real: um animal novo
chega ao brete e precisa ser reconhecido.

Uso: python -m biometria.build_demo_cache
"""
from __future__ import annotations
import pathlib
import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from biometria.finetune_arcface import MuzzleNet, DS, load_animals, DEVICE

ROOT = pathlib.Path(__file__).resolve().parents[2]
CKPT = ROOT / "data" / "interim" / "muzzlenet_arcface.pt"
OUT_EMB = ROOT / "data" / "interim" / "demo_emb_ft.npy"
OUT_META = ROOT / "data" / "interim" / "demo_meta_ft.npz"

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def main() -> None:
    animals = load_animals()
    random.Random(42).shuffle(animals)              # MESMO split do treino
    test_a = animals[:max(1, int(0.25 * len(animals)))]
    items, names, paths = [], [], []
    for i, (name, ps) in enumerate(test_a):
        for p in ps:
            items.append((p, i)); names.append(name); paths.append(str(p))
    print(f"{len(test_a)} animais nunca vistos | {len(items)} imagens")

    net = MuzzleNet(512).to(DEVICE)
    net.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    net.eval()

    vs = []
    for n, (x, _) in enumerate(DataLoader(DS(items, TF), batch_size=64), 1):
        vs.append(net(x.to(DEVICE)).cpu().numpy())
        print(f"  lote {n}", flush=True)
    emb = np.vstack(vs).astype("float32")
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    np.save(OUT_EMB, emb)
    np.savez(OUT_META, names=np.array(names), paths=np.array(paths))
    print(f"salvo {OUT_EMB.name} {emb.shape}")


if __name__ == "__main__":
    main()
