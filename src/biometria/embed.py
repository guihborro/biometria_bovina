"""Extrator de embedding off-the-shelf (Modelo B, baseline sem treino).

Usa um backbone ResNet50 pre-treinado no ImageNet, sem a ultima camada,
como extrator de vetor de 2048-d por imagem. L2-normalizado -> similaridade
por cosseno. Roda em CPU. Isto e a 'baseline burra' do re-ID: se o embedding
generico ja separa os focinhos, o fine-tuning com metric learning (depois,
em GPU) so melhora.
"""
from __future__ import annotations
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _load_backbone() -> nn.Module:
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Identity()          # remove classificador -> vetor 2048-d
    net.eval()
    return net


_BACKBONE = None


@torch.no_grad()
def embed_pil(img: "Image.Image") -> np.ndarray:
    """Embedding L2-normalizado de uma unica imagem PIL (para a demo)."""
    global _BACKBONE
    if _BACKBONE is None:
        _BACKBONE = _load_backbone()
    v = _BACKBONE(_TF(img.convert("RGB")).unsqueeze(0)).numpy()[0]
    return (v / (np.linalg.norm(v) + 1e-8)).astype("float32")


@torch.no_grad()
def embed_paths(paths: list[pathlib.Path], batch_size: int = 32,
                log_every: int = 500) -> np.ndarray:
    net = _load_backbone()
    out = []
    batch, done = [], 0
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            out.append(np.zeros(2048, dtype="float32")); done += 1; continue
        batch.append(_TF(img))
        if len(batch) == batch_size:
            v = net(torch.stack(batch)).numpy()
            out.append(v); batch = []; done += len(v)
            if done % log_every < batch_size:
                print(f"    embed {done}/{len(paths)}")
    if batch:
        out.append(net(torch.stack(batch)).numpy()); done += len(batch)
    emb = np.vstack(out).astype("float32")
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)   # L2 norm
    return emb
