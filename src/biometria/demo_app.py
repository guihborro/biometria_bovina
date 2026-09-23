"""Demo Streamlit — identidade de bovino por focinho (modelo TREINADO).

Cartao de visita da Fase 0. Usa o modelo com fine-tuning ArcFace.

HONESTIDADE DA DEMO: mostra apenas os 67 animais que o modelo NUNCA VIU no
treino. Exibir os 201 de treino inflaria o resultado (o ArcFace otimizou
justamente para separa-los). Este e o cenario real: animal novo no brete.

Rodar:
  cd src && streamlit run biometria/demo_app.py
"""
from __future__ import annotations
import pathlib
import sys

# `streamlit run` poe no sys.path a pasta DO SCRIPT (src/biometria/), nao src/.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
EMB = ROOT / "data" / "interim" / "demo_emb_ft.npy"
META = ROOT / "data" / "interim" / "demo_meta_ft.npz"


@st.cache_resource
def load():
    emb = np.load(EMB)
    m = np.load(META, allow_pickle=True)
    return emb, list(m["names"]), list(m["paths"])


st.set_page_config(page_title="Focinho ID — bovinos", layout="wide")
st.title("Identidade de bovino por focinho")

if not EMB.exists():
    st.error("Cache ausente. Rode antes:  python -m biometria.build_demo_cache")
    st.stop()

emb, names, paths = load()
uniq = sorted(set(names))
idx_by_name = {n: np.where(np.array(names) == n)[0] for n in uniq}

st.caption(f"Modelo ArcFace com fine-tuning · {len(uniq)} animais que o modelo "
           f"**nunca viu no treino** · Rank-1 99,5% · FAR 0,8% com 3 fotos")

with st.sidebar:
    st.metric("Animais na base", len(uniq))
    st.metric("Fotos indexadas", len(paths))
    st.divider()
    n_enroll = st.select_slider("Fotos cadastradas por animal", [1, 3, 5], value=3)
    thr = st.slider("Limiar de aceitação", 0.0, 1.0, 0.45, 0.01)
    st.caption("1 foto = protocolo severo · 3-5 fotos = cadastro real no brete. "
               "É a diferença entre 34% e 0,8% de falsa aceitação.")

tab1, tab2 = st.tabs(["Quem é esse boi?", "É o boi certo? (verificação)"])

with tab1:
    st.subheader("Identificação — 1 para N")
    if "probe" not in st.session_state:
        st.session_state.probe = int(np.random.randint(len(paths)))
    if st.button("Sortear outro focinho"):
        st.session_state.probe = int(np.random.randint(len(paths)))
    p = st.session_state.probe

    sims = emb @ emb[p]
    sims[p] = -1.0
    top = np.argsort(-sims)[:5]
    ok = names[top[0]] == names[p]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(paths[p], caption=f"Consulta — animal real: {names[p]}",
                 use_container_width=True)
    with c2:
        if ok:
            st.success(f"Identificado: {names[top[0]]} · similaridade {sims[top[0]]:.3f}")
        else:
            st.error(f"Errou: previu {names[top[0]]}, era {names[p]}")
        st.write("Top-5 mais parecidos na base:")
        cols = st.columns(5)
        for j, i in enumerate(top):
            with cols[j]:
                st.image(paths[i], use_container_width=True)
                st.caption(f"{names[i]} {'✅' if names[i]==names[p] else ''}\n{sims[i]:.3f}")

with tab2:
    st.subheader("Verificação — 1 para 1")
    st.write("O sistema compara a foto de consulta com as fotos **cadastradas** "
             "do animal reivindicado e fica com a melhor similaridade.")

    q = st.selectbox("Foto de consulta", range(len(paths)),
                     format_func=lambda i: f"{names[i]} · {pathlib.Path(paths[i]).name}",
                     index=int(st.session_state.get("probe", 0)))
    claim = st.selectbox("Animal reivindicado", uniq, index=uniq.index(names[q]))
    genuine = names[q] == claim

    pool = np.array([i for i in idx_by_name[claim] if i != q])
    if len(pool) == 0:
        st.warning("Sem outras fotos desse animal.")
    else:
        rng = np.random.default_rng(0)
        enroll = rng.choice(pool, min(n_enroll, len(pool)), replace=False)
        sim = float(np.max(emb[enroll] @ emb[q]))
        accept = sim >= thr

        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(paths[q], caption=f"Consulta ({names[q]})", use_container_width=True)
        with c2:
            st.write(f"Cadastro de **{claim}** ({len(enroll)} foto(s)):")
            cs = st.columns(len(enroll))
            for j, i in enumerate(enroll):
                with cs[j]:
                    st.image(paths[i], use_container_width=True)

        st.metric("Melhor similaridade", f"{sim:.3f}", f"limiar {thr:.2f}")
        if accept and genuine:
            st.success("ACEITO — é o mesmo animal (correto)")
        elif accept and not genuine:
            st.error("ACEITO — mas são animais diferentes (FALSA ACEITAÇÃO)")
        elif not accept and genuine:
            st.warning("REJEITADO — mas era o mesmo animal (falsa rejeição)")
        else:
            st.info("REJEITADO — animais diferentes (correto)")

        st.caption("Experimente: escolha um animal reivindicado diferente do real "
                   "e reduza para 1 foto cadastrada — é assim que aparece a falsa aceitação.")

st.divider()
with st.expander("Enviar uma foto de focinho (embedding ao vivo)"):
    up = st.file_uploader("JPG/PNG de um focinho", type=["jpg", "jpeg", "png"])
    if up is not None:
        import torch
        from PIL import Image
        from torchvision import transforms
        from biometria.finetune_arcface import MuzzleNet, DEVICE

        @st.cache_resource
        def _net():
            n = MuzzleNet(512).to(DEVICE)
            n.load_state_dict(torch.load(
                ROOT / "data" / "interim" / "muzzlenet_arcface.pt", map_location=DEVICE))
            n.eval()
            return n

        tf = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = Image.open(up).convert("RGB")
        with torch.no_grad():
            v = _net()(tf(img).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
        v /= np.linalg.norm(v) + 1e-8
        s = emb @ v
        b = int(np.argmax(s))
        st.image(img, width=260, caption="enviada")
        st.write(f"Mais parecido: **{names[b]}** · similaridade {s[b]:.3f} — "
                 f"{'ACEITO' if s[b] >= thr else 'REJEITADO'} no limiar {thr:.2f}")
