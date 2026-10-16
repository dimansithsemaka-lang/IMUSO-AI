"""
IMUSO AI Core — Intelligent Mutant Utility & Synthetic Organism Engine
Deep-Tech MVP: enzyme mutation, ESMFold structure prediction, 3D visualization.

Install:   pip install streamlit py3Dmol stmol requests pandas ipython_genomics
           (minimum: pip install streamlit py3Dmol stmol requests pandas)
Run:       streamlit run app.py

NOTE: ΔG, kcat and Tm values are fast heuristic surrogates computed from sequence
descriptors (not docking, MD, or experiment). The ESMFold structures are real
predictions from Meta's public API. Validate candidates with real tooling and wet lab.
"""

import math
import random
import re
import time

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import py3Dmol
except ImportError:  # pragma: no cover
    py3Dmol = None

try:
    from stmol import showmol
except Exception:  # pragma: no cover
    showmol = None

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMUSO AI Core",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CYAN, GREEN, BG = "#00E5FF", "#00FF88", "#0E1117"
ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
MAX_LEN, MIN_LEN = 400, 16  # ESMFold Atlas API practical limit

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', sans-serif;
    color: #E6F1F5;
}}
.stApp {{
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(0,229,255,0.09), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(0,255,136,0.07), transparent 60%),
        {BG};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; max-width: 1500px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0A0E14 0%, #0C131B 100%);
    border-right: 1px solid rgba(0,229,255,0.25);
    box-shadow: 4px 0 24px rgba(0,229,255,0.06);
}}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
    font-family: 'Orbitron', sans-serif; color: {CYAN}; letter-spacing: 1.5px;
}}

/* Header */
.imuso-title {{
    font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 3rem;
    letter-spacing: 3px; margin: 0; line-height: 1.1;
    background: linear-gradient(90deg, {CYAN}, {GREEN});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 14px rgba(0,229,255,0.35));
}}
.imuso-sub {{
    font-family: 'Inter', sans-serif; font-weight: 300; font-size: 1.05rem;
    color: #8FB7C4; letter-spacing: 4px; text-transform: uppercase; margin-top: 6px;
}}
.imuso-divider {{
    height: 2px; margin: 14px 0 22px 0; border: 0;
    background: linear-gradient(90deg, {CYAN}, {GREEN}, transparent);
    box-shadow: 0 0 12px rgba(0,229,255,0.6);
}}

/* Cards */
.card {{
    background: linear-gradient(145deg, rgba(18,28,38,0.92), rgba(12,18,26,0.92));
    border: 1px solid rgba(0,229,255,0.28); border-radius: 16px;
    padding: 18px 20px; margin-bottom: 18px;
    box-shadow: 0 0 22px rgba(0,229,255,0.07), inset 0 0 20px rgba(0,229,255,0.03);
}}
.card-title {{
    font-family: 'Orbitron', sans-serif; font-size: 0.9rem; color: {CYAN};
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px;
}}
.metric-card {{
    position: relative; overflow: hidden;
    background: linear-gradient(145deg, rgba(14,24,34,0.95), rgba(9,14,20,0.95));
    border: 1px solid {CYAN}; border-radius: 16px; padding: 18px 20px; height: 100%;
    box-shadow: 0 0 18px rgba(0,229,255,0.35), inset 0 0 18px rgba(0,229,255,0.07);
    transition: transform .25s ease, box-shadow .25s ease;
}}
.metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 0 30px rgba(0,255,136,0.5), inset 0 0 22px rgba(0,255,136,0.1);
    border-color: {GREEN};
}}
.metric-card::before {{
    content: ""; position: absolute; top: 0; left: -100%; width: 60%; height: 2px;
    background: linear-gradient(90deg, transparent, {GREEN}, transparent);
    animation: scan 3.2s linear infinite;
}}
@keyframes scan {{ to {{ left: 160%; }} }}
.mc-title {{ font-size: 0.78rem; letter-spacing: 2px; text-transform: uppercase; color: #8FB7C4; }}
.mc-value {{
    font-family: 'Orbitron', sans-serif; font-size: 2.2rem; font-weight: 700;
    color: {GREEN}; text-shadow: 0 0 14px rgba(0,255,136,0.6); margin: 8px 0 2px 0;
}}
.mc-unit {{ font-size: 0.8rem; color: {CYAN}; margin-left: 8px; font-weight: 500; }}
.mc-wt {{ font-size: 0.78rem; color: #6F8D99; font-family: 'JetBrains Mono', monospace; }}
.mc-delta {{ font-size: 0.9rem; font-weight: 600; margin-top: 6px; }}
.mc-delta.good {{ color: {GREEN}; }}
.mc-delta.bad  {{ color: #FF5C7A; }}

/* Sequence display */
.seqbox {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; line-height: 1.75;
    background: #080C12; border: 1px solid rgba(0,229,255,0.2); border-radius: 12px;
    padding: 14px 16px; overflow-x: auto; white-space: pre;
}}
.seqbox .lbl {{ color: #5F7C88; user-select: none; }}
.seqbox .wt  {{ color: {CYAN}; }}
.seqbox .mt  {{ color: #BFEFD8; }}
.seqbox .mut {{
    color: #04120B; background: {GREEN}; border-radius: 3px; padding: 0 2px;
    box-shadow: 0 0 8px {GREEN}; font-weight: 700;
}}
.seqbox .old {{
    color: #FF9AAE; background: rgba(255,92,122,0.18); border-radius: 3px; padding: 0 2px;
}}

/* Buttons */
.stButton > button, .stDownloadButton > button {{
    width: 100%; font-family: 'Orbitron', sans-serif; letter-spacing: 1.5px;
    color: #031014; font-weight: 700;
    background: linear-gradient(90deg, {CYAN}, {GREEN});
    border: none; border-radius: 12px; padding: 0.7rem 1rem;
    box-shadow: 0 0 18px rgba(0,229,255,0.45);
    transition: all .2s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-2px); box-shadow: 0 0 30px rgba(0,255,136,0.75); color: #000;
}}
.badge {{
    display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.72rem;
    border: 1px solid {CYAN}; color: {CYAN}; margin-right: 6px; letter-spacing: 1px;
    background: rgba(0,229,255,0.07);
}}
.badge.g {{ border-color: {GREEN}; color: {GREEN}; background: rgba(0,255,136,0.07); }}
.disclaimer {{ font-size: 0.75rem; color: #5F7C88; margin-top: 8px; }}
div[data-testid="stTabs"] button[aria-selected="true"] {{ color: {GREEN}; }}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# DATA: SCAFFOLDS & SUBSTRATE PROFILES
# ──────────────────────────────────────────────────────────────────────────────
# Mature IsPETase (Ideonella sakaiensis). Verify against UniProt A0A0K8P6T7 before production use.
PETASE_SEQ = (
    "QTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS"
)

# PE / Nylon entries are SCAFFOLD SEEDS for the demo (reusing an α/β-hydrolase fold).
# Replace with curated sequences (e.g., NylB / NylC from UniProt, or your PE-active
# candidates) or paste one via "Custom Sequence" in the sidebar.
SUBSTRATES = {
    "PET": dict(
        full="Polyethylene terephthalate", scaffold="IsPETase (wild-type)", seq=PETASE_SEQ,
        dg=-6.8, kcat=14.0, tm=48.8, t_opt=40.0, ph_opt=8.0, ph_w=1.6,
    ),
    "PE": dict(
        full="Polyethylene", scaffold="α/β-hydrolase seed (replace)", seq=PETASE_SEQ,
        dg=-4.3, kcat=2.1, tm=46.0, t_opt=37.0, ph_opt=7.5, ph_w=1.8,
    ),
    "Nylon": dict(
        full="Polyamide (Nylon-6/6,6)", scaffold="α/β-hydrolase seed (replace)", seq=PETASE_SEQ,
        dg=-5.2, kcat=3.6, tm=50.5, t_opt=45.0, ph_opt=8.5, ph_w=1.5,
    ),
}

KD = {  # Kyte-Doolittle hydropathy
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
AA_GROUPS = ["AVILM", "FWY", "ST", "NQ", "DE", "KRH", "GP", "C"]
AA20 = "ACDEFGHIKLMNPQRSTVWY"
VALID = set(AA20)

# ──────────────────────────────────────────────────────────────────────────────
# MUTATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def protected_positions(seq: str, is_petase_scaffold: bool) -> set:
    """Residues the engine will not touch: catalytic motif, Cys (disulfides), triad."""
    prot = set()
    for m in re.finditer(r"G.S.G", seq):  # α/β-hydrolase nucleophile elbow
        prot.update(range(m.start(), m.end()))
    prot.update(i for i, a in enumerate(seq) if a == "C")
    if is_petase_scaffold:  # IsPETase triad, mature numbering S131/D177/H208 (best effort)
        for idx, aa in ((130, "S"), (176, "D"), (207, "H")):
            if idx < len(seq) and seq[idx] == aa:
                prot.add(idx)
    return prot


def pick_substitution(rng: random.Random, old: str) -> str:
    """70% conservative (same physico-chemical group), 30% any other residue."""
    if rng.random() < 0.70:
        for g in AA_GROUPS:
            if old in g and len(g) > 1:
                return rng.choice([a for a in g if a != old])
    return rng.choice([a for a in AA20 if a != old])


def generate_mutant(wt: str, aggressiveness: float, seed: int, protected: set):
    """Return (mutant_sequence, [(pos0, old, new), ...])."""
    rng = random.Random(seed)
    n = int(round(len(wt) * (aggressiveness / 100.0) * 0.10))  # up to 10% of residues
    if aggressiveness > 0:
        n = max(1, n)
    candidates = [i for i in range(len(wt)) if i not in protected]
    n = min(n, len(candidates))
    positions = sorted(rng.sample(candidates, n))
    seq, muts = list(wt), []
    for p in positions:
        new = pick_substitution(rng, wt[p])
        seq[p] = new
        muts.append((p, wt[p], new))
    return "".join(seq), muts


# ──────────────────────────────────────────────────────────────────────────────
# PROPERTY ESTIMATION (heuristic surrogate model)
# ──────────────────────────────────────────────────────────────────────────────
def seq_features(seq: str) -> dict:
    n = len(seq)
    x = lambda a: 100.0 * seq.count(a) / n
    return dict(
        aliphatic=x("A") + 2.9 * x("V") + 3.9 * (x("I") + x("L")),
        charged=100.0 * sum(seq.count(a) for a in "DEKR") / n,
        gravy=sum(KD[a] for a in seq) / n,
    )


def estimate_properties(wt: str, mut: str, muts: list, prof: dict, temp_c: float, ph: float) -> dict:
    """Illustrative surrogate: sequence descriptors + conditions -> (ΔG, kcat, Tm)."""
    fw, fm = seq_features(wt), seq_features(mut)
    n, L = len(muts), len(wt)

    # Thermal stability
    ddtm = 0.0
    for _, old, new in muts:
        d = 0.35 + 0.15 * 0  # baseline mild stabilizing prior
        if new == "P" and old != "G":
            d += 0.8
        if new in "RKE":
            d += 0.5
        if new == "G":
            d -= 0.8
        d -= 0.15 * abs(KD[new] - KD[old])
        ddtm += d
    ddtm += 0.06 * (fm["aliphatic"] - fw["aliphatic"]) + 0.15 * (fm["charged"] - fw["charged"])
    overload = max(0.0, n / L - 0.04) * 100.0  # too many mutations destabilize the fold
    tm = max(30.0, min(95.0, prof["tm"] + ddtm - 1.8 * overload))

    # Binding free energy (more negative = tighter)
    aromatic_gain = sum(1 for _, o, nw in muts if nw in "FWY" and o not in "FWY")
    hydrophobic_gain = max(0.0, fm["gravy"] - fw["gravy"]) * 4.0
    dg = prof["dg"] - 0.22 * aromatic_gain - 0.35 * hydrophobic_gain + 0.06 * overload * 10
    dg = max(-14.0, min(-1.0, dg))

    # Turnover (kcat, 1/min): Q10 temperature scaling × folded fraction × pH window × mutational uplift
    folded = 1.0 / (1.0 + math.exp((temp_c - (tm - 10.0)) / 3.0))
    q10 = 2.0 ** ((temp_c - prof["t_opt"]) / 10.0)
    ph_f = math.exp(-((ph - prof["ph_opt"]) / prof["ph_w"]) ** 2)
    affinity_gain = math.exp(-(dg - prof["dg"]) * 0.35)
    kcat = prof["kcat"] * q10 * folded * ph_f * affinity_gain * 2.0
    return dict(dg=dg, kcat=max(kcat, 0.0), tm=tm)


# ──────────────────────────────────────────────────────────────────────────────
# ESMFOLD API
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fold_sequence(seq: str) -> str:
    """POST to the ESM Atlas API; returns PDB text. Raises RuntimeError on failure
    (exceptions are not cached by Streamlit, so failed calls are retried next time)."""
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(
                ESMFOLD_URL, data=seq,
                headers={"Content-Type": "text/plain"}, timeout=180,
            )
            if r.status_code == 200 and "ATOM" in r.text:
                return r.text
            last_err = f"HTTP {r.status_code}: {r.text[:160].strip()}"
        except requests.exceptions.Timeout:
            last_err = "request timed out"
        except requests.exceptions.ConnectionError:
            last_err = "network/connection error"
        except requests.exceptions.RequestException as e:
            last_err = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last_err or "unknown error")


def mean_plddt(pdb: str):
    vals = []
    for line in pdb.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                pass
    if not vals:
        return None
    m = sum(vals) / len(vals)
    return m * 100.0 if m <= 1.0 else m


# ──────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def metric_card(title, icon, value, unit, wt_value, delta, good):
    cls = "good" if good else "bad"
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "■")
    return (
        f'<div class="metric-card"><div class="mc-title">{icon} {title}</div>'
        f'<div class="mc-value">{value}<span class="mc-unit">{unit}</span></div>'
        f'<div class="mc-wt">Wild-type: {wt_value} {unit}</div>'
        f'<div class="mc-delta {cls}">{arrow} {delta:+.2f} {unit} vs WT</div></div>'
    )


def alignment_html(wt: str, mut: str, width: int = 50) -> str:
    out = []
    for s in range(0, len(wt), width):
        w, m = wt[s:s + width], mut[s:s + width]
        wl, ml = "", ""
        for a, b in zip(w, m):
            if a == b:
                wl += a
                ml += b
            else:
                wl += f'<span class="old">{a}</span>'
                ml += f'<span class="mut">{b}</span>'
        out.append(
            f'<span class="lbl">WT  {s+1:>4} </span><span class="wt">{wl}</span>\n'
            f'<span class="lbl">MUT {s+1:>4} </span><span class="mt">{ml}</span>\n'
        )
    return '<div class="seqbox">' + "\n".join(out) + "</div>"


def single_seq_html(seq: str, cls: str, width: int = 50) -> str:
    rows = [
        f'<span class="lbl">{i+1:>4} </span><span class="{cls}">{seq[i:i+width]}</span>'
        for i in range(0, len(seq), width)
    ]
    return '<div class="seqbox">' + "\n".join(rows) + "</div>"


def build_view(pdb: str, mut_resi: list, color_mode: str, surface: bool, spin: bool, h: int = 560):
    view = py3Dmol.view(width=900, height=h)
    view.addModel(pdb, "pdb")
    if color_mode == "pLDDT confidence":
        scale = 1.0 if (mean_plddt(pdb) or 100) > 1.5 else 100.0
        view.setStyle({"cartoon": {"colorscheme": {
            "prop": "b", "gradient": "roygb", "min": 50 / scale if scale == 1 else 0.5,
            "max": 90 if scale == 1 else 0.9}}})
    elif color_mode == "Spectrum":
        view.setStyle({"cartoon": {"color": "spectrum"}})
    else:
        view.setStyle({"cartoon": {"color": CYAN}})
    if surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.28, "color": "#7FDFFF"})
    if mut_resi:
        view.addStyle({"resi": mut_resi}, {"stick": {"color": GREEN, "radius": 0.22},
                                           "cartoon": {"color": GREEN}})
        view.addStyle({"resi": mut_resi}, {"sphere": {"color": GREEN, "radius": 0.55, "opacity": 0.55}})
    view.setBackgroundColor(BG)
    view.zoomTo()
    if spin:
        view.spin(True)
    return view


def show_view(view, h: int = 560):
    try:
        if showmol is None:
            raise ImportError
        showmol(view, height=h, width=900)
    except Exception:
        components.html(view._make_html(), height=h + 10, scrolling=False)


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="imuso-title">IMUSO AI Core</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="imuso-sub">Intelligent Mutant Utility &amp; Synthetic Organism Engine</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="badge">ESMFOLD</span><span class="badge g">PLASTIC-DEGRADING ENZYMES</span>'
    '<span class="badge">3D STRUCTURE ENGINE</span><hr class="imuso-divider">',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROL PANEL
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 CONTROL PANEL")
    substrate = st.selectbox(
        "Target Substrate", list(SUBSTRATES.keys()),
        format_func=lambda k: f"{k} — {SUBSTRATES[k]['full']}",
    )
    prof = SUBSTRATES[substrate]
    st.caption(f"Scaffold: {prof['scaffold']}")

    st.markdown("### Operating Conditions")
    temp_c = st.slider("Operating Temperature (°C)", 10, 80, int(prof["t_opt"]), 1)
    ph = st.slider("pH", 4.0, 11.0, float(prof["ph_opt"]), 0.1)

    st.markdown("### Evolution Engine")
    aggr = st.slider("Mutation Aggressiveness (%)", 0, 100, 30, 5,
                     help="100% ≈ up to 10% of residues substituted. Catalytic motif and Cys are protected.")
    seed = st.number_input("Random Seed", 0, 999999, 42, 1)

    with st.expander("Custom Sequence (optional)"):
        custom = st.text_area("Paste wild-type amino-acid sequence", height=110,
                              placeholder=f"{MIN_LEN}–{MAX_LEN} residues, 20 standard amino acids").strip()

    st.markdown("### Visualization")
    color_mode = st.selectbox("Cartoon Colouring", ["pLDDT confidence", "Spectrum", "IMUSO Cyan"])
    show_surface = st.toggle("Transparent Surface Overlay", True)
    spin = st.toggle("Auto-Rotate", False)

    run = st.button("⚡ RUN IMUSO ENGINE")

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    wt_seq = re.sub(r"\s+", "", custom).upper() if custom else prof["seq"]
    if not set(wt_seq) <= VALID:
        st.error("❌ Invalid characters in sequence. Use only the 20 standard amino-acid letters.")
        return None
    if not (MIN_LEN <= len(wt_seq) <= MAX_LEN):
        st.error(f"❌ Sequence length must be {MIN_LEN}–{MAX_LEN} residues (got {len(wt_seq)}).")
        return None

    is_petase = (wt_seq == PETASE_SEQ)
    prot = protected_positions(wt_seq, is_petase)
    mut_seq, muts = generate_mutant(wt_seq, aggr, int(seed), prot)

    res = dict(substrate=substrate, wt=wt_seq, mut=mut_seq, muts=muts, protected=len(prot),
               pdb_wt=None, pdb_mut=None, err_wt=None, err_mut=None)

    with st.spinner("🧬 Querying Meta ESMFold — folding mutant structure…"):
        try:
            res["pdb_mut"] = fold_sequence(mut_seq)
        except RuntimeError as e:
            res["err_mut"] = str(e)
    with st.spinner("🧪 Folding wild-type reference structure…"):
        try:
            res["pdb_wt"] = fold_sequence(wt_seq)
        except RuntimeError as e:
            res["err_wt"] = str(e)
    return res


if run or "result" not in st.session_state:
    out = run_pipeline()
    if out is not None:
        st.session_state["result"] = out
        if run:
            st.toast("IMUSO engine run complete", icon="✅")

R = st.session_state.get("result")
if R is None:
    st.info("Adjust the control panel and press **RUN IMUSO ENGINE**.")
    st.stop()

# Metrics use the substrate profile the mutant was generated for, live conditions from sliders
rprof = SUBSTRATES[R["substrate"]]
wt_props = estimate_properties(R["wt"], R["wt"], [], rprof, temp_c, ph)
mt_props = estimate_properties(R["wt"], R["mut"], R["muts"], rprof, temp_c, ph)

if R["substrate"] != substrate:
    st.warning(f"Displayed results were generated for **{R['substrate']}**. "
               f"Press **RUN IMUSO ENGINE** to regenerate for **{substrate}**.")

# ──────────────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ──────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
d_dg = mt_props["dg"] - wt_props["dg"]
d_k = mt_props["kcat"] - wt_props["kcat"]
d_tm = mt_props["tm"] - wt_props["tm"]
with c1:
    st.markdown(metric_card("Binding Energy", "🔗", f"{mt_props['dg']:.2f}", "kcal/mol ΔG",
                            f"{wt_props['dg']:.2f}", d_dg, d_dg <= 0), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Turnover Speed", "⚡", f"{mt_props['kcat']:.2f}", "kcat/min",
                            f"{wt_props['kcat']:.2f}", d_k, d_k >= 0), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Thermal Stability", "🌡️", f"{mt_props['tm']:.1f}", "°C Tm",
                            f"{wt_props['tm']:.1f}", d_tm, d_tm >= 0), unsafe_allow_html=True)
with c4:
    pl = mean_plddt(R["pdb_mut"]) if R["pdb_mut"] else None
    plw = mean_plddt(R["pdb_wt"]) if R["pdb_wt"] else None
    if pl is not None:
        dpl = pl - (plw if plw is not None else pl)
        st.markdown(metric_card("Fold Confidence", "🎯", f"{pl:.1f}", "pLDDT",
                                f"{plw:.1f}" if plw is not None else "n/a", dpl, dpl >= -2),
                    unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="metric-card"><div class="mc-title">🎯 Fold Confidence</div>'
            '<div class="mc-value">—</div><div class="mc-wt">Structure unavailable</div></div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">ΔG / kcat / Tm are heuristic surrogate estimates derived from sequence '
    'descriptors and operating conditions — not docking or experimental results. '
    'Structures and pLDDT are real ESMFold predictions.</div>',
    unsafe_allow_html=True,
)
st.write("")

# ──────────────────────────────────────────────────────────────────────────────
# SEQUENCE COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">🧬 Sequence Comparison — Wild-Type vs Mutated</div>',
            unsafe_allow_html=True)
n_mut = len(R["muts"])
ident = 100.0 * (len(R["wt"]) - n_mut) / len(R["wt"])
st.markdown(
    f'<span class="badge g">{n_mut} MUTATIONS</span><span class="badge">{len(R["wt"])} RESIDUES</span>'
    f'<span class="badge">{ident:.1f}% IDENTITY</span><span class="badge">{R["protected"]} PROTECTED SITES</span>',
    unsafe_allow_html=True,
)
t_align, t_wt, t_mut = st.tabs(["Aligned View", "Wild-Type", "Mutated"])
with t_align:
    st.markdown(alignment_html(R["wt"], R["mut"]), unsafe_allow_html=True)
with t_wt:
    st.markdown(single_seq_html(R["wt"], "wt"), unsafe_allow_html=True)
with t_mut:
    st.markdown(single_seq_html(R["mut"], "mt"), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3D VISUALIZER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">🔬 3D Structure Visualizer — ESMFold</div>',
            unsafe_allow_html=True)

if py3Dmol is None:
    st.error("py3Dmol is not installed. Run: `pip install py3Dmol stmol`")
elif R["pdb_mut"] is None and R["pdb_wt"] is None:
    st.warning(
        f"⚠️ ESMFold API unreachable — structure rendering unavailable "
        f"(mutant: {R['err_mut']}; wild-type: {R['err_wt']}). "
        "Sequence analytics and metrics above remain fully functional. Press **RUN IMUSO ENGINE** to retry."
    )
else:
    if R["err_wt"]:
        st.warning(f"Wild-type structure unavailable ({R['err_wt']}). Showing mutant only.")
    if R["err_mut"]:
        st.warning(f"Mutant structure unavailable ({R['err_mut']}). Showing wild-type only.")

    mut_resi = [p + 1 for p, _, _ in R["muts"]]
    tabs, kinds = [], []
    if R["pdb_mut"]:
        tabs.append("🟢 Mutant (mutations highlighted)"); kinds.append("mut")
    if R["pdb_wt"]:
        tabs.append("🔵 Wild-Type"); kinds.append("wt")
    for tab, kind in zip(st.tabs(tabs), kinds):
        with tab:
            if kind == "mut":
                show_view(build_view(R["pdb_mut"], mut_resi, color_mode, show_surface, spin))
            else:
                show_view(build_view(R["pdb_wt"], [], color_mode, show_surface, spin))
    st.caption("Drag to rotate · scroll to zoom · right-drag to pan. "
               "Green sticks/spheres = mutated residues.")

    dl1, dl2 = st.columns(2)
    if R["pdb_mut"]:
        dl1.download_button("⬇ Download IMUSO Proprietary PDB Structure", R["pdb_mut"],
                            file_name="IMUSO_mutant.pdb", mime="chemical/x-pdb")
    if R["pdb_wt"]:
        dl2.download_button("⬇ Download Wild-Type Reference PDB", R["pdb_wt"],
                            file_name="IMUSO_wildtype.pdb", mime="chemical/x-pdb")
st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MUTATION LEDGER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📋 Mutation Ledger</div>', unsafe_allow_html=True)
if R["muts"]:
    df = pd.DataFrame(
        [{"Position": p + 1, "Wild-Type": o, "Mutant": n, "Notation": f"{o}{p+1}{n}",
          "Δ Hydropathy": round(KD[n] - KD[o], 2)} for p, o, n in R["muts"]]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Aggressiveness is 0% — the mutant is identical to the wild-type.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer" style="text-align:center">IMUSO AI Core · MVP build · '
    'Structure prediction via Meta ESMFold (ESM Atlas API)</div>',
    unsafe_allow_html=True,
)