import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# PREPAY BLUEPRINT — mortgage prepayment predictor (RBF-SVM)
# Same model + same inputs as before, new UI.
# ============================================================

st.set_page_config(
    page_title="Prepay Blueprint",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "mortgage_prepayment_final.pkl"
THRESHOLD_PATH = ROOT / "models" / "final_threshold.pkl"

if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
    st.error("Required model files are missing from the models folder.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_artifacts():
    return joblib.load(MODEL_PATH), float(joblib.load(THRESHOLD_PATH))


model, threshold = load_artifacts()

# ============================================================
# PROJECT CONSTANTS (from the notebook)
# ============================================================

NUMERICAL_COLS = [
    "credit_score", "first_payment_date", "maturity_date", "msa",
    "mortgage_insurance_percentage", "number_of_units", "original_cltv",
    "original_dti", "original_upb", "original_ltv", "original_interest_rate",
    "postal_code", "original_loan_term", "number_of_borrowers",
]
CATEGORICAL_COLS = [
    "first_time_homebuyer_flag", "occupancy_status", "channel",
    "prepayment_penalty_mortgage_flag", "amortization_type", "property_state",
    "property_type", "loan_purpose", "seller_name", "servicer_name",
    "special_eligibility_program", "property_valuation_method",
    "relief_refinance_indicator", "interest_only_indicator",
]
COLS = NUMERICAL_COLS + CATEGORICAL_COLS

VAL_F1 = {
    "RBF SVM": 0.4304,
    "Neural Network": 0.4294,
    "Logistic Regression": 0.4281,
    "Random Forest": 0.4194,
}
TEST_METRICS = {
    "Accuracy": 0.7475,
    "Precision": 0.3722,
    "Recall": 0.5477,
    "F1": 0.4432,
    "ROC-AUC": 0.7393,
}
# Final test confusion matrix from the notebook: rows = actual, cols = predicted
CM = np.array([[6470, 1695], [830, 1005]])
TN, FP, FN, TP = CM[0, 0], CM[0, 1], CM[1, 0], CM[1, 1]

# Values below are the ones that actually appear in the Freddie Mac 2024 sample
STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "GU",
    "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI",
    "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY",
    "OH", "OK", "OR", "PA", "PR", "RI", "SC", "SD", "TN", "TX", "UT", "VA",
    "VI", "VT", "WA", "WI", "WV", "WY",
]
SELLERS = [
    "OTHER", "AMERIHOME MORTGAGE COMPANY, LLC", "CITIMORTGAGE, INC.", "CITIZENS BANK, NA",
    "CMG MORTGAGE, INC.", "CROSSCOUNTRY MORTGAGE, LLC", "FAIRWAY INDEPENDENT MORTGAGE CORPORATION",
    "FLAGSTAR BANK, NATIONAL ASSOCIATION", "FREEDOM MORTGAGE CORPORATION", "GUARANTEED RATE, INC.",
    "GUILD MORTGAGE COMPANY LLC", "JPMORGAN CHASE BANK, NATIONAL ASSOCIATION", "LOANDEPOT.COM, LLC",
    "NATIONSTAR MORTGAGE LLC DBA MR. COOPER", "NEWREZ LLC", "PENNYMAC CORP.",
    "PENNYMAC LOAN SERVICES, LLC", "PHH MORTGAGE CORPORATION", "ROCKET MORTGAGE, LLC", "TRUIST BANK",
    "U.S. BANK N.A.", "UNION HOME MORTGAGE CORPORATION", "UNITED WHOLESALE MORTGAGE, LLC",
    "WELLS FARGO BANK, N.A.",
]
# These five columns have ONE value across all 50,000 loans, so they carry no
# information and are filled in automatically instead of being form inputs.
CONSTANTS = dict(
    prepayment_penalty_mortgage_flag="N",
    amortization_type="FRM",
    special_eligibility_program="N",
    property_valuation_method="N",
    interest_only_indicator=9999,
)

# ---- form defaults = typical loan (dataset medians / most common values)
DEFAULTS = dict(
    credit_score=762, original_dti=40.0, number_of_borrowers=1,
    original_upb=300000.0, original_interest_rate=6.625, original_loan_term=360,
    original_ltv=80.0, original_cltv=80.0, mortgage_insurance_percentage=0,
    property_state="TX", occupancy_status="P", property_type="SF",
    number_of_units=1, msa=29820, postal_code=481,
    first_time_homebuyer_flag="N", channel="R", loan_purpose="P",
    seller_name="OTHER", servicer_name="N", relief_refinance_indicator=2,
    fp_year=2024, fp_month=9,
)

# ---- four real loans from the test split (random_state=42) with what really happened
REAL_LOANS = {
    "Real loan 1 (OH condo)": dict(actual=1, values=dict(
        credit_score=806, original_dti=48.0, number_of_borrowers=2, original_upb=338000.0,
        original_interest_rate=7.05, original_ltv=90.0, original_cltv=90.0,
        mortgage_insurance_percentage=25, property_state="OH", property_type="CO",
        msa=18140, postal_code=430, seller_name="OTHER", fp_year=2024, fp_month=9)),
    "Real loan 2 (NY house)": dict(actual=0, values=dict(
        credit_score=807, original_dti=36.0, number_of_borrowers=1, original_upb=487000.0,
        original_interest_rate=6.5, original_ltv=75.0, original_cltv=75.0,
        property_state="NY", property_type="SF", msa=28740, postal_code=124,
        seller_name="U.S. BANK N.A.", fp_year=2024, fp_month=12)),
    "Real loan 3 (IL refi)": dict(actual=1, values=dict(
        credit_score=752, original_dti=17.0, number_of_borrowers=1, original_upb=322000.0,
        original_interest_rate=7.375, original_ltv=73.0, original_cltv=73.0,
        property_state="IL", property_type="CO", msa=16984, postal_code=606, loan_purpose="N",
        seller_name="JPMORGAN CHASE BANK, NATIONAL ASSOCIATION", fp_year=2024, fp_month=5)),
    "Real loan 4 (NV investor)": dict(actual=0, values=dict(
        credit_score=749, original_dti=48.0, number_of_borrowers=2, original_upb=334000.0,
        original_interest_rate=7.625, original_ltv=75.0, original_cltv=75.0,
        property_state="NV", occupancy_status="I", number_of_units=4, msa=29820, postal_code=891,
        channel="C", seller_name="UNITED WHOLESALE MORTGAGE, LLC", fp_year=2024, fp_month=8)),
}

for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("preset", "Custom")


def apply_preset():
    name = st.session_state.preset
    if name in REAL_LOANS:
        for k, v in {**DEFAULTS, **REAL_LOANS[name]["values"]}.items():
            st.session_state[k] = v


def matching_real_loan():
    """Actual outcome (1/0) if the submitted form is exactly one of the real loans, else None."""
    name = st.session_state.get("preset")
    if name in REAL_LOANS:
        full = {**DEFAULTS, **REAL_LOANS[name]["values"]}
        if all(st.session_state[k] == v for k, v in full.items()):
            return REAL_LOANS[name]["actual"]
    return None


def build_row(ss):
    """Form values -> the exact 28 columns / types the model was trained on."""
    term = int(ss["original_loan_term"])
    first = int(ss["fp_year"]) * 12 + (int(ss["fp_month"]) - 1)
    mat = first + term - 1                      # in the data: maturity = first payment + term - 1 month
    row = {
        "credit_score": ss["credit_score"],
        "first_payment_date": int(ss["fp_year"]) * 100 + int(ss["fp_month"]),
        "maturity_date": (mat // 12) * 100 + (mat % 12 + 1),
        "msa": ss["msa"] if ss["msa"] else np.nan,   # blank in the data when not in an MSA
        "mortgage_insurance_percentage": ss["mortgage_insurance_percentage"],
        "number_of_units": ss["number_of_units"],
        "original_cltv": ss["original_cltv"],
        "original_dti": ss["original_dti"],
        "original_upb": ss["original_upb"],
        "original_ltv": ss["original_ltv"],
        "original_interest_rate": ss["original_interest_rate"],
        "postal_code": ss["postal_code"],
        "original_loan_term": term,
        "number_of_borrowers": ss["number_of_borrowers"],
        "first_time_homebuyer_flag": ss["first_time_homebuyer_flag"],
        "occupancy_status": ss["occupancy_status"],
        "channel": ss["channel"],
        "property_state": ss["property_state"],
        "property_type": ss["property_type"],
        "loan_purpose": ss["loan_purpose"],
        "seller_name": ss["seller_name"],
        "servicer_name": ss["servicer_name"],
        "relief_refinance_indicator": int(ss["relief_refinance_indicator"]),
        **CONSTANTS,
    }
    return row


# ============================================================
# STYLE
# ============================================================

INK = "#0b2447"
PANEL = "#071a33"
LINE = "#8fb8ff"
PAPER = "#eaf2ff"
ORANGE = "#ff7a3d"
MINT = "#5eead4"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;700;800;900&family=Figtree:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #0b2447;
    background-image:
        linear-gradient(rgba(143,184,255,.075) 1px, transparent 1px),
        linear-gradient(90deg, rgba(143,184,255,.075) 1px, transparent 1px),
        linear-gradient(rgba(143,184,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(143,184,255,.035) 1px, transparent 1px);
    background-size: 120px 120px, 120px 120px, 24px 24px, 24px 24px;
    color: #eaf2ff;
    font-family: 'Figtree', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1240px; padding-top: 1.2rem; padding-bottom: 4rem; }

h1, h2, h3 {
    font-family: 'Big Shoulders Display', sans-serif !important;
    font-weight: 800 !important;
    color: #eaf2ff !important;
    letter-spacing: .005em;
}
h2 { font-size: 2.7rem !important; line-height: 1 !important; margin-top: .6rem !important; }
h3 { font-size: 1.9rem !important; }

[data-testid="stWidgetLabel"] p, label p { color: #a9c4ee !important; font-weight: 500; font-size: 13.5px; }
.stCaption, [data-testid="stCaptionContainer"] { color: #8fb0e0 !important; }

/* tabs */
button[data-baseweb="tab"] { background: transparent; }
button[data-baseweb="tab"] p {
    font-family: 'Big Shoulders Display', sans-serif !important;
    font-size: 22px !important; font-weight: 700 !important; color: #8fb8ff !important;
}
button[data-baseweb="tab"][aria-selected="true"] p { color: #ff7a3d !important; }
div[data-baseweb="tab-highlight"] { background: #ff7a3d !important; height: 3px !important; }
div[data-baseweb="tab-border"] { background: rgba(143,184,255,.22) !important; }

/* form panel */
[data-testid="stForm"] {
    border: 1px dashed rgba(143,184,255,.4);
    border-radius: 3px;
    background: rgba(5,22,45,.74);
    padding: 20px 22px 24px;
}

/* inputs */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="select"] > div {
    background: #071a33 !important;
    border-radius: 2px !important;
    border-color: rgba(143,184,255,.32) !important;
}
div[data-baseweb="input"] input { color: #eaf2ff !important; font-weight: 500; }
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
    border-color: #ff7a3d !important;
}
[data-testid="stNumberInput"] button { background: #0d2a52 !important; color: #8fb8ff !important; }

/* buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background: #ff7a3d; color: #1a0a02; border: none; border-radius: 2px;
    min-height: 56px; width: 100%;
    box-shadow: 4px 4px 0 rgba(143,184,255,.35);
    transition: transform .12s ease, box-shadow .12s ease, background .12s ease;
}
.stButton > button p, [data-testid="stFormSubmitButton"] > button p {
    font-family: 'Big Shoulders Display', sans-serif !important;
    font-size: 26px !important; font-weight: 800 !important; color: #1a0a02 !important;
    letter-spacing: .02em;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    background: #ff9560; transform: translate(-2px,-2px); box-shadow: 6px 6px 0 rgba(143,184,255,.45);
}
.stButton > button:active, [data-testid="stFormSubmitButton"] > button:active {
    transform: translate(2px,2px); box-shadow: 1px 1px 0 rgba(143,184,255,.35);
}
button:focus-visible { outline: 2px solid #5eead4 !important; outline-offset: 2px; }

/* preset radio */
div[role="radiogroup"] { gap: 6px 18px; }
div[role="radiogroup"] label p { color: #cfe0ff !important; font-size: 14px; }

/* slider */
div[data-baseweb="slider"] div[role="slider"] { background: #ff7a3d !important; border-color: #ff7a3d !important; }
[data-testid="stSliderThumbValue"] { color: #ff7a3d !important; }

/* custom blocks */
.stat { border-left: 3px solid #8fb8ff; padding: 2px 0 4px 14px; }
.stat.hot { border-left-color: #ff7a3d; }
.stat.mint { border-left-color: #5eead4; }
.stat .v { font: 800 46px/1 'Big Shoulders Display', sans-serif; color: #eaf2ff; }
.stat .l { font-size: 13.5px; color: #a9c4ee; margin-top: 4px; }
.stat .s { font-size: 12.5px; color: #7f9fd1; margin-top: 2px; }

.note {
    border-left: 3px solid rgba(143,184,255,.5);
    background: rgba(5,22,45,.6);
    padding: 14px 18px; color: #c3d6f5; font-size: 14.5px; line-height: 1.6; max-width: 820px;
}
.note b { color: #eaf2ff; }

.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px 22px; margin-top: 8px; }
.step { border-top: 2px solid rgba(143,184,255,.4); padding: 14px 4px 18px 0; position: relative; }
.step:before {
    content: ""; position: absolute; top: -6px; left: 0; width: 10px; height: 10px;
    background: #ff7a3d; border-radius: 50%;
}
.step .sn { font: 800 44px/1 'Big Shoulders Display', sans-serif; color: #ff7a3d; }
.step .st { font: 700 24px/1.1 'Big Shoulders Display', sans-serif; margin-top: 6px; color: #eaf2ff; }
.step .sd { font-size: 14px; color: #c3d6f5; margin-top: 6px; line-height: 1.45; }
.step .sx { font-size: 12.5px; color: #7f9fd1; margin-top: 4px; }
@media (max-width: 900px) { .steps { grid-template-columns: repeat(2, 1fr); } }

.foot { color: #6f8fc4; font-size: 12px; margin-top: 56px; border-top: 1px dashed rgba(143,184,255,.25); padding-top: 14px; }
</style>
""", unsafe_allow_html=True)

FONT_LINK = ('<link href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;700;800;900'
             '&family=Figtree:wght@400;500;600;700&display=swap" rel="stylesheet">')

# ============================================================
# HERO — the roof is drawn in line-art once on load
# ============================================================

HERO_HTML = FONT_LINK + """
<style>
*{box-sizing:border-box}
body{margin:0;background:transparent;color:#eaf2ff;font-family:Figtree,sans-serif}
.wrap{display:grid;grid-template-columns:1.25fr .75fr;gap:24px;align-items:center;height:360px}
h1{font:800 clamp(56px,8.4vw,112px)/.86 'Big Shoulders Display',sans-serif;margin:0;letter-spacing:-.005em}
h1 em{font-style:normal;color:#ff7a3d}
p{max-width:520px;color:#b9cff3;font-size:16px;line-height:1.6;margin:18px 0 14px}
.live{display:flex;align-items:center;gap:9px;font-size:13.5px;color:#8fb0e0}
.dot{width:9px;height:9px;border-radius:50%;background:#5eead4;box-shadow:0 0 0 0 rgba(94,234,212,.6)}
svg{width:100%;height:auto;max-height:340px;overflow:visible}
.d{fill:none;stroke:#8fb8ff;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round;
   stroke-dasharray:1;stroke-dashoffset:1;animation:draw 1.1s cubic-bezier(.5,.1,.2,1) forwards;animation-delay:var(--t,0s)}
.roof{stroke:#ff7a3d;stroke-width:2.6}
.dim{stroke:#8fb8ff;stroke-width:1;opacity:.75}
svg text{font:500 11px Figtree,sans-serif;fill:#8fb8ff;opacity:0;animation:fade .5s ease forwards;animation-delay:1.5s}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes fade{to{opacity:1}}
@keyframes ping{70%{box-shadow:0 0 0 9px rgba(94,234,212,0)}100%{box-shadow:0 0 0 0 rgba(94,234,212,0)}}
.dot{animation:ping 2.2s infinite}
@media (max-width:760px){.wrap{grid-template-columns:1fr;height:auto;padding:10px 0 4px}.art{display:none}}
@media (prefers-reduced-motion:reduce){.d{animation:none;stroke-dashoffset:0}svg text{animation:none;opacity:1}.dot{animation:none}}
</style>
<div class="wrap">
  <div>
    <h1>Will this loan<br>walk away <em>early?</em></h1>
    <p>Enter a borrower and loan profile. An RBF-SVM trained on 50,000 Freddie Mac 2024 loans scores it
       against a cut-off tuned on held-out data.</p>
    <div class="live"><span class="dot"></span>Model loaded, 28 input features</div>
  </div>
  <div class="art">
    <svg viewBox="0 0 400 330" role="img" aria-label="Line drawing of a house with dimension marks">
      <path class="d" pathLength="1" style="--t:.0s" d="M20 270H380"/>
      <path class="d" pathLength="1" style="--t:.15s" d="M80 270V150H300V270"/>
      <path class="d" pathLength="1" style="--t:.35s" d="M250 107V74H274V122"/>
      <path class="d" pathLength="1" style="--t:.45s" d="M170 270V200H210V270"/>
      <circle class="d" pathLength="1" style="--t:.7s" cx="202" cy="238" r="2.5"/>
      <rect class="d" pathLength="1" style="--t:.55s" x="105" y="180" width="40" height="36"/>
      <path class="d" pathLength="1" style="--t:.7s" d="M125 180V216M105 198H145"/>
      <rect class="d" pathLength="1" style="--t:.6s" x="235" y="180" width="40" height="36"/>
      <path class="d" pathLength="1" style="--t:.75s" d="M255 180V216M235 198H275"/>
      <path class="d roof" pathLength="1" style="--t:.8s" d="M60 150L190 70L320 150"/>
      <path class="d dim" pathLength="1" style="--t:1s" d="M80 294H300M80 287V301M300 287V301"/>
      <path class="d dim" pathLength="1" style="--t:1.05s" d="M338 150V270M331 150H345M331 270H345"/>
      <text x="190" y="318" text-anchor="middle">term</text>
      <text x="358" y="214">UPB</text>
      <text x="46" y="108" text-anchor="end">rate</text>
    </svg>
  </div>
</div>
"""

# ============================================================
# RESULT CARD (ruler with an animated needle)
# ============================================================

RESULT_TEMPLATE = FONT_LINK + """
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Figtree,sans-serif;color:#eaf2ff;background:transparent}
.card{border:1px solid rgba(143,184,255,.38);background:rgba(5,22,45,.8);padding:28px 28px 22px;position:relative}
.card:before{content:"";position:absolute;left:-1px;top:-1px;width:64px;height:5px;background:__ACC__}
h2{font:800 clamp(42px,6.2vw,62px)/.92 'Big Shoulders Display',sans-serif;margin:6px 0 12px;color:__ACC__}
p{margin:0;line-height:1.55;color:#c3d6f5;font-size:15px}
p b{color:#eaf2ff}
svg{width:100%;height:auto;margin-top:12px;overflow:visible}
svg text{font:500 11px Figtree,sans-serif;fill:#8fb0e0}
svg .cap{font-size:12px;fill:#a9c4ee}
.row{display:flex;gap:30px;flex-wrap:wrap;margin-top:12px;border-top:1px dashed rgba(143,184,255,.3);padding-top:16px}
.n{font:800 38px/1 'Big Shoulders Display',sans-serif}
.l{font-size:12.5px;color:#8fb0e0;margin-top:3px}
#needle{transition:transform 1.1s cubic-bezier(.2,.9,.2,1)}
@media (prefers-reduced-motion:reduce){#needle{transition:none}}
</style>
<div class="card">
  <h2>__TITLE__</h2>
  <p>__BODY__</p>
  __ACTUAL__
  <svg viewBox="0 0 600 168" role="img" aria-label="Decision score ruler">
    <rect x="30" y="70" width="__XT_W__" height="14" fill="#5eead4" opacity=".85"/>
    <rect x="__XT__" y="70" width="__XR_W__" height="14" fill="#ff7a3d" opacity=".9"/>
    <g stroke="#8fb8ff" stroke-width="1">__TICKS__</g>
    <line x1="__XT__" x2="__XT__" y1="58" y2="132" stroke="#eaf2ff" stroke-width="1.6" stroke-dasharray="4 3"/>
    <text x="__XT__" y="146" text-anchor="middle" style="fill:#eaf2ff">cut-off __THR__</text>
    <text class="cap" x="30" y="164">not flagged</text>
    <text class="cap" x="570" y="164" text-anchor="end">flagged as likely prepay</text>
    __NEEDLE__
  </svg>
  <div class="row">__STATS__</div>
</div>
<script>
const n=document.getElementById('needle');
if(n){requestAnimationFrame(()=>requestAnimationFrame(()=>{n.style.transform='translate(__XS__px,0px)'}));}
</script>
"""


def result_card(score, thr, actual=None):
    lo, hi = -1.6, 1.6
    if score is not None:
        lo = min(lo, math.floor((score - 0.25) * 2) / 2)
        hi = max(hi, math.ceil((score + 0.25) * 2) / 2)
    X0, X1 = 30.0, 570.0

    def X(v):
        return X0 + (v - lo) / (hi - lo) * (X1 - X0)

    xt = X(thr)
    ticks, v = "", math.ceil(lo * 2) / 2
    while v <= hi + 1e-9:
        x, major = X(v), abs(v - round(v)) < 1e-9
        ticks += f'<line x1="{x:.1f}" x2="{x:.1f}" y1="88" y2="{95 if major else 92}"/>'
        if major:
            ticks += f'<text x="{x:.1f}" y="108" text-anchor="middle">{v:g}</text>'
        v += 0.5

    if score is None:
        acc, title = LINE, "Waiting for a loan"
        body = ("Fill in the form and hit <b>Score this loan</b>. A needle will land on the ruler: "
                "left of the dashed line is not flagged, right of it is flagged.")
        needle, xs, actual_html = "", xt, ""
        stats = (f'<div><div class="n">{thr:.3f}</div><div class="l">cut-off from validation</div></div>')
    else:
        flagged = score >= thr
        acc = ORANGE if flagged else MINT
        title = "Prepayment signal" if flagged else "No prepayment signal"
        body = ("Decision score is above the selected threshold." if flagged
                else "Decision score is below the selected threshold.")
        body += " This is an SVM decision score, not a probability."
        actual_html = ""
        if actual is not None:
            verdict = "right" if bool(actual) == flagged else "wrong"
            actual_html = (
                f'<p style="margin-top:10px;padding:10px 12px;border-left:3px solid {acc};'
                f'background:rgba(143,184,255,.08)">In the dataset this loan '
                f'<b>{"prepaid" if actual else "did not prepay"}</b>, so the model called it '
                f'<b>{verdict}</b>.</p>')
        xs = X(score)
        needle = (
            f'<g id="needle" style="transform:translate({xt:.1f}px,0px)">'
            f'<polygon points="0,68 -9,50 9,50" fill="{acc}"/>'
            f'<line x1="0" x2="0" y1="68" y2="86" stroke="{acc}" stroke-width="2.4"/>'
            f'<text x="0" y="40" text-anchor="middle" style="fill:{acc};font:800 22px \'Big Shoulders Display\',sans-serif">'
            f'{score:.3f}</text></g>'
        )
        margin = score - thr
        stats = (
            f'<div><div class="n" style="color:{acc}">{score:.3f}</div><div class="l">decision score</div></div>'
            f'<div><div class="n">{thr:.3f}</div><div class="l">cut-off</div></div>'
            f'<div><div class="n">{margin:+.3f}</div><div class="l">'
            f'{"past" if margin >= 0 else "short of"} the cut-off</div></div>'
        )

    html = (RESULT_TEMPLATE
            .replace("__ACC__", acc).replace("__TITLE__", title).replace("__BODY__", body)
            .replace("__XT_W__", f"{xt - X0:.1f}").replace("__XR_W__", f"{X1 - xt:.1f}")
            .replace("__XT__", f"{xt:.1f}").replace("__THR__", f"{thr:.3f}")
            .replace("__TICKS__", ticks).replace("__NEEDLE__", needle)
            .replace("__STATS__", stats).replace("__ACTUAL__", actual_html).replace("__XS__", f"{xs:.1f}"))
    return html


# ============================================================
# HELPERS
# ============================================================

def stat(value, label, sub="", tone=""):
    s = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="stat {tone}"><div class="v">{value}</div><div class="l">{label}</div>{s}</div>'


def style_fig(fig, h=380):
    grid = "rgba(143,184,255,.14)"
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, sans-serif", color="#cfe0ff", size=13),
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)"),
        title=dict(font=dict(family="Big Shoulders Display, sans-serif", size=24, color=PAPER), x=0),
    )
    fig.update_xaxes(gridcolor=grid, zeroline=False, linecolor="rgba(143,184,255,.4)")
    fig.update_yaxes(gridcolor=grid, zeroline=False, linecolor="rgba(143,184,255,.4)")
    return fig


def show_html(html, height):
    """Render an HTML snippet in an iframe (works on old and new Streamlit)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def show_chart(fig, container=st):
    try:
        container.plotly_chart(fig, width="stretch")
    except TypeError:
        container.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE
# ============================================================

show_html(HERO_HTML, 370)

tab_score, tab_report, tab_method = st.tabs(["Score a loan", "How good is it", "How it was built"])

# ------------------------------------------------------------
# TAB 1 — SCORE A LOAN
# ------------------------------------------------------------
with tab_score:
    st.radio(
        "Or start from a real loan in the dataset",
        ["Custom", *REAL_LOANS.keys()],
        horizontal=True, key="preset", on_change=apply_preset,
    )

    left, right = st.columns([1.12, .88], gap="large")

    with left:
        with st.form("loan_form", border=True):
            t_num, t_cat = st.tabs(["Numerical features", "Categorical features"])

            with t_num:
                a, b, c = st.columns(3)
                a.number_input("credit_score", 500, 850, key="credit_score", help="Dataset range: 591 to 830")
                b.selectbox("first_payment_date (year)", [2024, 2025, 2026], key="fp_year")
                c.selectbox("first_payment_date (month)", list(range(1, 13)), key="fp_month",
                            format_func=lambda m: f"{m:02d}")
                a, b, c = st.columns(3)
                a.number_input("msa", 0, 99999, key="msa", help="0 = blank in the data (not in an MSA)")
                b.selectbox("mortgage_insurance_percentage", [0, 6, 12, 16, 18, 25, 30, 35],
                            key="mortgage_insurance_percentage", help="Only values present in the data")
                c.number_input("number_of_units", 1, 4, key="number_of_units")
                a, b, c = st.columns(3)
                a.number_input("original_cltv", 1.0, 170.0, key="original_cltv")
                b.number_input("original_dti", 1.0, 60.0, key="original_dti", help="Dataset range: 1 to 50")
                c.number_input("original_upb", 10000.0, 2000000.0, step=1000.0, key="original_upb",
                               help="Dataset range: 11,000 to 1,980,000")
                a, b, c = st.columns(3)
                a.number_input("original_ltv", 1.0, 170.0, key="original_ltv")
                b.number_input("original_interest_rate", 2.5, 10.0, step=0.05, format="%.3f",
                               key="original_interest_rate", help="Dataset range: 2.99 to 9.18")
                c.number_input("postal_code", 0, 999, key="postal_code",
                               help="Dataset stores the first 3 digits only (7 to 999)")
                a, b, c = st.columns(3)
                a.number_input("original_loan_term", 60, 360, key="original_loan_term",
                               help="Dataset: mostly 360, then 180, 240, 120")
                b.number_input("number_of_borrowers", 1, 5, key="number_of_borrowers")

            with t_cat:
                a, b, c = st.columns(3)
                a.selectbox("first_time_homebuyer_flag", ["N", "Y"], key="first_time_homebuyer_flag")
                b.selectbox("occupancy_status", ["P", "I", "S"], key="occupancy_status")
                c.selectbox("channel", ["R", "C", "B"], key="channel")
                a, b, c = st.columns(3)
                a.selectbox("property_state", STATES, key="property_state")
                b.selectbox("property_type", ["SF", "PU", "CO", "MH", "CP"], key="property_type")
                c.selectbox("loan_purpose", ["P", "C", "N"], key="loan_purpose")
                a, b, c = st.columns(3)
                a.selectbox("servicer_name", ["N", "Y"], key="servicer_name")
                b.selectbox("relief_refinance_indicator", [1, 2, 4], key="relief_refinance_indicator")
                st.selectbox("seller_name", SELLERS, key="seller_name")

            st.caption("Not shown: prepayment_penalty_mortgage_flag, amortization_type, "
                       "special_eligibility_program, property_valuation_method and interest_only_indicator "
                       "have a single value in all 50,000 loans, so they are filled in automatically. "
                       "maturity_date is worked out from the first payment and the term.")
            submitted = st.form_submit_button("Score this loan")

        if submitted:
            row = build_row(st.session_state)
            try:
                df = pd.DataFrame([row], columns=COLS)
                score = float(model.decision_function(df)[0])
                st.session_state["last"] = dict(row=row, score=score, actual=matching_real_loan())
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    last = st.session_state.get("last")

    with right:
        show_html(result_card(last["score"] if last else None, threshold,
                          last["actual"] if last else None), 520)

# ------------------------------------------------------------
# TAB 2 — MODEL REPORT
# ------------------------------------------------------------
with tab_report:
    st.markdown("## Test-set results")
    st.caption("10,000 loans the model never saw while training or choosing its cut-off.")

    cols = st.columns(5)
    tones = ["", "", "mint", "hot", ""]
    for col, (name, val), tone in zip(cols, TEST_METRICS.items(), tones):
        col.markdown(stat(f"{val:.1%}", name, "", tone), unsafe_allow_html=True)

    st.markdown("")
    cm_col, bar_col = st.columns(2, gap="large")

    with cm_col:
        rates_cm = CM / CM.sum(axis=1, keepdims=True)
        text = [[f"{CM[i, j]:,}<br>{rates_cm[i, j]:.0%} of row" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(
            z=rates_cm, x=["Not flagged", "Flagged"], y=["Actually stayed", "Actually prepaid"],
            text=text, texttemplate="%{text}", textfont=dict(size=17, color=PAPER),
            colorscale=[[0, "#0c2d5a"], [1, "#2f6fd6"]], showscale=False, xgap=5, ygap=5,
        ))
        style_fig(fig, 380)
        fig.update_layout(title="Confusion matrix")
        fig.update_yaxes(autorange="reversed")
        show_chart(fig)

    with bar_col:
        names = list(VAL_F1.keys())[::-1]
        vals = [VAL_F1[n] for n in names]
        fig = go.Figure(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=[ORANGE if n == "RBF SVM" else LINE for n in names],
            text=[f"{v:.2%}" for v in vals], textposition="outside", cliponaxis=False,
        ))
        style_fig(fig, 380)
        fig.update_layout(title="Validation F1 by model")
        fig.update_xaxes(range=[0, 0.55], tickformat=".0%")
        show_chart(fig)

# ------------------------------------------------------------
# TAB 3 — METHOD
# ------------------------------------------------------------
with tab_method:
    st.markdown("## Raw data to deployed model")
    st.caption("Eight steps, straight from the notebook.")
    steps = [
        ("1", "Data", "Freddie Mac 2024 sample", "50,000 loans"),
        ("2", "Target", "Zero balance code 01 means prepaid", "18.35% positive"),
        ("3", "Features", "14 numerical + 14 categorical", "28 inputs"),
        ("4", "Split", "Stratified 60 / 20 / 20", "train / validation / test"),
        ("5", "Preprocess", "Impute, scale, one-hot encode", "one sklearn pipeline"),
        ("6", "Models", "Logistic, RBF-SVM, random forest, neural net", "4 candidates"),
        ("7", "Select", "Best validation F1, with a tuned cut-off", f"RBF-SVM, cut-off {threshold:.3f}"),
        ("8", "Test", "One independent evaluation", f"F1 {TEST_METRICS['F1']:.2%}"),
    ]
    st.markdown(
        '<div class="steps">' + "".join(
            f'<div class="step"><div class="sn">{n}</div><div class="st">{t}</div>'
            f'<div class="sd">{d}</div><div class="sx">{x}</div></div>' for n, t, d, x in steps
        ) + "</div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown(
        '<div class="note"><b>Decision score is not a probability.</b> The SVM uses a decision boundary '
        'whose threshold was selected on the validation set.</div>',
        unsafe_allow_html=True)

st.markdown(
    '<div class="foot">Prepay Blueprint · Freddie Mac 2024 sample · RBF-SVM · B.Tech machine learning project</div>',
    unsafe_allow_html=True)
