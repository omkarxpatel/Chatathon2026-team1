"""Presenter console. Runs BESIDE the dashboard, not inside it.

    streamlit run app/presenter.py --server.port 8502

Audience sees the dashboard on 8501. The presenter sees this on their own
laptop: whose turn it is, what should be on screen, the beats, the one line
that has to land, and whether the run is ahead or behind.

Deliberately separate from app/dashboard.py -- it imports nothing from the
project, so it cannot break the demo and the demo cannot break it. Edit
SEGMENTS freely; it is the only thing in here worth changing.
"""

from __future__ import annotations

import time

import streamlit as st

TOTAL_BUDGET = 300  # 5:00 hard cap

# --------------------------------------------------------------------------
# The run sheet. Beats are glance-length on purpose -- a presenter console
# is read in half a second, not studied.
# --------------------------------------------------------------------------
SEGMENTS = [
    {
        "who": "M1",
        "title": "The problem, and the claim",
        "budget": 40,
        "screen": "Cohort view · nothing selected",
        "beats": [
            "DTC sports nutrition — people lapse quietly",
            "Obvious build: rank by risk, email the top",
            "**That is the thing we did not build**",
            "Point at the headline: 136 of 200 left alone",
        ],
        "land": ("Most of the customers our model flags should not be contacted. "
                 "Not contacting someone is the default here, not the failure case."),
        "handoff": "…so the interesting question is what earns a message — "
                   "and that starts with how the number gets made.",
    },
    {
        "who": "M2",
        "title": "The architecture, and why to believe it",
        "budget": 45,
        "screen": "Still cohort view · funnel + sidebar stats. No clicking.",
        "beats": [
            "Two layers, hard boundary",
            "Model gets structured behaviour · LLM gets the ticket text",
            "Logistic regression, 6 features, **exact** attributions (no SHAP)",
            "PR-AUC 0.825 · precision@20 80% · 47% base rate — say once, move on",
        ],
        "land": ("The LLM never computes the risk number. It receives the score "
                 "and the attributions as input."),
        "handoff": "So let's watch it run on someone it decided to contact.",
    },
    {
        "who": "M3",
        "title": "CUST-0001 — full trace",
        "budget": 65,
        "screen": "Case 1 → Signals → Policy gate → Agent → Guardrails",
        "beats": [
            "**Signals:** 64% MEDIUM · gap 2.81× · ran out ~20 days ago",
            "**Gate:** all four pass → eligible",
            "**Agent:** service_failure 88% · 2 unresolved damaged-shipment "
            "tickets, oldest 85 days",
            "Read one line of the quoted ticket — do not paraphrase",
            "**Guardrails:** clear. A discount requires a price diagnosis.",
        ],
        "land": ("They did not drift away. We broke something and then did not "
                 "answer. The right action is to fix it, not to discount it — "
                 "a discount here buys silence."),
        "handoff": "That one we acted on. Here's one the model also flagged — "
                   "and we did nothing.",
        "note": "You are also the driver for all five minutes.",
    },
    {
        "who": "M4",
        "title": "CUST-0002 — the gate",
        "budget": 55,
        "screen": "Case 2 → **Policy gate tab**. No Agent tab — say why.",
        "beats": [
            "36% risk **but gap 3.06×** — bigger than the one we just emailed",
            "5 lb tub → ~21 days of product still on hand",
            "The gate stops them. Not the model — **the gate**.",
            "Agent never ran. Zero tokens spent.",
            "One line: two more cases in the build — cooldown, price sensitivity",
        ],
        "land": ("A 55-day gap means nothing on a 76-day tub and everything on a "
                 "15-day one. The model handles uncertainty; the gate handles "
                 "certainty."),
        "handoff": "That's the deterministic layer catching what the model "
                   "missed. Now the other direction.",
    },
    {
        "who": "M5",
        "title": "CUST-0005 — the agent, caveats, close",
        "budget": 75,
        "screen": "Case 5 → Policy gate (briefly) → **Agent**",
        "beats": [
            "85% HIGH — the highest-risk case we show",
            "**Show the gate passing.** All four green. A rules-only system "
            "emails them right here.",
            "Agent read the ticket: torn rotator cuff, out 3 months. CSAT 5, 5.",
            "lifestyle_change → **no_action**",
            "**Caveats, one breath:** synthetic data · no uplift modelling · "
            "single split on 200 rows",
            "Close: no send path anywhere. Every message simulated, human "
            "approval required.",
        ],
        "land": ("The model says 85% and the model is right — they have churned. "
                 "Emailing them protein anyway is the failure mode this whole "
                 "system exists to avoid."),
        "handoff": "— end —",
    },
]

CUM_START = []
_acc = 0
for _s in SEGMENTS:
    CUM_START.append(_acc)
    _acc += _s["budget"]


# --------------------------------------------------------------------------
# clock
# --------------------------------------------------------------------------
def init() -> None:
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("total_accum", 0.0)
    st.session_state.setdefault("total_t0", 0.0)
    st.session_state.setdefault("seg_accum", 0.0)
    st.session_state.setdefault("seg_t0", 0.0)


def elapsed(prefix: str) -> float:
    acc = st.session_state[f"{prefix}_accum"]
    if st.session_state["running"]:
        acc += time.time() - st.session_state[f"{prefix}_t0"]
    return acc


def start_pause() -> None:
    if st.session_state["running"]:
        st.session_state["total_accum"] = elapsed("total")
        st.session_state["seg_accum"] = elapsed("seg")
        st.session_state["running"] = False
    else:
        now = time.time()
        st.session_state["total_t0"] = now
        st.session_state["seg_t0"] = now
        st.session_state["running"] = True


def reset() -> None:
    st.session_state.update(idx=0, running=False, total_accum=0.0,
                            total_t0=0.0, seg_accum=0.0, seg_t0=0.0)


def jump(i: int) -> None:
    """Move to a segment and restart that segment's clock."""
    st.session_state["idx"] = max(0, min(i, len(SEGMENTS) - 1))
    st.session_state["seg_accum"] = 0.0
    st.session_state["seg_t0"] = time.time()


def mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def signed(seconds: float) -> str:
    sign = "+" if seconds >= 0 else "−"
    return f"{sign}{mmss(abs(seconds))}"


# --------------------------------------------------------------------------
st.set_page_config(page_title="Presenter", layout="wide",
                   initial_sidebar_state="collapsed")
init()

st.markdown("""
<style>
  .block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px; }
  #MainMenu, footer, [data-testid="stStatusWidget"],
  [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"],
  [data-testid="stHeader"] { display: none !important; }

  .clock  { font-size: 4.2rem; font-weight: 720; line-height: 1;
            font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
  .clab   { font-size: .72rem; font-weight: 600; letter-spacing: .09em;
            text-transform: uppercase; color: #898781; }
  .drift  { font-size: 2.1rem; font-weight: 700; line-height: 1.1;
            font-variant-numeric: tabular-nums; }

  .who    { font-size: 3.1rem; font-weight: 740; letter-spacing: -.02em;
            line-height: 1; }
  .ttl    { font-size: 1.35rem; font-weight: 600; color: #52514e;
            margin-top: .25rem; }
  .screen { font-size: 1.05rem; background: #eef4fd; border: 1px solid #c8ddf7;
            border-radius: 9px; padding: .6rem .9rem; margin: .9rem 0 1.1rem;
            color: #12457f; }
  .beat   { font-size: 1.32rem; line-height: 1.5; margin: .5rem 0;
            padding-left: 1.1rem; border-left: 3px solid #e1e0d9; }
  .land   { font-size: 1.5rem; line-height: 1.45; font-weight: 600;
            background: #fffbe9; border: 1px solid #f2dfa4; border-radius: 10px;
            padding: 1rem 1.2rem; margin-top: 1.1rem; }
  .landlab{ font-size: .7rem; font-weight: 700; letter-spacing: .1em;
            text-transform: uppercase; color: #8a5d00; margin-bottom: .35rem; }
  .hand   { font-size: 1.02rem; color: #52514e; font-style: italic;
            margin-top: .9rem; padding-left: 1.1rem;
            border-left: 3px solid #c3c2b7; }
  .note   { font-size: .92rem; font-weight: 650; color: #a8431d;
            background: rgba(235,104,52,.10); border-radius: 7px;
            padding: .4rem .7rem; display: inline-block; margin-top: .5rem; }
  .dots   { display: flex; gap: .4rem; margin: .1rem 0 .3rem; }
  .dot    { flex: 1; height: 7px; border-radius: 999px; background: #e1e0d9; }
</style>
""", unsafe_allow_html=True)

seg = SEGMENTS[st.session_state["idx"]]


# --- clock strip: reruns once a second, rest of the page stays put --------
@st.fragment(run_every="1s")
def clock_strip() -> None:
    total = elapsed("total")
    seg_e = elapsed("seg")
    drift = total - (CUM_START[st.session_state["idx"]] + seg_e)

    c1, c2, c3 = st.columns([1.15, 1, 1.15])

    over_total = total > TOTAL_BUDGET
    c1.markdown(
        f"<div class='clab'>Total</div>"
        f"<div class='clock' style='color:{"#d03b3b" if over_total else "#0b0b0b"}'>"
        f"{mmss(total)}</div>"
        f"<div class='clab' style='margin-top:.2rem'>of {mmss(TOTAL_BUDGET)}</div>",
        unsafe_allow_html=True)

    over_seg = seg_e > seg["budget"]
    c2.markdown(
        f"<div class='clab'>This segment</div>"
        f"<div class='clock' style='font-size:3.2rem;color:"
        f"{"#d03b3b" if over_seg else "#0b0b0b"}'>{mmss(seg_e)}</div>"
        f"<div class='clab' style='margin-top:.2rem'>budget {seg['budget']}s</div>",
        unsafe_allow_html=True)

    # The number that actually matters mid-run.
    if drift > 12:
        col, word = "#d03b3b", "behind"
    elif drift < -12:
        col, word = "#1baf7a", "ahead"
    else:
        col, word = "#0ca30c", "on time"
    c3.markdown(
        f"<div class='clab'>Pace</div>"
        f"<div class='drift' style='color:{col}'>{signed(drift)}</div>"
        f"<div class='clab' style='margin-top:.25rem;color:{col}'>{word}</div>",
        unsafe_allow_html=True)


clock_strip()

# --- transport ------------------------------------------------------------
b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
b1.button("Pause" if st.session_state["running"] else "Start",
          width="stretch", type="primary", on_click=start_pause)
b2.button("Prev", width="stretch", shortcut="Left",
          disabled=st.session_state["idx"] == 0,
          on_click=jump, args=(st.session_state["idx"] - 1,))
b3.button("Next", width="stretch", shortcut="Right",
          disabled=st.session_state["idx"] == len(SEGMENTS) - 1,
          on_click=jump, args=(st.session_state["idx"] + 1,))
b4.button("Reset", width="stretch", on_click=reset)

st.markdown(
    "<div class='dots'>"
    + "".join(
        f"<div class='dot' style='background:"
        f"{'#2a78d6' if i == st.session_state['idx'] else ('#b7d3f6' if i < st.session_state['idx'] else '#e1e0d9')}'>"
        f"</div>"
        for i in range(len(SEGMENTS))
    )
    + "</div>",
    unsafe_allow_html=True,
)

# --- the segment ----------------------------------------------------------
st.markdown(
    f"<div class='who'>{seg['who']}</div><div class='ttl'>{seg['title']}</div>",
    unsafe_allow_html=True)
if seg.get("note"):
    st.markdown(f"<div class='note'>{seg['note']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='screen'><b>ON SCREEN &nbsp;·&nbsp;</b> {seg['screen']}</div>",
            unsafe_allow_html=True)

left, right = st.columns([1.25, 1])
with left:
    for beat in seg["beats"]:
        st.markdown(f"<div class='beat'>{beat}</div>", unsafe_allow_html=True)
with right:
    st.markdown(
        f"<div class='land'><div class='landlab'>Land this</div>"
        f"“{seg['land']}”</div>",
        unsafe_allow_html=True)
    st.markdown(f"<div class='hand'>Handoff: {seg['handoff']}</div>",
                unsafe_allow_html=True)
