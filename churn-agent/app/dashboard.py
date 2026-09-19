"""Streamlit dashboard. Two views: the cohort, and one customer in full.

Run with:   streamlit run app/dashboard.py

Thin by design. Everything shown comes from pipeline.run_cohort(); there is
no logic here beyond layout, and there is no send button, because there is
no send path anywhere in this project to wire one to.

Colour is doing a job everywhere it appears: risk bands carry the fixed
status palette (never a categorical hue), the funnel is one sequential blue
ramp, and attributions are a diverging blue/red pair around a neutral zero.
Tokens live in app/theme.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import config as cfg
from agent.schemas import Action
from app import theme as T
from data.store import EventStore
from features.extract import FEATURE_LABELS, FEATURE_NAMES, MODEL_FEATURES
from pipeline import CohortRun, run_cohort

st.set_page_config(page_title="Churn agent", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(T.css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Scoring cohort and running the agent...")
def load_run() -> CohortRun:
    return run_cohort(EventStore.load())


try:
    run = load_run()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

table = run.table()
results = run.by_id()

# Narrative order, not risk order. config.ARCHETYPE_IDS is the running order
# the demo is meant to be walked in, and CUST-0005 is deliberately last.
DEMO_IDS = [cid for cid in cfg.ARCHETYPE_IDS if cid in results]


def rule(label: str) -> None:
    st.markdown(f'<div class="sec"><span>{label}</span></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Navigation. The drill-down sits a long way below the cohort table, so
# selecting a customer has to take you there -- otherwise the view changes
# somewhere off-screen and nothing appears to have happened.
# --------------------------------------------------------------------------
def goto(anchor: str) -> None:
    """Queue a scroll. The counter is what makes it fire only on a change."""
    n = st.session_state.get("scroll_n", 0) + 1
    st.session_state["scroll_n"] = n
    st.session_state["scroll_to"] = (anchor, n)


def select(customer_id: str) -> None:
    st.session_state["selected"] = customer_id
    goto("detail-anchor")


def step_case(delta: int) -> None:
    """Walk the demo cases. Nothing selected yet steps to the first."""
    current = st.session_state.get("selected")
    i = DEMO_IDS.index(current) if current in DEMO_IDS else -1
    select(DEMO_IDS[min(max(i + delta, 0), len(DEMO_IDS) - 1)])


def emit_scroll() -> None:
    """Drive the scroll container.

    st.markdown strips <script>, so this has to go through a component
    iframe. The token in the body is what makes the iframe reload: an
    unchanged token means an unchanged iframe, so a filter change does not
    yank the page around.
    """
    target = st.session_state.get("scroll_to")
    if not target:
        return
    anchor, token = target
    components.html(
        f"""<script>
          /* {token} */
          const doc = window.parent.document;
          const go = () => {{
            const el = doc.getElementById("{anchor}");
            if (!el) return false;
            el.scrollIntoView({{behavior: "smooth", block: "start"}});
            return true;
          }};
          if (!go()) setTimeout(go, 400);
        </script>""",
        height=0,
    )


# ==========================================================================
# Sidebar
# ==========================================================================
with st.sidebar:
    st.markdown(
        f"<div style='font-size:1.02rem;font-weight:680;letter-spacing:-.01em'>"
        f"Churn agent</div>"
        f"<div style='font-size:.76rem;color:{T.MUTED};margin-top:.15rem'>"
        f"DTC sports nutrition &middot; {len(table)} customers<br>"
        f"cutoff {run.as_of}</div>",
        unsafe_allow_html=True,
    )

    rule("Demo cases")
    st.caption("Walk them in order — 5 is the one to finish on.")
    blurb = {
        "service_recovery": "we broke it — fix it, don't discount it",
        "false_positive_stocked": "big gap, still has product",
        "price_sensitivity": "waiting for a sale",
        "price_sensitive": "waiting for a sale",
        "cooldown_suppressed": "at risk, contacted 6 days ago",
        "legitimate_stop": "injured — correct action is nothing",
    }
    active = st.session_state.get("selected")
    for n, cid in enumerate(DEMO_IDS, 1):
        row = table[table["customer_id"] == cid].iloc[0]
        # on_click fires before the rerun, so the highlight is never a beat behind.
        st.button(f"{n}.  {row['archetype'].replace('_', ' ')}",
                  key=f"jump-{cid}", width="stretch",
                  type="primary" if cid == active else "secondary",
                  on_click=select, args=(cid,))
        st.markdown(
            f"<div style='font-size:.7rem;color:{T.MUTED};margin:-.5rem 0 .55rem .1rem'>"
            f"{row['risk']:.0%} &middot; {blurb.get(row['archetype'], '')}</div>",
            unsafe_allow_html=True,
        )

    # Slide-deck stepping: the sidebar stays on screen at any scroll depth,
    # so this is the one control that is always reachable mid-demo.
    idx = DEMO_IDS.index(active) if active in DEMO_IDS else -1
    bprev, bnext = st.columns(2)
    # No arrow glyphs in the labels: shortcut= already renders a key badge,
    # and two arrows per button reads as a stutter.
    bprev.button("Prev", width="stretch", disabled=idx <= 0,
                 on_click=step_case, args=(-1,), shortcut="Left")
    bnext.button("Next", width="stretch", disabled=idx >= len(DEMO_IDS) - 1,
                 on_click=step_case, args=(1,), shortcut="Right")

    rule("Model")
    st.markdown(
        f"<div style='font-size:.78rem;color:{T.INK_2};line-height:1.55'>"
        f"Logistic regression, {len(MODEL_FEATURES)} behavioural features. "
        f"<b>No LLM touches the risk number.</b></div>",
        unsafe_allow_html=True,
    )
    # Name, number, and one line of plain English. A judge reading this cold
    # should not have to know what a Brier score is to know which way is good.
    rows = [
        ("PR-AUC", f"{run.fit.pr_auc:.3f}",
         f"ranking quality; {run.fit.base_rate:.2f} would be random"),
        (f"Precision@{run.fit.k}", f"{run.fit.precision_at_k:.0%}",
         f"of the top {run.fit.k} we’d work, this share really churned"),
        ("ROC-AUC", f"{run.fit.roc_auc:.3f}", "0.50 would be random"),
        ("Brier", f"{run.fit.brier:.3f}", "calibration error — lower is better"),
    ]
    st.markdown(
        "<div style='margin:.5rem 0 .3rem'>"
        + "".join(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:.3rem 0 .05rem;font-size:.79rem'>"
            f"<span style='color:{T.INK_2}'>{k}</span>"
            f"<span style='font-weight:650;font-variant-numeric:tabular-nums'>"
            f"{v}</span></div>"
            f"<div class='gloss' style='border-bottom:1px solid {T.GRID};"
            f"padding-bottom:.3rem'>{g}</div>"
            for k, v, g in rows
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"{run.fit.lift_at_k:.2f}× the {run.fit.base_rate:.0%} base rate, "
        f"held-out n={run.fit.n_test}. **Synthetic data** — ground-truth "
        f"recovery, not a claim about real customers."
    )

    rule("Policy")
    st.markdown(
        f"<div style='font-size:.75rem;color:{T.MUTED};line-height:1.7'>"
        f"cooldown <b style='color:{T.INK_2}'>{cfg.COOLDOWN_DAYS}d</b> &middot; "
        f"min band <b style='color:{T.INK_2}'>{cfg.MIN_BAND_TO_ACT}</b> &middot; "
        f"supply buffer <b style='color:{T.INK_2}'>{cfg.SUPPLY_BUFFER_DAYS}d</b><br>"
        f"diagnoser <code>{run.diagnoser_name}</code></div>",
        unsafe_allow_html=True,
    )

# ==========================================================================
# View 1 — cohort
# ==========================================================================
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
st.markdown("# Spot churn, take a helpful first action")
st.markdown(
    f"<div style='color:{T.INK_2};font-size:.92rem;margin-bottom:.2rem'>"
    f"Signals &rarr; risk &rarr; policy gate &rarr; agent diagnosis &rarr; "
    f"drafted action. Every message is simulated; nothing is sent.</div>",
    unsafe_allow_html=True,
)

funnel = run.funnel()
acted = funnel[-1]["n"]
left_alone = len(table) - acted

rule("The headline")
st.markdown(
    f'<div class="hero"><div class="fig">{left_alone}</div>'
    f'<div class="txt">of {len(table)} customers were <b>deliberately left '
    f'alone</b> — {left_alone / len(table):.0%} of the cohort.<br>'
    f'{acted} got a message. Not contacting someone is the default here, '
    f'not the failure case.</div></div>',
    unsafe_allow_html=True,
)

# --- funnel + distribution ------------------------------------------------
rule("Why the list narrows")
fcol, dcol = st.columns([1.25, 1])

with fcol:
    ramp = [T.B150, T.B250, T.B350, T.B450, T.B550, T.B650]
    fdf = pd.DataFrame(funnel)
    fdf["order"] = range(len(fdf))
    fdf["colour"] = ramp[: len(fdf)]
    fdf["label"] = fdf["n"].astype(str)

    st.vega_lite_chart(
        fdf,
        {
            "height": {"step": 38},
            "layer": [
                {
                    "mark": {"type": "bar", "cornerRadiusEnd": 4, "height": 20},
                    "encoding": {
                        # No x-axis: every bar carries a direct label, so an
                        # axis would repeat information and add clutter.
                        "x": {"field": "n", "type": "quantitative", "title": None,
                              "axis": None,
                              "scale": {"domain": [0, fdf["n"].max() * 1.12]}},
                        "color": {"field": "colour", "type": "nominal", "scale": None,
                                  "legend": None},
                    },
                },
                {
                    "mark": {"type": "text", "align": "left", "dx": 6,
                             "fontSize": 11, "fontWeight": 600, "color": T.INK},
                    "encoding": {"x": {"field": "n", "type": "quantitative",
                                       "scale": {"domain": [0, fdf["n"].max() * 1.12]}},
                                 "text": {"field": "label"}},
                },
            ],
            "encoding": {
                "y": {"field": "stage", "type": "nominal", "sort": None, "title": None,
                      "axis": {"domain": False, "ticks": False, "labelFontSize": 11,
                               "labelColor": T.INK_2, "labelLimit": 220}},
                "tooltip": [
                    {"field": "stage", "title": "Stage"},
                    {"field": "n", "title": "Remaining"},
                    {"field": "note", "title": "What happened"},
                ],
            },
            "config": {"view": {"stroke": None}, "font": T.FONT,
                       "background": T.SURFACE},
        },
        width="stretch",
    )
    st.caption(
        "Each bar is what survives that rule. The cooldown and supply rules "
        "remove more people than the risk model does — which is the point."
    )

with dcol:
    ddf = pd.DataFrame({"risk": [r.risk.probability for r in run.results],
                        "band": [r.risk.band.value for r in run.results]})
    st.vega_lite_chart(
        ddf,
        {
            "height": 244,
            "mark": {"type": "bar", "cornerRadiusEnd": 3, "stroke": T.SURFACE,
                     "strokeWidth": 1},
            "encoding": {
                "x": {"field": "risk", "type": "quantitative",
                      "bin": {"maxbins": 22}, "title": "Churn risk",
                      "axis": {"format": "%", "grid": False, "domain": False,
                               "ticks": False, "labelColor": T.MUTED,
                               "labelFontSize": 10, "titleColor": T.MUTED,
                               "titleFontSize": 10, "titleFontWeight": 500}},
                "y": {"aggregate": "count", "title": "Customers",
                      "axis": {"grid": True, "gridColor": T.GRID, "domain": False,
                               "ticks": False, "labelColor": T.MUTED,
                               "labelFontSize": 10, "titleColor": T.MUTED,
                               "titleFontSize": 10, "titleFontWeight": 500}},
                "color": {
                    "field": "band", "type": "nominal",
                    "scale": {"domain": ["LOW", "MEDIUM", "HIGH"],
                              "range": [T.GOOD, T.WARNING, T.CRITICAL]},
                    "legend": {"orient": "top", "direction": "horizontal",
                               "title": None, "labelFontSize": 10,
                               "labelColor": T.INK_2, "symbolType": "square",
                               "offset": 4},
                },
                "tooltip": [{"field": "band", "title": "Band"},
                            {"aggregate": "count", "title": "Customers"}],
            },
            "config": {"view": {"stroke": None}, "font": T.FONT,
                       "background": T.SURFACE},
        },
        width="stretch",
    )
    st.caption(
        f"Bands follow Klaviyo's published cut points "
        f"(<{cfg.BAND_LOW_MAX:.0%} low, >{cfg.BAND_MEDIUM_MAX:.0%} high)."
    )

# --- cohort table ---------------------------------------------------------
rule("Cohort")

f1, f2, f3 = st.columns([1, 2, 1.1])
# Empty means all. Defaulting every option to selected filtered nothing and
# cost seven chips of visual noise, one of them truncated.
bands = f1.multiselect("Band", ["HIGH", "MEDIUM", "LOW"], default=[],
                       placeholder="All bands")
outcomes = f2.multiselect("Outcome", sorted(table["outcome"].unique()), default=[],
                          placeholder="All outcomes")
only_acted = f3.toggle("Only where we acted", value=False,
                       help="Hide everyone the gate or the agent left alone.")

view = table
if bands:
    view = view[view["band"].isin(bands)]
if outcomes:
    view = view[view["outcome"].isin(outcomes)]
if only_acted:
    view = view[view["action"] != Action.NO_ACTION.value]

view = view.rename(columns={
    "customer_id": "Customer", "risk": "Risk", "band": "Band",
    "outcome": "Outcome", "cause": "Diagnosed cause", "action": "Action",
    "days_supply_left": "Supply left", "gap_vs_median": "Gap ×",
    "churned": "Churned",
}).drop(columns=["gate", "archetype"])


def on_row_select() -> None:
    rows = st.session_state["cohort_table"].selection.rows
    if not rows:
        return
    picked = str(view.iloc[rows[0]]["Customer"])
    if picked != st.session_state.get("selected"):
        select(picked)


st.dataframe(
    view,
    key="cohort_table",
    width="stretch",
    height=430,
    hide_index=True,
    on_select=on_row_select,
    selection_mode="single-row",
    column_config={
        "Customer": st.column_config.TextColumn(width=100),
        "Risk": st.column_config.ProgressColumn(
            min_value=0.0, max_value=1.0, format="percent", width=115),
        "Band": st.column_config.TextColumn(width=78),
        "Outcome": st.column_config.TextColumn(width=168),
        "Diagnosed cause": st.column_config.TextColumn(width=145),
        "Action": st.column_config.TextColumn(width=178),
        "Supply left": st.column_config.NumberColumn(
            format="%d d", width=95,
            help="Days of product still on hand. Negative means they ran out."),
        "Gap ×": st.column_config.NumberColumn(
            format="%.1f×", width=78,
            help="Current gap over their own median reorder interval."),
        "Churned": st.column_config.CheckboxColumn(
            width=86, help="Held-out label. The model never saw it."),
    },
)
st.caption(f"{len(view)} of {len(table)} customers. Click a row for the full trace.")

# ==========================================================================
# View 2 — drill-down
# ==========================================================================
selected = st.session_state.get("selected")
if not selected:
    st.info(
        "Pick a demo case in the sidebar — or click any row above — to see the "
        "full trace for one customer.",
        icon=":material/arrow_upward:",
    )
    st.stop()

r = results[selected]
fill, tint, ink = T.BAND[r.risk.band.value]

st.markdown('<div id="detail-anchor"></div>', unsafe_allow_html=True)
rule("Customer detail")

nav_back, nav_pos = st.columns([1, 3], vertical_alignment="center")
nav_back.button("↑ Back to cohort", width="stretch",
                on_click=goto, args=("top-anchor",))
if selected in DEMO_IDS:
    nav_pos.markdown(
        f"<div style='font-size:.75rem;color:{T.MUTED}'>Demo case "
        f"<b style='color:{T.INK_2}'>{DEMO_IDS.index(selected) + 1} of "
        f"{len(DEMO_IDS)}</b></div>",
        unsafe_allow_html=True,
    )

hl, hr = st.columns([2.4, 1])
with hl:
    st.markdown(
        f"<div style='font-size:1.5rem;font-weight:670;letter-spacing:-.01em'>"
        f"{selected}</div>"
        f"<div style='margin-top:.35rem'>{T.band_pill(r.risk.band.value)} "
        f"{T.outcome_pill(r.outcome_label)}"
        + (f" {T.pill(r.archetype.replace('_', ' '), 'rgba(74,58,167,0.10)', '#3b2e86')}"
           if r.archetype else "")
        + "</div>",
        unsafe_allow_html=True,
    )
with hr:
    st.markdown(
        f"<div style='text-align:right'>"
        f"<span style='font-size:2.6rem;font-weight:700;color:{fill};line-height:1'>"
        f"{r.risk.probability:.0%}</span>"
        f"<div style='font-size:.72rem;color:{T.MUTED};margin-top:.1rem'>"
        f"modelled churn risk</div></div>"
        + T.meter(r.risk.probability, r.risk.band.value),
        unsafe_allow_html=True,
    )

t1, t2, t3, t4 = st.tabs([
    "Signals & risk", "Policy gate", "Agent", "Guardrails",
])

# ------------------------------------------------------------- signals
with t1:
    st.caption(
        "Deterministic layer — pure Python and sklearn. No LLM is involved in "
        "producing this number or these attributions."
    )

    supply = r.features["days_of_supply_remaining"]
    st.markdown(
        f'<div class="tiles">'
        + T.tile("Days of supply left", f"{supply:+.0f}",
                 "servings ÷ per-day, minus elapsed")
        + T.tile("Gap vs. their median", f"{r.features['reorder_gap_ratio']:.2f}×",
                 f"{r.features['days_since_last_order']:.0f}d since, median "
                 f"{r.features['personal_median_interpurchase']:.0f}d")
        + T.tile("Email click rate", f"{r.features['click_rate_90d']:.0%}",
                 "90d — clicks; opens are inflated by Apple Mail")
        + T.tile("Open tickets", f"{r.features['unresolved_tickets']:.0f}",
                 "unstructured signal — read by the agent")
        + "</div>",
        unsafe_allow_html=True,
    )

    if supply > cfg.SUPPLY_BUFFER_DAYS:
        st.success(
            f"**Still has product.** A gap this long is pack size, not "
            f"disengagement — the case a naive recency score gets wrong.",
            icon=":material/inventory_2:",
        )

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("##### What drove the score")
        adf = pd.DataFrame([
            {"Feature": a.label, "c": round(a.contribution, 4),
             "Value": round(a.value, 2),
             "dir": "increases risk" if a.contribution > 0 else "decreases risk"}
            for a in r.risk.top(8)
        ])
        st.vega_lite_chart(
            adf,
            {
                "height": {"step": 27},
                "mark": {"type": "bar", "cornerRadius": 3, "height": 15},
                "encoding": {
                    "x": {"field": "c", "type": "quantitative",
                          "title": "contribution to log-odds",
                          "axis": {"grid": True, "gridColor": T.GRID,
                                   "domain": False, "ticks": False,
                                   "labelColor": T.MUTED, "labelFontSize": 10,
                                   "titleColor": T.MUTED, "titleFontSize": 10,
                                   "titleFontWeight": 500}},
                    "y": {"field": "Feature", "type": "nominal", "title": None,
                          "sort": {"field": "c", "op": "min", "order": "descending"},
                          "axis": {"domain": False, "ticks": False,
                                   "labelFontSize": 11, "labelColor": T.INK_2,
                                   "labelLimit": 230}},
                    # Diverging: two hues around a neutral zero, never a ramp.
                    "color": {"field": "dir", "type": "nominal",
                              "scale": {"domain": ["increases risk", "decreases risk"],
                                        "range": [T.CRITICAL, T.BLUE]},
                              "legend": {"orient": "top", "title": None,
                                         "labelFontSize": 10, "labelColor": T.INK_2,
                                         "symbolType": "square", "offset": 4}},
                    "tooltip": [{"field": "Feature"}, {"field": "Value"},
                                {"field": "c", "title": "Contribution",
                                 "format": "+.3f"}],
                },
                "config": {"view": {"stroke": None}, "font": T.FONT,
                           "background": T.SURFACE},
            },
            width="stretch",
        )
        st.caption(
            "Exact, not approximated: for a linear model each contribution is "
            "`coefficient × standardised value`, and they sum to the log-odds. "
            "No SHAP needed."
        )

    with right:
        st.markdown("##### All features")
        st.dataframe(
            pd.DataFrame([
                {"Feature": FEATURE_LABELS.get(k, k), "Value": round(v, 2),
                 "In model": k in MODEL_FEATURES}
                for k, v in r.features.items()
            ]),
            hide_index=True, width="stretch", height=430,
            column_config={
                "Feature": st.column_config.TextColumn(width=330),
                "Value": st.column_config.NumberColumn(width=78),
                "In model": st.column_config.CheckboxColumn(width=82),
            },
        )
        st.caption(
            f"{len(MODEL_FEATURES)} of {len(FEATURE_NAMES)} are fed to the model. The rest are "
            f"computed and shown but withheld — see `features/extract.py` for "
            f"why each one is out."
        )

# ---------------------------------------------------------------- gate
with t2:
    st.caption(
        "Deterministic, and it runs **before** the agent. A customer who fails "
        "here never reaches the LLM — no tokens are spent reasoning about "
        "someone we are not allowed to contact."
    )
    for c in r.policy.checks:
        colour, mark = ((T.GOOD, "PASS") if c.passed else (T.CRITICAL, "STOP"))
        bg = "rgba(12,163,12,0.05)" if c.passed else "rgba(208,59,59,0.06)"
        st.markdown(
            f'<div class="chk" style="background:{bg}">'
            f'<div class="m" style="color:{colour}">{mark}</div>'
            f'<div class="b"><b>{c.name}</b> — {c.detail}</div></div>',
            unsafe_allow_html=True,
        )
    if r.policy.eligible:
        st.success(r.policy.reason, icon=":material/arrow_forward:")
    else:
        st.warning(
            f"**{r.policy.outcome.value}** — {r.policy.reason}  \n"
            f"The agent was not run for this customer.",
            icon=":material/block:",
        )

# --------------------------------------------------------------- agent
with t3:
    if r.diagnosis is None:
        st.warning("Agent not run — the policy gate stopped this customer.",
                   icon=":material/block:")
        st.caption(r.policy.reason)
    else:
        d = r.diagnosis
        st.markdown(
            f'<div class="tiles" style="grid-template-columns:repeat(3,1fr)">'
            + T.tile("Diagnosed cause", d.cause.value.replace("_", " "),
                     f"{d.cause_confidence:.0%} confidence", word=True)
            + T.tile("Proposed action", d.action.value.replace("_", " "),
                     f"via {d.diagnoser}", word=True)
            + T.tile("Message", "Drafted" if d.has_message else "None",
                     "awaiting human approval" if d.has_message
                     else "doing nothing is the answer", word=True)
            + "</div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown("##### Reasoning trace")
            st.markdown(
                "".join(
                    f'<div class="step"><div class="n">{i}</div>'
                    f'<div class="t">{step}</div></div>'
                    for i, step in enumerate(d.reasoning_trace, 1)
                ),
                unsafe_allow_html=True,
            )
            st.markdown("##### Why this action")
            st.info(d.action_rationale)

        with c2:
            st.markdown("##### Drafted message")
            if not d.has_message:
                st.success(
                    "No message. Doing nothing is a first-class outcome here, "
                    "not a failure to produce output.",
                    icon=":material/do_not_disturb_on:",
                )
            else:
                st.markdown(
                    f'<div class="sim"><div class="bar">'
                    f'{cfg.SIMULATION_BANNER} &nbsp;·&nbsp; no send path exists '
                    f'in this project</div>'
                    f'<div class="subj">{d.message_subject}</div>'
                    f'<div class="body">{d.message_body}</div></div>',
                    unsafe_allow_html=True,
                )
                st.button("Approve and send", disabled=True, width="stretch",
                          help="Intentionally inert. There is no send path.")

# ---------------------------------------------------------- guardrails
with t4:
    if r.guardrails is None:
        st.warning("No guardrail review — the agent did not run.",
                   icon=":material/block:")
    else:
        g = r.guardrails
        if g.passed:
            st.success(g.verdict, icon=":material/verified:")
        else:
            st.error(g.verdict, icon=":material/gpp_maybe:")
        if g.downgraded:
            st.warning(
                f"Action downgraded from **{g.original_action.value}** to "
                f"**{g.final_action.value}**.", icon=":material/trending_down:",
            )
        st.caption(
            "These run on the agent's *output*, not its input. They are the "
            "reason a non-deterministic layer is safe to put in the middle."
        )
        for c in g.checks:
            colour, mark = ((T.GOOD, "PASS") if c.passed else (T.CRITICAL, "FAIL"))
            bg = "rgba(12,163,12,0.05)" if c.passed else "rgba(208,59,59,0.06)"
            st.markdown(
                f'<div class="chk" style="background:{bg}">'
                f'<div class="m" style="color:{colour}">{mark}</div>'
                f'<div class="b"><b>{c.name}</b> — {c.detail}</div></div>',
                unsafe_allow_html=True,
            )


# Last thing on the page: act on whatever navigation queued up this run.
emit_scroll()
