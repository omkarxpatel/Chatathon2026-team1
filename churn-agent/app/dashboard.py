"""An action-first Streamlit workspace. All responses remain local demo drafts.

Run from churn-agent with: streamlit run app/dashboard.py
"""

from __future__ import annotations

import math
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import config as cfg
from agent.schemas import Action
from app import theme as T
from app.workflow import CHECK_LABELS, can_review, explanation, next_step, reason, validate_draft
from data.store import EventStore
from features.extract import FEATURE_LABELS, MODEL_FEATURES
from pipeline import CohortRun, CustomerResult, run_cohort

st.set_page_config(page_title="Cadence · Retention workspace", page_icon="🌱",
                   layout="wide", initial_sidebar_state="auto")
st.markdown(T.css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Finding customers who may need attention…")
def load_run() -> CohortRun:
    return run_cohort(EventStore.load())


try:
    run = load_run()
except FileNotFoundError:
    st.title("Let's set up your workspace")
    st.info("The demo customer data has not been generated yet.")
    st.write("Run this command from the churn-agent folder, then reload this page.")
    st.code("python run_generate.py", language="bash")
    st.stop()

results = run.by_id()
st.session_state.setdefault("page", "Review queue")
st.session_state.setdefault("selected", None)
st.session_state.setdefault("drafts", {})
st.session_state.setdefault("queue_page", 0)
# Keep queue filters when their widgets are temporarily absent on a detail page.
for key, default in [("queue_scope", "Needs review"), ("customer_search", ""),
                     ("risk_filter", "All risk levels")]:
    st.session_state[key] = st.session_state.get(key, default)


def html(content: str) -> None:
    st.markdown(content, unsafe_allow_html=True)


def navigate(page: str) -> None:
    st.session_state.page = page
    st.session_state.selected = None
    st.session_state.queue_page = 0
    st.session_state["navigation_epoch"] = st.session_state.get("navigation_epoch", 0) + 1


def select(customer_id: str | None) -> None:
    st.session_state.selected = customer_id
    st.session_state["navigation_epoch"] = st.session_state.get("navigation_epoch", 0) + 1


def reset_paging() -> None:
    st.session_state.queue_page = 0


def turn_page(delta: int) -> None:
    st.session_state.queue_page += delta


def reviewed(r: CustomerResult) -> bool:
    return st.session_state.drafts.get(r.customer_id, {}).get("reviewed", False)


def review_status(r: CustomerResult) -> str:
    if not can_review(r):
        return "On hold" if r.risk.band.value != "LOW" else "Monitoring"
    if reviewed(r):
        return "Reviewed · not sent"
    saved = st.session_state.drafts.get(r.customer_id)
    if saved:
        return "Needs edits" if validate_draft(r, saved["subject"], saved["body"]) else "Draft saved"
    if r.guardrails and not r.guardrails.passed:
        return "Needs edits"
    return "Needs review"


pending = [r for r in run.results if can_review(r) and not reviewed(r)]
high_risk = [r for r in run.results if r.risk.band.value == "HIGH"]
on_hold = [r for r in run.results if not can_review(r) and r.risk.band.value != "LOW"]
# Not high risk yet, but moving the way customers who left moved. Nothing
# else in this workspace surfaces them.
early_warning = sorted([r for r in run.results if r.early_warning],
                       key=lambda r: -r.trajectory.relative_momentum)

with st.sidebar:
    html('<div class="brand"><span class="brand-mark" aria-hidden="true">↗</span>Cadence</div>'
         '<div class="brand-sub">Better timing. Better retention.</div>'
         '<div class="eyebrow">Workspace</div>')
    for page, icon in [("Review queue", ":material/inbox:"),
                       ("All customers", ":material/group:"),
                       ("How it works", ":material/help_outline:")]:
        st.button(page, icon=icon, key=f"nav-{page}", width="stretch",
                  type="primary" if st.session_state.page == page else "secondary",
                  on_click=navigate, args=(page,))
    html('<div class="sidebar-note"><strong>A helpful next step, for the right person.</strong><br>'
         'Start with a customer, review the context, then refine the suggested response.</div>')
    html('<div class="sidebar-note">' + T.pill("Demo workspace") +
         '<p style="margin:.65rem 0 .2rem">Synthetic customers. Nothing is sent.</p>'
         '<span>Drafts and review progress last for this browser session.</span></div>')

html('<div id="workspace-top" class="topline"><strong>WORKSPACE &nbsp;/&nbsp; DTC sports nutrition</strong>'
     f'<span>{T.pill("Demo data")} &nbsp; Snapshot · {run.as_of.strftime("%b %d, %Y")}</span></div>')


def render_stats() -> None:
    ready_high = sum(r.risk.band.value == "HIGH" for r in pending)
    html('<div class="stats">'
         + T.stat("Needs your review", str(len(pending)), "Suggested responses to work through", True)
         + T.stat("High-risk customers", str(len(high_risk)), f"{ready_high} still need a response reviewed")
         + T.stat("Outreach on hold", str(len(on_hold)), "At-risk customers best left alone for now")
         + T.stat("Drifting quietly", str(len(early_warning)), "Not high risk yet, but trending that way")
         + '</div>')


def render_queue() -> None:
    all_customers = st.session_state.page == "All customers"
    st.title("All customers" if all_customers else "Your retention workspace")
    html('<div class="lead">' + ("A clear view of every customer, their risk, and the recommended next step."
         if all_customers else "See who may leave, understand why, and review the right response.") + '</div>')
    render_stats()
    html('<div class="section-heading"><h2>' + ("Customer directory" if all_customers else "Your review queue")
         + '</h2><span>Highest risk first</span></div>')
    if all_customers:
        scope = "All customers"
    else:
        scope = st.radio("Queue view", ["Needs review", "High risk", "Drifting", "On hold", "Reviewed"],
                         horizontal=True, label_visibility="collapsed", key="queue_scope", on_change=reset_paging)
    with st.container(key="queue-filters"):
        search_col, filter_col = st.columns([3, 1.4])
    search = search_col.text_input("Search customers", placeholder="Search by customer ID, reason, or next step…",
                                    label_visibility="collapsed", key="customer_search", on_change=reset_paging)
    band = filter_col.selectbox("Filter by risk", ["All risk levels", "High", "Medium", "Low"],
                               label_visibility="collapsed", key="risk_filter", on_change=reset_paging)
    scopes = {"Needs review": pending, "High risk": high_risk, "Drifting": early_warning,
              "On hold": on_hold,
              "Reviewed": [r for r in run.results if reviewed(r)], "All customers": run.results}
    view = scopes[scope]
    if band != "All risk levels":
        view = [r for r in view if r.risk.band.value == band.upper()]
    if search.strip():
        query = search.strip().casefold()
        view = [r for r in view if query in f"{r.customer_id} {reason(r)} {next_step(r)}".casefold()]
    descriptions = {
        "Needs review": "Review a draft, make it your own, and mark it reviewed. Nothing is sent.",
        "High risk": "High risk does not always mean contact them. Check the recommended next step.",
        "Drifting": "Their risk is still below high, but it is climbing the way it climbed for customers who left. Worth a look before it becomes urgent.",
        "On hold": "Outreach is paused for a reason. Open a customer to understand why.",
        "Reviewed": "Responses you have reviewed in this session. These have not been sent.",
        "All customers": "Open any customer to see the evidence and contact guidance.",
    }
    st.caption(f"{len(view)} customer{'s' if len(view) != 1 else ''} · {descriptions[scope]}")
    if not view:
        title, message = "No customers match these filters", "Try a different search or risk level."
        if not search.strip() and band == "All risk levels":
            title, message = {
                "Needs review": ("You're all caught up", "Every suggested response has been reviewed. Explore high-risk customers for more context."),
                "Reviewed": ("Your reviewed responses will appear here", "Open a customer in Needs review and mark their response reviewed."),
                "Drifting": ("Nobody is quietly drifting right now", "Every customer whose risk is climbing is already in the high-risk view."),
            }.get(scope, ("Nothing here right now", "Choose another view to explore your customers."))
        html(f'<div class="empty"><h3>{escape(title)}</h3><p>{escape(message)}</p></div>')
        return
    page_size = 6
    pages = math.ceil(len(view) / page_size)
    st.session_state.queue_page = max(0, min(st.session_state.queue_page, pages - 1))
    start = st.session_state.queue_page * page_size
    for r in view[start:start + page_size]:
        with st.container(border=True, key=f"customer-row-{r.customer_id}"):
            customer, risk, action, open_col = st.columns([2.2, 1, 2, 1], vertical_alignment="center")
            with customer:
                html(f'<div class="customer-name">{escape(r.customer_id)}</div>'
                     f'<div class="customer-reason">{escape(reason(r))}</div>')
            with risk:
                html('<div class="row-label">Churn risk</div>'
                     + T.band_pill(r.risk.band.value, r.risk.probability)
                     + ('<div style="margin-top:.35rem">'
                        + T.trend_pill(r.trajectory.state.value) + '</div>'
                        if r.drifting else ''))
            with action:
                html(f'<div class="action-name">{escape(next_step(r))}</div>'
                     f'<div class="row-status">{escape(review_status(r))}</div>')
            open_col.button("Open review" if can_review(r) else "View reason", key=f"open-{r.customer_id}",
                            width="stretch", on_click=select, args=(r.customer_id,))
    count, prev_col, next_col = st.columns([4, 1, 1], vertical_alignment="center")
    count.caption(f"Showing {start + 1}–{min(start + page_size, len(view))} of {len(view)}")
    prev_col.button("Previous", width="stretch", disabled=st.session_state.queue_page == 0,
                    on_click=turn_page, args=(-1,))
    next_col.button("Next", width="stretch", disabled=st.session_state.queue_page >= pages - 1,
                    on_click=turn_page, args=(1,))


def render_evidence(r: CustomerResult) -> None:
    f = r.features
    supply = f["days_of_supply_remaining"]
    supply_value = f"{supply:.0f} days left" if supply >= 0 else f"{abs(supply):.0f} days ago"
    html('<div class="facts">'
         f'<div class="fact"><div class="fact-label">Last delivery</div><div class="fact-value">{f["days_since_last_order"]:.0f} days ago</div></div>'
         f'<div class="fact"><div class="fact-label">{"Estimated supply" if supply >= 0 else "Estimated runout"}</div><div class="fact-value">{supply_value}</div></div>'
         f'<div class="fact"><div class="fact-label">Open support tickets</div><div class="fact-value">{f["unresolved_tickets"]:.0f}</div></div></div>')
    render_trajectory(r)
    with st.expander("Why this risk score?"):
        st.write(f"This customer's estimated likelihood of not reordering is **{r.risk.probability:.0%}**. "
                 "It is a signal to investigate, not a certainty or a reason to send a message on its own.")
        st.caption("The model compares their purchase and engagement patterns. Strongest influences:")
        for a in r.risk.top(3):
            direction = "Raises risk" if a.contribution > 0 else "Lowers risk"
            st.write(f"**{direction}** · {a.label}")
        st.caption(f"High: {cfg.BAND_MEDIUM_MAX:.0%} or above · Medium: {cfg.BAND_LOW_MAX:.0%} to below "
                   f"{cfg.BAND_MEDIUM_MAX:.0%} · Low: below {cfg.BAND_LOW_MAX:.0%}. "
                   "Scores come from a statistical model trained on synthetic data.")
    with st.expander("Contact checks"):
        html("".join(T.check(CHECK_LABELS.get(c.name, c.name), c.detail, c.passed) for c in r.policy.checks))
        st.caption("All four must pass before a response is suggested.")
    if r.diagnosis:
        with st.expander("How the recommendation was reached"):
            st.caption(f"Suggested reason: {reason(r)} · {r.diagnosis.cause_confidence:.0%} reported confidence.")
            for i, step in enumerate(r.diagnosis.reasoning_trace, 1):
                st.write(f"{i}. {step}")
            st.caption("The default demo uses rules to interpret the evidence and draft a response.")
    with st.expander("Customer history & signals"):
        ev = EventStore.load().events_for(r.customer_id, run.as_of)
        st.markdown("**Recent support conversations**")
        if ev.tickets.empty:
            st.caption("No support conversations on record.")
        else:
            for _, ticket in ev.tickets.tail(3).iloc[::-1].iterrows():
                st.caption(f"{ticket['created_ts']:%b %d, %Y} · {str(ticket['status']).title()}")
                st.text(str(ticket["text"]))
        st.markdown("**Recorded signals**")
        st.dataframe(pd.DataFrame([
            {"Signal": FEATURE_LABELS.get(k, k), "Value": round(v, 2), "Used in risk score": k in MODEL_FEATURES}
            for k, v in f.items()
        ]), hide_index=True, width="stretch")


def render_trajectory(r: CustomerResult) -> None:
    """The risk curve. Direction of travel, which the score alone cannot show."""
    t = r.trajectory
    with st.container(border=True, key=f"trend-{r.customer_id}"):
        html('<div class="row-label">Risk over the last '
             f'{t.offsets[0]} days</div>'
             + T.trend_pill(t.state.value, f"{t.relative_momentum:+.0f} pts/100d vs cohort")
             + T.sparkline(t.curve, t.offsets, t.state.value,
                           cfg.BAND_LOW_MAX, cfg.BAND_MEDIUM_MAX))
        if r.drifting:
            st.caption("This is the shape customers showed before they stopped ordering. "
                       "It is a reason to look closer, not a reason to send anything.")
        worsening = t.worsening_signals[:3]
        if worsening:
            st.markdown("**What changed**")
            for d in worsening:
                st.caption(f"{d.label}: {d.earliest:,.2f} → {d.latest:,.2f}")
        st.caption("The same model, re-run on what we knew at each earlier date. "
                   "The oldest points are the least certain: the customer had less "
                   "history behind them then.")


def render_draft(r: CustomerResult) -> None:
    cid = r.customer_id
    if not can_review(r):
        with st.container(border=True):
            html('<div class="panel-label">Recommended response</div>')
            st.subheader("No outreach for now")
            st.write(explanation(r))
            st.caption("A high risk score can still lead to a decision to wait. Respecting the customer's situation is part of retention.")
        return
    d = r.diagnosis
    saved = st.session_state.drafts.get(cid, {})
    subject = saved.get("subject", d.message_subject or "")
    body = saved.get("body", d.message_body or "")
    with st.form(f"draft-form-{cid}"):
        html('<div class="panel-label">' + ("Internal handoff" if r.final_action == Action.HUMAN_ESCALATION else "Suggested response") + '</div>')
        st.subheader("Make it personal")
        st.caption("Edit the draft, then mark it reviewed. This demo does not send messages.")
        subject_input = st.text_input("Subject", value=subject, key=f"subject-{cid}", disabled=reviewed(r))
        body_input = st.text_area("Message" if r.final_action != Action.HUMAN_ESCALATION else "Internal note",
                                  value=body, height=320, key=f"body-{cid}", disabled=reviewed(r))
        st.caption(f"Keep it helpful: {cfg.MAX_MESSAGE_WORDS} words or fewer. Edits are checked when you save or mark reviewed.")
        save_col, review_col = st.columns([1, 1.3])
        save_clicked = save_col.form_submit_button("Save draft", width="stretch", disabled=reviewed(r))
        review_clicked = review_col.form_submit_button("Mark reviewed", type="primary", width="stretch", disabled=reviewed(r))
    if save_clicked or review_clicked:
        errors = validate_draft(r, subject_input, body_input)
        st.session_state.drafts[cid] = {
            "subject": subject_input.strip(), "body": body_input.strip(),
            "reviewed": bool(review_clicked and not errors),
        }
        st.session_state["draft_notice"] = (cid, "reviewed" if review_clicked and not errors else "saved")
        st.rerun()
    errors = validate_draft(r, subject, body)
    notice = st.session_state.pop("draft_notice", None)
    if reviewed(r):
        st.success("Marked reviewed. Nothing has been sent.", icon=":material/check_circle:")
        export_text = (f"{cfg.SIMULATION_BANNER}\nCustomer: {cid}\n"
                       f"Recommended action: {next_step(r)}\n\nSubject: {subject}\n\n{body}\n")
        export_col, reopen_col = st.columns(2)
        export_col.download_button("Download reviewed draft", data=export_text,
                                    file_name=f"{cid}-reviewed-draft.txt", mime="text/plain", width="stretch")
        if reopen_col.button("Reopen review", width="stretch"):
            st.session_state.drafts[cid]["reviewed"] = False
            st.rerun()
    elif errors:
        st.warning("Draft saved, but it needs edits before review." if notice else "This draft needs a small edit before review.")
        for error in errors:
            st.caption(error)
    elif notice:
        st.success("Draft saved for this session. Ready for your review.")
    else:
        st.caption("✓ Draft checks passed · Awaiting your review")
    if reviewed(r):
        remaining = [item for item in pending if item.customer_id != cid]
        if remaining:
            st.button("Review next customer", icon=":material/arrow_forward:", width="stretch",
                      on_click=select, args=(remaining[0].customer_id,))
    st.caption("Progress is saved only in this browser session. Download a reviewed draft to keep a copy.")


def render_customer(r: CustomerResult) -> None:
    st.button(f"Back to {st.session_state.page.lower()}", icon=":material/arrow_back:",
              on_click=select, args=(None,))
    st.title(r.customer_id)
    html(T.band_pill(r.risk.band.value, r.risk.probability) + ' &nbsp; '
         + T.trend_pill(r.trajectory.state.value) + ' &nbsp; '
         + T.pill(review_status(r), "#e9ece7", "#52604f"))
    with st.container(key="customer-detail"):
        left, right = st.columns([1, 1.2], gap="large")
    with left:
        html('<div class="panel-label" style="margin-top:1.1rem">The recommended next step</div>'
             f'<div class="recommendation"><h3>{escape(next_step(r))}</h3>'
             f'<p>{escape(explanation(r))}</p></div>')
        st.subheader("A little context")
        render_evidence(r)
    with right:
        render_draft(r)


def render_how() -> None:
    st.title("Thoughtful outreach starts with context.")
    html('<div class="lead">Three steps from an early warning to a helpful response. You make the final call.</div>')
    with st.container(border=True):
        for n, title, text in [
            (1, "Spot a change", "Purchase timing, browsing, and email engagement help estimate who may not reorder. Customers with higher risk appear first."),
            (2, "Notice the direction",
             "A score is a snapshot. We re-run it on what we knew at a dozen earlier dates to see whether someone is drifting or steady. Customers whose risk is climbing show up under Drifting before they reach high risk."),
            (3, "Check the context", f"Before suggesting outreach, we check consent, a {cfg.COOLDOWN_DAYS}-day break between targeted messages, risk, and remaining product. We also consider support conversations and personal circumstances."),
            (4, "Choose a helpful response", "Read the recommendation, edit the draft, and mark it reviewed. If waiting is the better choice, the customer stays on hold with a clear explanation."),
        ]:
            html(f'<div class="how-step"><div class="how-number">{n}</div>'
                 f'<div><h3>{title}</h3><p>{escape(text)}</p></div></div>')
    st.info("This is a demo with synthetic customers. Responses are drafts, nothing is sent, and your review progress lasts for this browser session.", icon=":material/info:")
    st.subheader("Explore a few examples")
    demos = {"CUST-0001": "A service issue to resolve", "CUST-0002": "A customer who still has product",
             "CUST-0003": "A customer looking for better value", "CUST-0004": "A recent message means wait",
             "CUST-0005": "A life change means give them space"}
    for cid, label in demos.items():
        if cid in results:
            st.button(f"{label}  ·  {cid}", key=f"demo-{cid}", on_click=select, args=(cid,))
    with st.expander("Model details & evaluation"):
        st.write(f"Risk is calculated by logistic regression using {len(MODEL_FEATURES)} behavioral signals. "
                 "The response generator receives the score; it does not calculate or change it.")
        st.caption(f"Response generator: {run.diagnoser_name}. The default is a deterministic, rule-based demo.")
        st.dataframe(pd.DataFrame([
            {"Measure": f"Precision among the top {run.fit.k}", "Result": f"{run.fit.precision_at_k:.0%}", "Meaning": "Share of the highest-ranked test customers who churned"},
            {"Measure": "PR-AUC", "Result": f"{run.fit.pr_auc:.3f}", "Meaning": "Ranking quality across precision and recall"},
            {"Measure": "ROC-AUC", "Result": f"{run.fit.roc_auc:.3f}", "Meaning": "Ability to rank churners above non-churners"},
            {"Measure": "Brier score", "Result": f"{run.fit.brier:.3f}", "Meaning": "Prediction error; lower is better"},
        ]), hide_index=True, width="stretch")
        st.caption(f"Evaluated on {run.fit.n_test} held-out synthetic customers in a single split. "
                   "These results do not establish real-world accuracy or prove that outreach prevents churn.")
        st.markdown("**How the outreach list narrows**")
        funnel = pd.DataFrame(run.funnel())[["stage", "n"]]
        funnel.loc[funnel.index[-1], "stage"] = "Response suggested"
        st.dataframe(funnel.rename(columns={"stage": "Step", "n": "Customers remaining"}),
                     hide_index=True, width="stretch")


if st.session_state.selected in results:
    render_customer(results[st.session_state.selected])
elif st.session_state.page == "How it works":
    render_how()
else:
    render_queue()

# A new view must start at its heading, even when opened from the bottom of
# the queue. The token changes only on navigation, not while editing a draft.
epoch = st.session_state.get("navigation_epoch", 0)
if epoch:
    # st.iframe replaces components.html in recent Streamlit releases.
    embed_navigation = getattr(st, "iframe", components.html)
    embed_navigation(f"""<script>
      // Navigation {int(epoch)}
      const anchor = window.parent.document.getElementById('workspace-top');
      if (anchor) anchor.scrollIntoView({{block: 'start'}});
    </script>""", height=1, tab_index=-1)
