"""The retention workspace: what the agent found, and what you want to do about it.

Written for the person who does retention, not for the person who built
this. The screen leads with the customer and the recommended next step;
the investigation behind it is one scroll down; the statistics are behind
a disclosure and stay there.

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
from agent.loop import MAX_STEPS
from agent.schemas import Action, StepKind
from agent.tools import DECIDE_TOOLS, INVESTIGATE_TOOLS
from app import theme as T
from app.workflow import (
    CHECK_LABELS,
    can_review,
    confidence_label,
    explanation,
    hold_reason,
    next_step,
    reason,
    run_summary,
    step_label,
    validate_draft,
    what_changes,
)
from data.store import EventStore
from features.extract import FEATURE_LABELS, FEATURE_NAMES, MODEL_FEATURES
from pipeline import CohortRun, CustomerResult, run_cohort

st.set_page_config(page_title="Retention agent · Workspace", page_icon="🌱",
                   layout="wide", initial_sidebar_state="auto")
st.markdown(T.css(), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_store() -> EventStore:
    return EventStore.load()


@st.cache_resource(show_spinner="The agent is going through your customers…")
def load_run() -> CohortRun:
    return run_cohort(load_store())


try:
    run = load_run()
except FileNotFoundError:
    st.title("Let's set up your workspace")
    st.info("The demo customer data has not been generated yet.")
    st.write("Run this command from the churn-agent folder, then reload this page.")
    st.code("python run_generate.py", language="bash")
    st.stop()

results = run.by_id()
stats = run.agent_stats()
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

AGENT_KIND = ("Rule-based agent" if run.agent_name == "rules"
              else f"Claude ({run.agent_name})")

with st.sidebar:
    html('<div class="brand"><span class="brand-mark" aria-hidden="true">↗</span>Retention agent</div>'
         '<div class="brand-sub">Finds who is leaving. Works out why.</div>'
         '<div class="eyebrow">Workspace</div>')
    for page, icon in [("Review queue", ":material/inbox:"),
                       ("All customers", ":material/group:"),
                       ("Flow map", ":material/account_tree:"),
                       ("How it works", ":material/help_outline:")]:
        st.button(page, icon=icon, key=f"nav-{page}", width="stretch",
                  type="primary" if st.session_state.page == page else "secondary",
                  on_click=navigate, args=(page,))
    html('<div class="sidebar-note"><strong>Last run</strong><br>'
         f'{run.as_of.strftime("%b %d, %Y")} · {AGENT_KIND}<br>'
         f'{stats["customers_investigated"]} customers investigated · '
         f'{stats["tool_calls"]} tool calls · {stats["drafted"]} drafts'
         '</div>')
    html('<div class="sidebar-note">' + T.pill("Demo workspace") +
         '<p style="margin:.65rem 0 .2rem">Synthetic customers. Nothing is sent.</p>'
         '<span>Drafts and review progress last for this browser session.</span></div>')

html('<div id="workspace-top" class="topline"><strong>WORKSPACE &nbsp;/&nbsp; DTC sports nutrition</strong>'
     f'<span>{T.pill("Demo data")} &nbsp; Snapshot · {run.as_of.strftime("%b %d, %Y")}</span></div>')


def render_run_strip() -> None:
    """The agent's last run as four numbers, in the order they happened."""
    html(T.flow([
        (str(len(run.results)), "customers scanned",
         "orders, visits, emails, support tickets"),
        (str(stats["customers_investigated"]), "worth a closer look",
         "the rest were ruled out before the agent ran"),
        (str(stats["tool_calls"]), "things the agent checked",
         f"about {stats['avg_steps']:g} per customer, its own choice"),
        (str(stats["drafted"]), "responses drafted",
         f"{stats['left_alone']} customers it chose to leave alone"),
    ], highlight=1))


def render_stats() -> None:
    ready_high = sum(r.risk.band.value == "HIGH" for r in pending)
    html('<div class="stats">'
         + T.stat("Needs your review", str(len(pending)), "Suggested responses to work through", True)
         + T.stat("High-risk customers", str(len(high_risk)), f"{ready_high} still need a response reviewed")
         + T.stat("Outreach on hold", str(len(on_hold)), "At-risk customers best left alone for now")
         + '</div>')


def render_queue() -> None:
    all_customers = st.session_state.page == "All customers"
    st.title("All customers" if all_customers else "Your retention workspace")
    html('<div class="lead">' + (
        "Every customer, their risk, and what the agent recommends doing about it."
        if all_customers else
        f"The agent went through {len(run.results)} customers, looked closely at "
        f"{stats['customers_investigated']}, and left you {len(pending)} to review. "
        f"Nothing goes out without you.") + '</div>')
    render_run_strip()
    render_stats()
    html('<div class="section-heading"><h2>' + ("Customer directory" if all_customers else "Your review queue")
         + '</h2><span>Highest risk first</span></div>')
    if all_customers:
        scope = "All customers"
    else:
        scope = st.radio("Queue view", ["Needs review", "High risk", "On hold", "Reviewed"],
                         horizontal=True, label_visibility="collapsed", key="queue_scope", on_change=reset_paging)
    with st.container(key="queue-filters"):
        search_col, filter_col = st.columns([3, 1.4])
    search = search_col.text_input("Search customers", placeholder="Search by customer ID, reason, or next step…",
                                    label_visibility="collapsed", key="customer_search", on_change=reset_paging)
    band = filter_col.selectbox("Filter by risk", ["All risk levels", "High", "Medium", "Low"],
                               label_visibility="collapsed", key="risk_filter", on_change=reset_paging)
    scopes = {"Needs review": pending, "High risk": high_risk, "On hold": on_hold,
              "Reviewed": [r for r in run.results if reviewed(r)], "All customers": run.results}
    view = scopes[scope]
    if band != "All risk levels":
        view = [r for r in view if r.risk.band.value == band.upper()]
    if search.strip():
        query = search.strip().casefold()
        view = [r for r in view if query in f"{r.customer_id} {reason(r)} {next_step(r)}".casefold()]
    descriptions = {
        "Needs review": "Read what the agent found, edit the draft, mark it reviewed. Nothing is sent.",
        "High risk": "High risk does not always mean contact them. Check the recommended next step.",
        "On hold": "Outreach is paused for a reason. Open a customer to understand why.",
        "Reviewed": "Responses you have reviewed in this session. These have not been sent.",
        "All customers": "Open any customer to see what the agent found and what it recommends.",
    }
    st.caption(f"{len(view)} customer{'s' if len(view) != 1 else ''} · {descriptions[scope]}")
    if not view:
        title, message = "No customers match these filters", "Try a different search or risk level."
        if not search.strip() and band == "All risk levels":
            title, message = {
                "Needs review": ("You're all caught up", "Every suggested response has been reviewed. Explore high-risk customers for more context."),
                "Reviewed": ("Your reviewed responses will appear here", "Open a customer in Needs review and mark their response reviewed."),
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
                html('<div class="row-label">Churn risk</div>' + T.band_pill(r.risk.band.value, r.risk.probability))
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


# --------------------------------------------------------------------------
# customer detail
# --------------------------------------------------------------------------


def render_investigation(r: CustomerResult) -> None:
    """The agent's run, step by step. This is the explanation."""
    html('<div class="panel-label">'
         + ("How the agent worked this out" if r.agent_run
            else "Why the agent didn't look at this customer") + '</div>')
    if r.agent_run is None:
        st.write(hold_reason(r))
        st.caption("Checks that cost nothing run before the agent does, so we never "
                   "spend a model call on someone we cannot help.")
        return
    st.caption(run_summary(r))
    html('<div class="timeline">' + "".join(
        T.step(step_label(s), s.thought, s.headline, decide=s.kind is StepKind.DECIDE)
        for s in r.agent_run.steps) + '</div>')
    with st.expander("See the raw data the agent pulled"):
        st.caption("Exactly what each tool returned, in the order the agent asked for it.")
        for s in r.agent_run.investigation:
            st.markdown(f"**{s.index}. {step_label(s)}**"
                        + (f" · `{s.arguments}`" if s.arguments else ""))
            st.json(s.observation, expanded=False)


def render_evidence(r: CustomerResult) -> None:
    f = r.features
    supply = f["days_of_supply_remaining"]
    supply_value = f"{supply:.0f} days left" if supply >= 0 else f"{abs(supply):.0f} days ago"
    html('<div class="facts">'
         f'<div class="fact"><div class="fact-label">Last delivery</div><div class="fact-value">{f["days_since_last_order"]:.0f} days ago</div></div>'
         f'<div class="fact"><div class="fact-label">{"Estimated supply" if supply >= 0 else "Estimated runout"}</div><div class="fact-value">{supply_value}</div></div>'
         f'<div class="fact"><div class="fact-label">Open support tickets</div><div class="fact-value">{f["unresolved_tickets"]:.0f}</div></div></div>')
    with st.expander("Why this risk score?"):
        st.write(f"This customer's estimated likelihood of not reordering is **{r.risk.probability:.0%}**. "
                 "It is a signal to investigate, not a certainty or a reason to send a message on its own.")
        st.caption("A statistical model compares their purchase and engagement patterns against "
                   "everyone else's. The agent is handed this number and cannot change it. "
                   "Strongest influences:")
        for a in r.risk.top(3):
            direction = "Raises risk" if a.contribution > 0 else "Lowers risk"
            st.write(f"**{direction}** · {a.label}")
        st.caption(f"High: {cfg.BAND_MEDIUM_MAX:.0%} or above · Medium: {cfg.BAND_LOW_MAX:.0%} to below "
                   f"{cfg.BAND_MEDIUM_MAX:.0%} · Low: below {cfg.BAND_LOW_MAX:.0%}. "
                   "Trained on synthetic data.")
    with st.expander("Contact checks"):
        html("".join(T.check(CHECK_LABELS.get(c.name, c.name), c.detail, c.passed) for c in r.policy.checks))
        st.caption("All four must pass before the agent is asked to look at someone.")
    if r.guardrails:
        failed = len(r.guardrails.failures)
        label = "Safety checks on the draft" + (f" · {failed} to look at" if failed else " · all clear")
        with st.expander(label):
            html("".join(T.check(CHECK_LABELS.get(c.name, c.name), c.detail, c.passed)
                         for c in r.guardrails.checks))
            st.caption("These run on whatever the agent wrote. A failure here downgrades the "
                       "recommendation to sending nothing — when we are unsure, we stay quiet.")
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


def render_draft(r: CustomerResult) -> None:
    cid = r.customer_id
    if not can_review(r):
        # The reasoning is already on the left. This panel answers the only
        # question left over: so what happens to them now?
        if not r.policy.eligible:
            outcome = what_changes(r)
        elif r.guardrails and r.guardrails.downgraded:
            outcome = ("The agent drafted something, but it did not clear our safety "
                       "checks, so nothing is being suggested. The details are under "
                       "Safety checks on the draft.")
        else:
            outcome = ("There is nothing to draft. They stay on your list, and the "
                       "agent will take another look on the next run.")
        with st.container(border=True):
            html('<div class="panel-label">What happens now</div>')
            st.subheader("No outreach for now")
            st.write(outcome)
            st.caption("A high risk score can still lead to a decision to wait. "
                       "Respecting the customer's situation is part of retention.")
        return
    d = r.diagnosis
    saved = st.session_state.drafts.get(cid, {})
    subject = saved.get("subject", d.message_subject or "")
    body = saved.get("body", d.message_body or "")
    with st.form(f"draft-form-{cid}"):
        html('<div class="panel-label">' + ("Internal handoff" if r.final_action == Action.HUMAN_ESCALATION else "The agent's draft") + '</div>')
        st.subheader("Make it personal")
        st.caption("Edit anything you like, then mark it reviewed. This demo does not send messages.")
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
         + T.pill(review_status(r), "#e9ece7", "#52604f"))
    with st.container(key="customer-detail"):
        left, right = st.columns([1, 1.2], gap="large")
    with left:
        html('<div class="panel-label" style="margin-top:1.1rem">The recommended next step</div>'
             f'<div class="recommendation"><h3>{escape(next_step(r))}</h3>'
             f'<p>{escape(explanation(r))}</p></div>')
        if r.diagnosis:
            st.caption(f"The agent read this as: {reason(r).lower()} — {confidence_label(r)}.")
        render_investigation(r)
        st.subheader("A little context")
        render_evidence(r)
    with right:
        render_draft(r)


# --------------------------------------------------------------------------
# how it works
# --------------------------------------------------------------------------


def render_flow() -> None:
    """The whole pipeline as one picture, and nothing else.

    Five bands, top to bottom: what goes in, what the deterministic layer
    does with it, what the agent chose to call, what is checked, and where a
    person takes over. Every number is read off the loaded run, so the
    diagram cannot drift away from what the code actually did.
    """
    store = load_store()
    usage = {row["tool"]: row["calls"] for row in run.tool_usage()}
    total = len(run.results)
    gate_stages = run.funnel()[:5]      # the last funnel row belongs to the agent
    checks = next((r.guardrails.checks for r in run.results if r.guardrails), [])
    # One scale across both tool groups, so the bars stay comparable.
    scale = max([usage.get(t.name, 0) for t in INVESTIGATE_TOOLS + DECIDE_TOOLS] + [1])

    st.title("The flow, end to end")
    html('<div class="lead">What goes in, what the agent chose to look at, and where a person '
         f'takes over. Every number is from the {run.as_of:%b %d, %Y} run.</div>')

    data_in = (
        '<div class="srcs">'
        + T.map_source(f"{len(store.orders):,}", "Orders", "what they bought, and when")
        + T.map_source(f"{len(store.sessions):,}", "Site sessions", "visits between purchases")
        + T.map_source(f"{len(store.messages):,}", "Email events", "clicks, not opens")
        + T.map_source(f"{len(store.tickets):,}", "Support tickets", "every word, answered or not")
        + '</div>'
        f'<div class="mcap">Recorded across {total} customers · nothing asked of them, '
        'nothing guessed</div>'
    )

    gate = (
        '<div class="mcap">A logistic regression, and no LLM anywhere in this band</div>'
        f'<div class="mnote">{len(MODEL_FEATURES)} of the {len(FEATURE_NAMES)} signals go into the '
        'model. Out comes one calibrated probability per customer, plus the exact contribution '
        'of every signal behind it.</div>'
        '<div class="mcap">Then four rules, cheapest and most absolute first</div>'
        + "".join(
            T.map_bar(s["stage"], s["n"], total,
                      f"−{s['dropped']}" if s["dropped"] else "",
                      ghost=None if i == 0 else gate_stages[i - 1]["n"])
            for i, s in enumerate(gate_stages))
    )

    agent = (
        '<div class="mcap">Handed in: the risk score and what drove it — never the raw data</div>'
        '<div class="loop">'
        + T.chips(["Think", "Call a tool", "Read the result"], arrow=True)
        + f'<div class="loop-note">Repeats until it commits to a decision · hard stop at '
          f'{MAX_STEPS} steps · running out of steps means no action, never a guess</div>'
        + '</div>'
        f'<div class="mcap">Looking things up — {stats["tool_calls"]} calls across '
        f'{stats["customers_investigated"]} customers, about {stats["avg_steps"]:g} each</div>'
        + "".join(T.map_bar(t.label, usage.get(t.name, 0), scale) for t in INVESTIGATE_TOOLS)
        + '<div class="mcap">Deciding — exactly one of these ends every run</div>'
        + "".join(T.map_bar(t.label, usage.get(t.name, 0), scale, muted=True)
                  for t in DECIDE_TOOLS)
    )

    guard = (
        '<div class="mcap">Run on what the agent produced, not on how it got there</div>'
        + T.chips([CHECK_LABELS.get(c.name, c.name) for c in checks], plain=True)
        + f'<div class="mcap">A failed check downgrades the recommendation to no action · '
          f'{stats["blocked"]} downgraded on this run</div>'
    )

    human = (
        '<div class="mcap">Every draft waits for a person</div>'
        + T.chips(["Open the customer", "Read the evidence", "Edit the draft",
                   "Checks re-run", "Mark reviewed"], arrow=True)
        + T.terminal("No send path",
                     "Nothing in this product sends a message. The flow ends with a "
                     "reviewed draft and a person deciding what to do next.")
    )

    html('<div class="map">'
         + T.map_stage("01", "Data in", "orders, visits, emails, support tickets", data_in)
         + T.map_joint(f"{len(FEATURE_NAMES)} signals per customer, "
                       "from events before the snapshot")
         + T.map_stage("02", "Scored, then gated",
                       "deterministic · the LLM never sees this band", gate)
         + T.map_joint(f"{stats['customers_investigated']} of {total} customers reach the agent")
         + T.map_stage("03", "The agent investigates",
                       f"{AGENT_KIND} · it chooses every call", agent, "ai")
         + T.map_joint("every proposal is checked")
         + T.map_stage("04", "Guardrails", "on the output, not the reasoning", guard)
         + T.map_joint(f"{stats['drafted']} drafts reach a person")
         + T.map_stage("05", "You decide", "a person reviews and edits every draft",
                       human, "human")
         + '</div>')


def render_how() -> None:
    st.title("From raw customer data to a draft you can edit.")
    html('<div class="lead">What customers did goes in. A short list of people, a reason '
         'for each, and a draft you can edit comes out. The AI does one step of this, and '
         'you sign off on the last.</div>')
    html(T.flow([
        (str(len(run.results)), "customers scanned", "everything they did, no surveys"),
        (str(len(run.results) - stats["customers_investigated"]), "ruled out first",
         "opted out, contacted recently, or still stocked"),
        (str(stats["customers_investigated"]), "investigated by the AI",
         f"{stats['tool_calls']} tool calls, its own choices"),
        (str(stats["drafted"]), "drafts for you", f"{stats['left_alone']} left alone on purpose"),
        ("0", "sent automatically", "there is no send button in this product"),
    ], highlight=2))

    usage = {row["tool"]: row["calls"] for row in run.tool_usage()}
    steps = [
        (1, "We watch what customers actually do",
         "Orders, site visits, email clicks, and every word of every support conversation. "
         "Nothing is asked of the customer and nothing is guessed."),
        (2, "A statistical model ranks who is most likely to stop buying",
         "One number per customer, plus the handful of behaviours behind it. This is ordinary "
         "statistics, not AI — and the AI is handed the number rather than allowed to invent it."),
        (3, "Cheap checks run before anything expensive does",
         f"Opted out, contacted in the last {cfg.COOLDOWN_DAYS} days, low risk, or still has "
         f"product on hand. That removed "
         f"{len(run.results) - stats['customers_investigated']} of "
         f"{len(run.results)} customers before the AI ran at all."),
        (4, "The AI agent investigates whoever is left",
         "It gets the risk score and a set of tools, and decides for itself what to look up. "
         f"On this run it made {stats['tool_calls']} tool calls across "
         f"{stats['customers_investigated']} customers — about {stats['avg_steps']:g} each — "
         "then committed to one of three decisions."),
        (5, "Safety checks run on whatever it proposed",
         "A discount with no price reason, pushy language, or outreach to someone who told "
         "support they are injured: each one downgrades the recommendation to sending nothing."),
        (6, "You decide",
         "Read what it found, edit the words, mark it reviewed. There is no send button in this "
         "product and no message leaves this browser."),
    ]
    with st.container(border=True):
        for n, title, text in steps:
            html(f'<div class="how-step"><div class="how-number">{n}</div>'
                 f'<div><h3>{escape(title)}</h3><p>{escape(text)}</p></div></div>')

    html('<div class="section-heading"><h2>What the agent can do</h2>'
         '<span>Its own choice, every time</span></div>')
    st.caption("The agent is not following a script. Each turn it picks one of these, reads the "
               "answer, and decides what to check next — until it knows enough to commit.")
    st.markdown("**Looking things up**")
    html('<div class="toolgrid">' + "".join(
        T.tool_card(t.label, t.blurb,
                    f"used {usage.get(t.name, 0)} times on this run")
        for t in INVESTIGATE_TOOLS) + '</div>')
    st.markdown("**Deciding what to do**")
    html('<div class="toolgrid">' + "".join(
        T.tool_card(t.label, t.blurb,
                    f"chosen {usage.get(t.name, 0)} times on this run", decide=True)
        for t in DECIDE_TOOLS) + '</div>')

    st.info("This is a demo with synthetic customers. Responses are drafts, nothing is sent, and "
            "your review progress lasts for this browser session.", icon=":material/info:")

    st.subheader("Watch it work on a real case")
    st.caption("Each of these is a different judgement call. The last one is the important one.")
    demos = {"CUST-0001": "It found a complaint nobody answered",
             "CUST-0002": "It stopped: they still have product",
             "CUST-0003": "It chose value over a discount",
             "CUST-0004": "We messaged them recently, so it never ran",
             "CUST-0005": "It read an injury and recommended silence"}
    for cid, label in demos.items():
        if cid in results:
            st.button(f"{label}  ·  {cid}", key=f"demo-{cid}", on_click=select, args=(cid,))

    with st.expander("Under the hood — model, agent, and evaluation"):
        st.markdown(f"**The agent** · {AGENT_KIND}")
        st.caption("Both the rule-based demo agent and Claude run the same loop over the same "
                   "tools and produce the same record of what they did, so this screen looks "
                   "the same either way. Set `USE_REAL_LLM = True` in config.py to swap.")
        st.dataframe(pd.DataFrame([
            {"Tool": t.label, "Kind": "Look something up" if t.kind is StepKind.INVESTIGATE
             else "Make a decision", "Calls on this run": usage.get(t.name, 0)}
            for t in INVESTIGATE_TOOLS + DECIDE_TOOLS
        ]), hide_index=True, width="stretch")

        st.markdown(f"**The risk model** · logistic regression on {len(MODEL_FEATURES)} "
                    "behavioural signals")
        st.dataframe(pd.DataFrame([
            {"Measure": f"Precision among the top {run.fit.k}", "Result": f"{run.fit.precision_at_k:.0%}", "Meaning": "Share of the highest-ranked test customers who churned"},
            {"Measure": "PR-AUC", "Result": f"{run.fit.pr_auc:.3f}", "Meaning": "Ranking quality across precision and recall"},
            {"Measure": "ROC-AUC", "Result": f"{run.fit.roc_auc:.3f}", "Meaning": "Ability to rank churners above non-churners"},
            {"Measure": "Brier score", "Result": f"{run.fit.brier:.3f}", "Meaning": "Prediction error; lower is better"},
        ]), hide_index=True, width="stretch")
        st.caption(f"Evaluated on {run.fit.n_test} held-out synthetic customers in a single split. "
                   "These results do not establish real-world accuracy or prove that outreach prevents churn.")

        st.markdown("**How the list narrows**")
        st.dataframe(pd.DataFrame(run.funnel())[["stage", "n"]]
                     .rename(columns={"stage": "Step", "n": "Customers remaining"}),
                     hide_index=True, width="stretch")


if st.session_state.selected in results:
    render_customer(results[st.session_state.selected])
elif st.session_state.page == "Flow map":
    render_flow()
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
