"""Design tokens and small HTML helpers for the dashboard.

Values are the validated default palette from the data-viz reference
instance: categorical slots, a fixed status palette, and the chrome/ink
roles. Kept in one file so the dashboard is written against roles rather
than raw hex, and so swapping in a brand palette is one edit.

The status four are deliberately NOT themeable and never double as series
colors -- a status hue must never impersonate a category.
"""

from __future__ import annotations

# --- chrome & ink ---------------------------------------------------------
SURFACE = "#ffffff"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
RULE = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"

# --- categorical (fixed order, never cycled) ------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"

# --- sequential blue, light -> dark ---------------------------------------
B150, B250, B350, B450, B550, B650 = (
    "#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281",
)

# --- status (fixed; always paired with a text label, never colour alone) --
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

# Risk band -> (fill, tint, ink). Tints are for pill backgrounds; the ink
# step is darkened so the label clears contrast on its own tint.
BAND = {
    "HIGH": (CRITICAL, "rgba(208,59,59,0.12)", "#992c2c"),
    "MEDIUM": (WARNING, "rgba(250,178,25,0.18)", "#8a5d00"),
    "LOW": (GOOD, "rgba(12,163,12,0.12)", "#0a7a0a"),
}

# Outcome -> pill colour. Acting is the exception, not the default, so it
# is the only one that gets an accent.
OUTCOME_TINT = {
    "acted": (BLUE, "rgba(42,120,214,0.12)", "#1c5cab"),
    "no action (agent)": (AQUA, "rgba(27,175,122,0.14)", "#0f7a55"),
    "no action (low risk)": (MUTED, "rgba(137,135,129,0.14)", "#5c5a55"),
    "suppressed (cooldown)": (MUTED, "rgba(137,135,129,0.14)", "#5c5a55"),
    "suppressed (consent)": (MUTED, "rgba(137,135,129,0.14)", "#5c5a55"),
    "suppressed (still stocked)": (MUTED, "rgba(137,135,129,0.14)", "#5c5a55"),
    "blocked (guardrail)": (ORANGE, "rgba(235,104,52,0.14)", "#a8431d"),
}

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


# --------------------------------------------------------------------------
# HTML fragments
# --------------------------------------------------------------------------


def css() -> str:
    """Injected once. Everything here is spacing, weight and hairlines --
    no colour decisions live in the stylesheet that are not tokens above."""
    return f"""
<style>
  .block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1500px; }}
  #MainMenu, footer, [data-testid="stStatusWidget"] {{ visibility: hidden; }}
  /* Nothing in the Streamlit toolbar is actionable during a demo. */
  [data-testid="stAppDeployButton"],
  [data-testid="stToolbarActions"] {{ display:none !important; }}

  h1, h2, h3, h4 {{ letter-spacing: -0.015em; }}
  h1 {{ font-size: 1.95rem !important; font-weight: 680 !important;
        margin-bottom: .15rem !important; }}
  h2 {{ font-size: 1.12rem !important; font-weight: 640 !important;
        margin: .2rem 0 .6rem !important; }}
  h3 {{ font-size: .95rem !important; font-weight: 640 !important; }}

  /* Section rule: a labelled hairline, cheaper than a heading */
  .sec {{ display:flex; align-items:center; gap:.7rem; margin:1.6rem 0 .8rem; }}
  .sec span {{ font-size:.72rem; font-weight:650; letter-spacing:.09em;
               text-transform:uppercase; color:{MUTED}; white-space:nowrap; }}
  .sec:after {{ content:""; flex:1; height:1px; background:{GRID}; }}

  /* Stat tile */
  .tiles {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; }}
  .tile {{ background:{SURFACE}; border:1px solid {GRID}; border-radius:10px;
           padding:.85rem 1rem; }}
  .tile .lab {{ font-size:.74rem; font-weight:560; color:{INK_2};
                letter-spacing:.01em; }}
  .tile .val {{ font-size:1.85rem; font-weight:660; color:{INK};
                line-height:1.15; margin-top:.15rem; }}
  /* Word values are read, not scanned -- they do not want a numeral's size */
  .tile .val.word {{ font-size:1.2rem; font-weight:640; line-height:1.3;
                     margin-top:.3rem; }}
  .tile .sub {{ font-size:.72rem; color:{MUTED}; margin-top:.1rem; }}

  /* Hero: exactly one per view */
  .hero {{ background:{PLANE}; border:1px solid {GRID}; border-radius:12px;
           padding:1.15rem 1.35rem; display:flex; align-items:baseline; gap:1.1rem; }}
  .hero .fig {{ font-size:3.1rem; font-weight:700; color:{INK}; line-height:1; }}
  .hero .txt {{ font-size:.86rem; color:{INK_2}; line-height:1.45; }}
  .hero .txt b {{ color:{INK}; font-weight:620; }}

  /* Pills -- always carry their label, never colour alone */
  .pill {{ display:inline-block; padding:.12rem .5rem; border-radius:999px;
           font-size:.71rem; font-weight:620; letter-spacing:.02em;
           white-space:nowrap; }}

  /* Risk meter: fill carries severity, track is a lighter step */
  .meter {{ height:9px; border-radius:999px; overflow:hidden; margin-top:.45rem; }}
  .meter > div {{ height:100%; border-radius:999px; }}

  /* Reasoning trace */
  .step {{ display:flex; gap:.75rem; padding:.45rem 0; border-bottom:1px solid {GRID}; }}
  .step .n {{ flex:0 0 1.35rem; height:1.35rem; border-radius:50%;
              background:{B150}; color:{B650}; font-size:.7rem; font-weight:700;
              display:flex; align-items:center; justify-content:center; }}
  .step .t {{ font-size:.85rem; color:{INK_2}; line-height:1.5; }}
  .step:last-child {{ border-bottom:none; }}

  /* Gate check rows */
  .chk {{ display:flex; gap:.6rem; align-items:flex-start; padding:.5rem .7rem;
          border-radius:8px; margin-bottom:.35rem; border:1px solid {GRID}; }}
  .chk .m {{ font-weight:700; font-size:.8rem; flex:0 0 auto; }}
  .chk .b {{ font-size:.82rem; color:{INK_2}; line-height:1.45; }}
  .chk .b b {{ color:{INK}; }}

  /* Simulated-message card */
  .sim {{ border:1px solid {GRID}; border-radius:10px; overflow:hidden; }}
  .sim .bar {{ background:rgba(235,104,52,0.10); border-bottom:1px solid {GRID};
               padding:.5rem .9rem; font-size:.74rem; font-weight:650;
               color:#a8431d; letter-spacing:.03em; }}
  .sim .subj {{ padding:.7rem .9rem .1rem; font-size:.9rem; font-weight:650; }}
  .sim .body {{ padding:.35rem .9rem 1rem; font-size:.86rem; color:{INK_2};
                line-height:1.62; white-space:pre-wrap; }}

  /* Plain-English line under a technical stat name. The name stays --
     the meaning just stops being a lookup for someone reading it cold. */
  .gloss {{ font-size:.66rem; color:{MUTED}; line-height:1.32; margin:-.14rem 0 0; }}

  .stTabs [data-baseweb="tab-list"] {{ gap:1.4rem; border-bottom:1px solid {GRID}; }}
  .stTabs [data-baseweb="tab"] {{ padding:.35rem 0; font-size:.86rem; font-weight:560; }}
  section[data-testid="stSidebar"] {{ border-right:1px solid {GRID}; }}
</style>
"""


def pill(text: str, tint: str, ink: str) -> str:
    return f'<span class="pill" style="background:{tint};color:{ink}">{text}</span>'


def band_pill(band: str) -> str:
    _, tint, ink = BAND[band]
    return pill(band, tint, ink)


def outcome_pill(outcome: str) -> str:
    _, tint, ink = OUTCOME_TINT.get(outcome, (MUTED, "rgba(137,135,129,0.14)", "#5c5a55"))
    return pill(outcome, tint, ink)


def tile(label: str, value: str, sub: str = "", word: bool = False) -> str:
    """`word=True` for text values -- a phrase set at numeral size shouts."""
    cls = "val word" if word else "val"
    return (f'<div class="tile"><div class="lab">{label}</div>'
            f'<div class="{cls}">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def meter(probability: float, band: str) -> str:
    fill, tint, _ = BAND[band]
    pct = max(2.0, min(100.0, probability * 100))
    return (f'<div class="meter" style="background:{tint}">'
            f'<div style="width:{pct:.1f}%;background:{fill}"></div></div>')
