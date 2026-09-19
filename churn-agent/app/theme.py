"""Visual language for a quiet, action-focused retention workspace."""

from html import escape

SURFACE = "#ffffff"
PLANE = "#f6f7f4"
INK = "#202b27"
INK_2 = "#59655e"
MUTED = "#68756d"
GRID = "#e1e6df"
GOOD = "#266854"
WARNING = "#956515"
CRITICAL = "#b7463c"
BAND = {
    "HIGH": (CRITICAL, "#fbeeea", "#a03830"),
    "MEDIUM": (WARNING, "#faf2df", "#80580e"),
    "LOW": (GOOD, "#edf4ef", "#26604b"),
}


def css() -> str:
    return f"""
<style>
  .stApp {{ background: {PLANE}; }}
  .block-container {{ max-width: 1280px; padding: 2.6rem 3.2rem 3rem; container-type: inline-size; container-name: workspace; }}
  #MainMenu, footer {{ visibility: hidden; }}
  [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"] {{ display: none; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  h1, h2, h3 {{ color: {INK}; letter-spacing: -.04em; }}
  h1 {{ font-size: 2.55rem !important; font-weight: 650 !important; line-height: 1.15 !important;
        padding-bottom: .45rem !important; }}
  h2 {{ font-size: 1.35rem !important; font-weight: 650 !important; padding-bottom: .4rem !important; }}
  h3 {{ font-size: 1.06rem !important; font-weight: 620 !important; }}
  p, label {{ line-height: 1.55; }}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: {MUTED}; }}
  [data-testid="stVerticalBlock"] {{ gap: .85rem; }}
  [data-testid="stSidebar"] {{ background: #eef1eb; border-right: 1px solid {GRID}; }}
  [data-testid="stSidebar"] > div:first-child {{ padding-top: 2.2rem; }}
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .65rem; }}
  [data-testid="stSidebar"] .stButton button {{ justify-content: flex-start; padding: .65rem .85rem; }}
  [data-testid="stSidebar"] .stButton button[kind="secondary"] {{ background: transparent; border-color: transparent; }}
  [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {{ background: #e3e9e0; }}
  .brand {{ display: flex; align-items: center; gap: .65rem; font-size: 1.22rem;
            font-weight: 730; letter-spacing: -.04em; color: {INK}; margin-bottom: .2rem; }}
  .brand-mark {{ display: grid; place-items: center; width: 32px; height: 32px;
                 background: #264f3d; color: #dff2b1; border-radius: 10px; font-size: 23px; }}
  .brand-sub {{ color: {MUTED}; font-size: .76rem; margin: 0 0 2.3rem 2.65rem; }}
  .eyebrow {{ font-size: .68rem; font-weight: 700; letter-spacing: .12em;
              text-transform: uppercase; color: {MUTED}; margin: 1.1rem 0 .5rem; }}
  .sidebar-note {{ margin-top: 2.4rem; border-top: 1px solid #d7ded3; padding-top: 1.1rem;
                   font-size: .77rem; color: {INK_2}; line-height: 1.65; }}
  .topline {{ display: flex; justify-content: space-between; align-items: center; gap: 1rem;
              margin-bottom: 1.65rem; font-size: .76rem; color: {MUTED}; scroll-margin-top: 4rem; }}
  .topline strong {{ font-weight: 550; color: {INK_2}; }}
  .lead {{ font-size: 1rem; color: {INK_2}; margin-bottom: 1.1rem; }}
  .pill {{ display: inline-block; padding: .22rem .58rem; border-radius: 6px;
           font-size: .7rem; line-height: 1.4; font-weight: 650; white-space: nowrap; }}
  .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: .3rem 0 1.4rem; }}
  .stat {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 12px;
           padding: 1.1rem 1.3rem; }}
  .stat.featured {{ background: #e9efde; border-color: #dce5cb; }}
  .stat-label {{ font-size: .79rem; font-weight: 550; color: {INK_2}; }}
  .stat-value {{ font-size: 2.35rem; font-weight: 620; line-height: 1.3; letter-spacing: -.045em;
                 color: {INK}; margin: .3rem 0; font-variant-numeric: tabular-nums; }}
  .stat-sub {{ font-size: .73rem; color: {INK_2}; }}
  .section-heading {{ display: flex; align-items: center; justify-content: space-between;
                      gap: .5rem; margin: .3rem 0 .2rem; }}
  .section-heading h2 {{ margin: 0; padding: 0 !important; }}
  .section-heading > span {{ font-size: .74rem; color: {MUTED}; }}
  [class*="st-key-customer-row-"] {{ background: {SURFACE}; border-radius: 12px; }}
  [data-testid="stVerticalBlockBorderWrapper"] > div {{ border-radius: 12px; }}
  [data-testid="stVerticalBlockBorderWrapper"]:has(.customer-name),
  [data-testid="stVerticalBlockBorderWrapper"]:has(.panel-label) {{ background: {SURFACE}; }}
  .customer-name {{ font-size: .93rem; font-weight: 650; letter-spacing: -.015em; color: {INK}; }}
  .customer-reason {{ margin-top: .28rem; font-size: .76rem; color: {INK_2}; line-height: 1.5; }}
  .row-label {{ font-size: .64rem; text-transform: uppercase; letter-spacing: .07em;
                color: {MUTED}; margin-bottom: .25rem; }}
  .action-name {{ font-size: .82rem; font-weight: 550; color: {INK}; }}
  .row-status {{ font-size: .72rem; color: {MUTED}; margin-top: .2rem; }}
  .panel-label {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .1em;
                  color: {MUTED}; font-weight: 650; margin-bottom: .6rem; }}
  .recommendation {{ background: #edf2e5; border: 1px solid #dde6d1; border-radius: 10px;
                     padding: 1.2rem; margin: .3rem 0 1rem; }}
  .recommendation h3 {{ margin: 0 0 .4rem; padding: 0; }}
  .recommendation p {{ font-size: .85rem; color: {INK_2}; margin: 0; }}
  .facts {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .5rem; margin: .7rem 0 1rem; }}
  .fact {{ border-right: 1px solid {GRID}; padding-right: .6rem; }}
  .fact:last-child {{ border-right: 0; }}
  .fact-value {{ font-size: 1.1rem; font-weight: 640; margin: .25rem 0; }}
  .fact-label {{ font-size: .72rem; color: {MUTED}; line-height: 1.4; }}
  .check {{ display: flex; align-items: flex-start; gap: .65rem; padding: .7rem 0;
            border-bottom: 1px solid {GRID}; font-size: .8rem; color: {INK_2}; }}
  .check:last-child {{ border-bottom: 0; }}
  .check strong {{ color: {INK}; font-weight: 600; }}
  .check-mark {{ font-size: .65rem; font-weight: 700; padding-top: .15rem; min-width: 30px; }}
  .empty {{ text-align: center; padding: 3rem 1rem; background: {SURFACE};
            border: 1px dashed #cbd5c8; border-radius: 12px; }}
  .empty h3 {{ font-size: 1.1rem !important; margin-bottom: .4rem; }}
  .empty p {{ color: {MUTED}; font-size: .86rem; margin: 0; }}
  .how-step {{ display: flex; gap: 1.1rem; padding: 1rem 0; }}
  .how-number {{ flex: 0 0 34px; height: 34px; background: #e9efde; border-radius: 50%;
                 display: grid; place-items: center; color: #355239; font-weight: 650; }}
  .how-step h3 {{ margin: 0 0 .25rem; padding: 0; }}
  .how-step p {{ margin: 0; font-size: .88rem; color: {INK_2}; }}
  .stButton button, .stDownloadButton button, .stFormSubmitButton button {{ font-size: .8rem;
        min-height: 2.45rem; border-radius: 8px; font-weight: 550; }}
  button:focus-visible, input:focus-visible, textarea:focus-visible {{ outline: 2px solid #266854 !important;
       outline-offset: 3px; }}
  [data-testid="stRadio"] [role="radiogroup"] {{ gap: .45rem 1.3rem; }}
  [data-testid="stRadio"] label p {{ font-size: .82rem; }}
  [data-testid="stForm"] {{ background: {SURFACE}; border-color: {GRID}; border-radius: 12px; }}
  [data-testid="stForm"] input, [data-testid="stForm"] textarea {{
    border: 1px solid #d4ddd1; border-radius: 8px; background: #fcfdfb; padding: .75rem;
  }}
  [data-testid="stExpander"] {{ background: {SURFACE}; border-radius: 10px; }}
  @container workspace (max-width: 760px) {{
    .st-key-customer-detail [data-testid="stHorizontalBlock"]:has(.recommendation) > [data-testid="stColumn"] {{
      width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important;
    }}
    [class*="st-key-customer-row-"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
      width: calc(50% - 1rem) !important; flex: 1 1 calc(50% - 1rem) !important; min-width: 0 !important;
    }}
    .st-key-queue-filters [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
      width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important;
    }}
  }}
  @media (max-width: 850px) {{
    .block-container {{ padding: 2.2rem 1.2rem; }}
    h1 {{ font-size: 2rem !important; }}
    .stats {{ gap: .6rem; }}
    .stat {{ padding: .8rem; }}
    .stat-value {{ font-size: 1.85rem; }}
    .topline {{ flex-wrap: wrap; }}
  }}
  @media (max-width: 520px) {{
    .stats {{ grid-template-columns: 1fr; }}
    .stat {{ display: grid; grid-template-columns: 1fr auto; align-items: center; }}
    .stat-value {{ grid-column: 2; grid-row: 1 / 3; margin: 0; }}
    .stat-sub {{ grid-column: 1; }}
    .section-heading {{ align-items: flex-start; flex-direction: column; }}
  }}
</style>
"""


def pill(text: str, tint: str = "#e8eee3", ink: str = "#37533c") -> str:
    return f'<span class="pill" style="background:{tint};color:{ink}">{escape(text)}</span>'


def band_pill(band: str, probability: float | None = None) -> str:
    _, tint, ink = BAND[band]
    label = f"{band.title()} risk" if probability is None else f"{probability:.0%} · {band.title()}"
    return pill(label, tint, ink)


def stat(label: str, value: str, sub: str, featured: bool = False) -> str:
    return (f'<div class="stat{" featured" if featured else ""}">'
            f'<div class="stat-label">{escape(label)}</div>'
            f'<div class="stat-value">{escape(value)}</div>'
            f'<div class="stat-sub">{escape(sub)}</div></div>')


def check(label: str, detail: str, passed: bool) -> str:
    color, mark = (GOOD, "PASS") if passed else (WARNING, "HOLD")
    return (f'<div class="check"><span class="check-mark" style="color:{color}">{mark}</span>'
            f'<div><strong>{escape(label)}</strong><br>{escape(detail)}</div></div>')
