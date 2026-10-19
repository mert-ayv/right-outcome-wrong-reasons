"""Finer proof steps for OpenExempt Task AE (H2b pilot, PREREGISTRATION_H2B.md).

Steps a model cites inline:
  [730: <State>] or [730: MULTIPLE]          domicile over the 730 days before the petition
  [180-DAYS: <State> <n>, <State> <n>]       days per state in the 180 days before that window
                                             (only if MULTIPLE)
  [DECISIVE: <State>]                        the state whose law applies
  [OPT-OUT <State>: yes|no]                  whether that state excludes the federal exemptions

process_reward_v2 weights the deciding computation (DECISIVE and 180-DAYS) and credits the easy
steps (730, OPT-OUT) only when DECISIVE is correct. solver_trace writes a worked derivation used for
the SFT warm start.
"""
import re
from datetime import datetime, timedelta

OPTED_OUT = {"ARIZONA", "ILLINOIS"}
DAY_TOLERANCE = 2
TAG_INSTRUCTION_V2 = (
    "While reasoning, state each rule application in exactly these forms: [730: <State>] if the debtor "
    "was domiciled in a single state for the whole 730 days before the petition, otherwise [730: MULTIPLE] "
    "followed by [180-DAYS: <State> <days>, <State> <days>] giving the days domiciled in each state during "
    "the 180 days before that 730-day period; then [DECISIVE: <State>] for the state whose exemption law "
    "applies, and [OPT-OUT <State>: yes] or [OPT-OUT <State>: no] for whether that state's law excludes "
    "the federal exemptions."
)


def tag_prompt_v2(prompt):
    marker = "Reason step by step"
    i = prompt.rfind(marker)
    return prompt[:i] + TAG_INSTRUCTION_V2 + "\n\n" + prompt[i:] if i >= 0 else prompt + "\n\n" + TAG_INSTRUCTION_V2


def _days(ds, start, end, petition):
    out = {}
    for i, (d, s) in enumerate(ds):
        nxt = ds[i + 1][0] if i + 1 < len(ds) else petition
        a, b = max(d, start), min(nxt, end)
        if (b - a).days > 0:
            out[s] = out.get(s, 0) + (b - a).days
    return out


def gold_steps_v2(case):
    petition = datetime.fromisoformat(case["petition_date"])
    ds = sorted((datetime.fromisoformat(k), v) for k, v in case["domicile_dates"].items())
    w730, w180 = petition - timedelta(days=730), petition - timedelta(days=910)
    in730 = _days(ds, w730, petition, petition)
    steps = {"petition": petition.date().isoformat(), "w730": w730.date().isoformat(),
             "w180": w180.date().isoformat(), "states730": sorted(s.title() for s in in730)}
    if len(in730) == 1:
        state = next(iter(in730))
        steps["730"] = state.title()
    else:
        steps["730"] = "MULTIPLE"
        in180 = _days(ds, w180, w730, petition)
        steps["180"] = {s.title(): n for s, n in in180.items()}
        state = max(in180, key=in180.get)
    steps["DECISIVE"] = state.title()
    steps["OPTOUT"] = "yes" if state in OPTED_OUT else "no"
    answer = [state.title()] if state in OPTED_OUT else sorted(["Federal", state.title()])
    return steps, answer


def solver_trace(steps, answer):
    st = steps["DECISIVE"]
    lines = [f"The petition was filed on {steps['petition']}, so the 730-day period starts on {steps['w730']}. "
             f"States of domicile in that period: {', '.join(steps['states730'])}."]
    if steps["730"] != "MULTIPLE":
        lines.append(f"[730: {steps['730']}]")
    else:
        days = ", ".join(f"{s} {n}" for s, n in sorted(steps["180"].items(), key=lambda x: -x[1]))
        lines.append("[730: MULTIPLE] The debtor lived in more than one state, so the 180 days before the 730-day "
                     f"period count, from {steps['w180']} to {steps['w730']}: {days} days.")
        lines.append(f"[180-DAYS: {days}]")
    lines.append(f"[DECISIVE: {st}]")
    excl = "excludes" if steps["OPTOUT"] == "yes" else "does not exclude"
    lines.append(f"{st} law {excl} the federal exemptions. [OPT-OUT {st}: {steps['OPTOUT']}]")
    lines.append("FINAL ANSWER: " + ", ".join(answer))
    return "\n".join(lines)


def parse_steps_v2(text):
    out = {}
    m = re.findall(r"\[\s*730\s*:\s*([A-Za-z]+)\s*\]", text)
    if m:
        out["730"] = m[-1].upper()
    m = re.findall(r"\[\s*180-DAYS\s*:\s*([^\]]+)\]", text, re.IGNORECASE)
    if m:
        days = {}
        for s, n in re.findall(r"([A-Za-z]+)\s+(\d+)", m[-1]):
            days[s.upper()] = int(n)
        out["180"] = days
    m = re.findall(r"\[\s*DECISIVE\s*:\s*([A-Za-z]+)\s*\]", text, re.IGNORECASE)
    if m:
        out["DECISIVE"] = m[-1].upper()
    m = re.findall(r"\[\s*OPT-?OUT\s+([A-Za-z]+)\s*:\s*(yes|no)\s*\]", text, re.IGNORECASE)
    if m:
        out["OPTOUT"] = (m[-1][0].upper(), m[-1][1].lower())
    return out


def step_score_v2(text, steps):
    """Step score in [0, 1]: DECISIVE and 180-DAYS carry the weight; easy steps only count if DECISIVE is right."""
    c = parse_steps_v2(text)
    r_dec = 1.0 if c.get("DECISIVE") == steps["DECISIVE"].upper() else 0.0
    easy = [1.0 if c.get("730") == steps["730"].upper() else 0.0,
            1.0 if c.get("OPTOUT") == (steps["DECISIVE"].upper(), steps["OPTOUT"]) else 0.0]
    r_easy = r_dec * sum(easy) / 2
    if steps["730"] != "MULTIPLE":
        return 0.8 * r_dec + 0.2 * r_easy
    gold = {s.upper(): n for s, n in steps["180"].items()}
    cited = c.get("180", {})
    hit = sum(1 for s, n in gold.items() if s in cited and abs(cited[s] - n) <= DAY_TOLERANCE)
    r_counts = hit / max(len(gold), len(cited)) if cited else 0.0
    return 0.5 * r_dec + 0.3 * r_counts + 0.2 * r_easy


def full_derivation_correct(text, steps):
    return step_score_v2(text, steps) >= 0.999
