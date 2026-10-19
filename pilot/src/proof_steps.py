"""Gold proof steps for OpenExempt Task AE and a parser for the steps a model cites.

The derivation under 11 U.S.C. 522(b)(3)(A) as encoded in OpenExempt (CaseGenerator.
determine_applicable_state_jurisdiction) has up to three steps:
  730    the state of domicile for the whole 730 days before the petition, or MULTIPLE
  180    if MULTIPLE: the state domiciled longest in the 180 days before the 730-day window
  OPTOUT whether that state's law excludes the federal exemptions (Arizona, Illinois)
The answer is [state] if the state opted out, else [Federal, state].

Models cite steps inline as  [730: Oregon]  or  [730: MULTIPLE],  [180: Oregon],
[OPT-OUT Oregon: no].  step_f1 compares the cited set with the gold set.
"""
import re
from datetime import datetime, timedelta

OPTED_OUT = {"ARIZONA", "ILLINOIS"}
TAG_INSTRUCTION = (
    "While reasoning, state each rule application in exactly one of these forms: "
    "[730: <State>] if the debtor was domiciled in a single state for the whole 730 days before the petition, "
    "otherwise [730: MULTIPLE] followed by [180: <State>] for the state of longest domicile in the 180 days "
    "before that 730-day period; then [OPT-OUT <State>: yes] or [OPT-OUT <State>: no] for whether that "
    "state's law excludes the federal exemptions."
)


def tag_prompt(prompt):
    """Insert the step instruction before the final reasoning/format line of a Task AE prompt."""
    marker = "Reason step by step"
    i = prompt.rfind(marker)
    return prompt[:i] + TAG_INSTRUCTION + "\n\n" + prompt[i:] if i >= 0 else prompt + "\n\n" + TAG_INSTRUCTION


def _segments(petition, dates_states, start):
    """Days per state within [start, petition), from an ordered list of (move date, state)."""
    days = {}
    for i, (d, s) in enumerate(dates_states):
        nxt = dates_states[i + 1][0] if i + 1 < len(dates_states) else petition
        a, b = max(d, start), min(nxt, petition)
        if (b - a).days > 0:
            days[s] = days.get(s, 0) + (b - a).days
    return days


def gold_steps(case):
    """case: serialized OpenExempt case (petition_date, domicile_dates {iso: STATE})."""
    petition = datetime.fromisoformat(case["petition_date"])
    ds = sorted((datetime.fromisoformat(k), v) for k, v in case["domicile_dates"].items())
    w730 = petition - timedelta(days=730)
    in730 = _segments(petition, ds, w730)
    steps = {}
    if len(in730) == 1:
        state = next(iter(in730))
        steps["730"] = state.title()
    else:
        steps["730"] = "MULTIPLE"
        w180_start = petition - timedelta(days=730 + 180)
        in180 = {}
        for i, (d, s) in enumerate(ds):
            nxt = ds[i + 1][0] if i + 1 < len(ds) else petition
            a, b = max(d, w180_start), min(nxt, w730)
            if (b - a).days > 0:
                in180[s] = in180.get(s, 0) + (b - a).days
        state = max(in180, key=in180.get)
        steps["180"] = state.title()
    steps["OPTOUT"] = (state.title(), "yes" if state in OPTED_OUT else "no")
    answer = [state.title()] if state in OPTED_OUT else sorted(["Federal", state.title()])
    return steps, answer


def _step_set(steps):
    out = {("730", steps["730"].lower())}
    if "180" in steps:
        out.add(("180", steps["180"].lower()))
    st, yn = steps["OPTOUT"]
    out.add(("OPTOUT", f"{st.lower()}:{yn}"))
    return out


def cited_steps(text):
    out = set()
    for v in re.findall(r"\[\s*730\s*:\s*([A-Za-z]+)\s*\]", text):
        out.add(("730", v.lower()))
    for v in re.findall(r"\[\s*180\s*:\s*([A-Za-z]+)\s*\]", text):
        out.add(("180", v.lower()))
    for s, yn in re.findall(r"\[\s*OPT-?OUT\s+([A-Za-z]+)\s*:\s*(yes|no)\s*\]", text, re.IGNORECASE):
        out.add(("OPTOUT", f"{s.lower()}:{yn.lower()}"))
    return out


def step_f1(text, steps):
    gold, cited = _step_set(steps), cited_steps(text)
    if not cited:
        return 0.0
    tp = len(gold & cited)
    if tp == 0:
        return 0.0
    p, r = tp / len(cited), tp / len(gold)
    return 2 * p * r / (p + r)
