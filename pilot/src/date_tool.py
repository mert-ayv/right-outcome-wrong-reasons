"""Date tool for the H2d check (PREREGISTRATION_H2D.md): calendar arithmetic is done for the model.

The tool output lists each move as a day number relative to the filing date (filing = day 0). The
model still has to know the 730-day and 180-day windows from the statute, count the days per state
in them and apply the rule. tool_trace writes the solver's derivation with that window arithmetic,
using the same step tags as proof_steps_v2, for the SFT warm start.
"""
from datetime import datetime, timedelta

OPTED_OUT = {"ARIZONA", "ILLINOIS"}


def twin_case(case, tw):
    ds = sorted(case["domicile_dates"].items())
    d, s = ds[tw["edited_index"]]
    ds[tw["edited_index"]] = ((datetime.fromisoformat(d) + timedelta(days=tw["shift_days"])).isoformat(), s)
    return dict(case, domicile_dates=dict(ds))


def periods(case):
    """[(state, start_day, end_day)] relative to the filing date; the last period ends on day 0."""
    petition = datetime.fromisoformat(case["petition_date"])
    ds = sorted((datetime.fromisoformat(k), v.title()) for k, v in case["domicile_dates"].items())
    out = []
    for i, (d, s) in enumerate(ds):
        end = ds[i + 1][0] if i + 1 < len(ds) else petition
        out.append((s, (d - petition).days, (end - petition).days))
    return out


def tool_block(case):
    petition = datetime.fromisoformat(case["petition_date"])
    lines = ["Date calculator output (day numbers relative to the filing date; the filing is day 0, earlier "
             "days are negative):"]
    for k, v in sorted(case["domicile_dates"].items()):
        d = datetime.fromisoformat(k)
        lines.append(f"- {d.date().isoformat()}, start of domicile in {v.title()}: day {(d - petition).days}")
    lines.append(f"- {petition.date().isoformat()}, filing date: day 0")
    return "\n".join(lines)


def add_tool(prompt, case):
    assert prompt.count("\nStatutes:") == 1
    return prompt.replace("\nStatutes:", "\n" + tool_block(case) + "\n\nStatutes:", 1)


def _overlap(a, b, lo, hi):
    return max(0, min(b, hi) - max(a, lo))


def tool_trace(case):
    """Solver derivation in day numbers; returns (text, answer)."""
    ps = periods(case)
    in730 = [s for s, a, b in ps if _overlap(a, b, -730, 0) > 0]
    states730 = sorted(set(in730))
    lines = ["The filing is day 0. The 730-day period is days -730 to 0, and the 180 days before it are days "
             "-910 to -730.",
             "Domicile: " + ", ".join(f"{s} from day {a} to day {b}" for s, a, b in ps) + ".",
             f"States of domicile in the 730-day period: {', '.join(states730)}."]
    if len(states730) == 1:
        state = states730[0]
        lines.append(f"[730: {state}]")
    else:
        days = {}
        for s, a, b in ps:
            n = _overlap(a, b, -910, -730)
            if n:
                days[s] = days.get(s, 0) + n
        work = ", ".join(f"{s} from day {max(a, -910)} to day {min(b, -730)} = {_overlap(a, b, -910, -730)} days"
                         for s, a, b in ps if _overlap(a, b, -910, -730))
        lines.append(f"[730: MULTIPLE] The debtor lived in more than one state, so the 180 days before the 730-day "
                     f"period count (days -910 to -730): {work}.")
        lines.append("[180-DAYS: " + ", ".join(f"{s} {n}" for s, n in sorted(days.items(), key=lambda x: -x[1])) + "]")
        state = max(days, key=days.get)
    opt = "yes" if state.upper() in OPTED_OUT else "no"
    answer = [state] if opt == "yes" else sorted(["Federal", state])
    lines.append(f"[DECISIVE: {state}]")
    lines.append(f"{state} law {'excludes' if opt == 'yes' else 'does not exclude'} the federal exemptions. "
                 f"[OPT-OUT {state}: {opt}]")
    lines.append("FINAL ANSWER: " + ", ".join(answer))
    return "\n".join(lines), answer
