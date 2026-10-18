"""Build the pilot dataset: OpenExempt Task AE (allowable exemption jurisdictions) with
solver-verified counterfactual pairs.

The vendored OpenExempt framework supplies case sampling, natural-language templates and the
730/180-day domicile rule. Two deviations from the published benchmark, both for the pilot:
  1. The prompt contains only the statutes that decide Task AE (11 U.S.C. 522(b) and the state
     opt-out provisions), not every exemption statute, to keep prompts short enough to train on.
  2. Each test case gets edited twins (one domicile date shifted) whose label is recomputed with
     OpenExempt's own rule. Original and twin are rendered with the same random seed, so their
     texts differ only in the shifted date.

Usage: python -I src/ae_data.py --vendor vendor/OpenExempt --out data --train 1000 --test 300
"""
import argparse
import json
import os
import random
import sys
from datetime import timedelta


def load_openexempt(vendor_dir):
    sys.path.insert(0, os.path.abspath(vendor_dir))
    from source.config import Config
    from source.case_generator import CaseGenerator
    from source.task_generator import TaskGenerator
    from source.jurisdiction import Jurisdiction
    return Config, CaseGenerator, TaskGenerator, Jurisdiction


STATES = ["ARIZONA", "ILLINOIS", "OREGON", "PENNSYLVANIA", "WISCONSIN"]


def make_config(Config, vendor_dir, domicile_min, domicile_max, size):
    v = os.path.abspath(vendor_dir)
    cfg = {
        "data_directory": f"{v}/data", "asset_directory": f"{v}/data/assets",
        "statute_directory": f"{v}/data/statutes", "template_directory": f"{v}/data/templates",
        "output_directory": f"{v}/tasks", "start_task_id": 1, "terminal_task_id": 1,
        "dataset_size": size, "asset_count_min": 1, "asset_count_max": 1,
        "asset_selection": "random", "married_ratio": 0.5,
        "domicile_count_min": domicile_min, "domicile_count_max": domicile_max,
        "irrelevant_asset_facts": False, "irrelevant_domicile_facts": False,
        "asset_opinions": False, "domicile_opinions": False, "state_jurisdictions": STATES,
    }
    return Config(cfg, "pilot_ae", verbose=False)


def focused_statutes(task_generator):
    """Statute text that decides Task AE: federal 522(b) plus each state's opt-out provision."""
    parts = []
    for jurisdiction, statute_set in task_generator.statute_set_map.items():
        if jurisdiction.is_federal():
            keep = [s for s in statute_set.non_exemptions() if s.citation.endswith("522(b)")]
        else:
            keep = [s for s in statute_set.non_exemptions()
                    if "522" in s.description and ("not entitled" in s.description or "prohibited" in s.description)]
        body = "\n\n".join(s.display_content() for s in keep) if keep else "(No provision on the federal exemptions.)"
        parts.append(f"{statute_set.authority}\n{body}")
    return "Statutes:\n" + "\n\n".join(parts)


def render_prompt(TaskGenerator, config, case, seed, statutes):
    """Render the fact pattern deterministically for a given seed."""
    random.seed(seed)
    tg = TaskGenerator(config)
    facts = "Facts:\n" + tg.create_preamble(case.debtor, case.joint_debtor, case.petition_date)
    names = tg.create_name_variant_sampler(case.debtor, case.joint_debtor)
    facts += " " + " ".join(tg.create_domicile_facts(case, names))
    ins = tg.instructions
    return "\n\n".join([
        ins["1"],
        ins["meta_without_assets"],
        "Response Format: " + ins["answer_format"] + " " + ins["1_response_format"],
        facts,
        statutes,
        "Reason step by step in at most 200 words, then give your answer. " + ins["format_reminder_list"],
    ])


def gold_answer(task_generator, Jurisdiction, state):
    allowable = task_generator.statute_set_map[state].allowable_exemption_jurisdictions()
    return sorted(j.display_name() for j in allowable)


def applicable_state(case_generator, case, dates_states):
    """OpenExempt's own 730/180-day rule on an ordered {date: Jurisdiction} map."""
    return case_generator.determine_applicable_state_jurisdiction(case.petition_date, dates_states)


def make_twins(case_generator, case, rng, tries=200):
    """Shift one in-window domicile date; keep one twin whose label flips and one whose label holds."""
    dates = list(case.domicile_dates.keys())
    states = [case.domicile_dates[d] for d in dates]
    lookback = case.petition_date - timedelta(days=730 + 180)
    base_state = applicable_state(case_generator, case, dict(zip(dates, states)))
    found = {}
    for _ in range(tries):
        if len(found) == 2:
            break
        i = rng.randrange(1, len(dates))
        shift = rng.choice([-1, 1]) * rng.randint(20, 240)
        new = dates[i] + timedelta(days=shift)
        lower = max(dates[i - 1], lookback)
        upper = dates[i + 1] if i + 1 < len(dates) else case.petition_date
        if not (lower < new < upper):
            continue
        new_dates = dates[:i] + [new] + dates[i + 1:]
        new_state = applicable_state(case_generator, case, dict(zip(new_dates, states)))
        if new_state is None:  # tie between two states: label undefined
            continue
        kind = "operative" if new_state != base_state else "invariant"
        if kind not in found:
            found[kind] = (dict(zip(new_dates, states)), new_state, i, shift)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domicile_min", type=int, default=2)
    ap.add_argument("--domicile_max", type=int, default=4)
    args = ap.parse_args()

    Config, CaseGenerator, TaskGenerator, Jurisdiction = load_openexempt(args.vendor)
    random.seed(args.seed)
    config = make_config(Config, args.vendor, args.domicile_min, args.domicile_max, args.train + args.test)
    case_generator = CaseGenerator(config)
    task_generator = TaskGenerator(config)
    statutes = focused_statutes(task_generator)
    rng = random.Random(args.seed + 1)
    cases = [case_generator.generate_case() for _ in range(args.train + args.test)]

    os.makedirs(args.out, exist_ok=True)
    stats = {"train": 0, "test": 0, "operative": 0, "invariant": 0, "label_check_failures": 0}
    with open(os.path.join(args.out, "train.jsonl"), "w") as ftr, open(os.path.join(args.out, "test.jsonl"), "w") as fte:
        for idx, case in enumerate(cases):
            state = applicable_state(case_generator, case, case.domicile_dates)
            if state != case.state_jurisdiction:  # generator guarantees this; check anyway
                stats["label_check_failures"] += 1
                continue
            seed = args.seed * 1_000_003 + idx
            record = {
                "id": f"ae-{idx}",
                "prompt": render_prompt(TaskGenerator, config, case, seed, statutes),
                "answer": gold_answer(task_generator, Jurisdiction, state),
                "domicile_count": case.domicile_count(),
                "case": case.serialize(),
            }
            if idx < args.train:
                ftr.write(json.dumps(record) + "\n")
                stats["train"] += 1
                continue
            twins = []
            for kind, (dates_states, new_state, i, shift) in make_twins(case_generator, case, rng).items():
                orig_dates = case.domicile_dates
                case.domicile_dates = dates_states
                twin_prompt = render_prompt(TaskGenerator, config, case, seed, statutes)
                case.domicile_dates = orig_dates
                twins.append({"kind": kind, "prompt": twin_prompt,
                              "answer": gold_answer(task_generator, Jurisdiction, new_state),
                              "edited_index": i, "shift_days": shift})
                stats[kind] += 1
            record["twins"] = twins
            fte.write(json.dumps(record) + "\n")
            stats["test"] += 1
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
