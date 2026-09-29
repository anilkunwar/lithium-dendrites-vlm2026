#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Hao Tang (Nov 2025)
# Purpose: Generate randomized MOOSE input files from a template with placeholders like $ko
import random
import json
from pathlib import Path

# template file
TEMPLATE_FILE = "template.i"

# output path
OUTPUT_DIR = Path("MooseProject/generated_inputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# param range -> (min, max)
PARAM_RANGES = {
    "POT_LEFT": (-0.5, -0.2),
    "fo": (1e-3, 1e-1),
    "Al": (0, 10),
    "Bl": (0, 10),
    "Cl": (0, 10),
    "As": (0, 10),
    "Bs": (0, 10),
    "Cs": (0, 10),
    "cleq": (0.01, 0.99),
    "cseq": (0.01, 0.99),
    "L1o": (0.01, 0.5),
    "L2o": (0.01, 0.5),
    "ko": (1e-11, 1e-9),
    "Noise": (5e-4, 5e-3)
}

NUM_CASES = 100

# ====== generate and replace ======
def generate_case(template_text: str, param_ranges: dict):
    """generate with template"""
    values = {}
    new_text = template_text
    for name, (low, high) in param_ranges.items():
        val = random.uniform(low, high)
        values[name] = val
        new_text = new_text.replace(f"${name}", f"{val:.6g}")
    return new_text, values


# ====== main ======
def main():
    template = Path(TEMPLATE_FILE).read_text()

    for i in range(1, NUM_CASES + 1):
        case_name = f"case_{i:03d}"
        new_text, params = generate_case(template, PARAM_RANGES)
        new_text = new_text.replace("$CASE", f"case_{i:03d}")

        # save .i file
        i_file = OUTPUT_DIR / f"{case_name}.i"
        i_file.write_text(new_text)

        # save JSON file
        json_file = OUTPUT_DIR / f"{case_name}.json"
        with open(json_file, "w") as f:
            json.dump(params, f, indent=2)

        # console output
        print(f"[+] Generated {case_name}.i and {case_name}.json")
        for k, v in params.items():
            print(f"    {k:10s} = {v:.4e}")
        print("-" * 40)

if __name__ == "__main__":
    main()
