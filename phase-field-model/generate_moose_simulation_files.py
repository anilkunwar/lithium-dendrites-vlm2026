#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Hao Tang (Nov 2025)
# Purpose: Generate randomized MOOSE input files from template_pnk.i. Only the applied
# overpotential (POT_LEFT), the Langevin noise amplitude (Noise) and the gradient energy
# coefficient (ko) vary; all other parameters are fixed in the template (Hong & Viswanathan 2018,
# Chen et al. 2015). The ranges below come from benchmarks.
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
    "POT_LEFT": (-0.3, -0.1),  # V; below -0.33 V the c >= 0 bound starts to create Li
    "Noise": (0.0, 2.0),       # 1/s; 0 gives a planar front, 0.5-2 grow fingers
    "ko": (2e-6, 6e-6),        # J/m; 0.1-0.9 interface width 6-14 elements of 0.5 um
}

NUM_CASES = 1000
SEED = 2025

# ====== generate and replace ======
def generate_case(template_text: str, param_ranges: dict, rng: random.Random):
    """generate with template"""
    values = {}
    new_text = template_text
    for name, (low, high) in param_ranges.items():
        val = rng.uniform(low, high)
        values[name] = val
        new_text = new_text.replace(f"${name}", f"{val:.6g}")
    return new_text, values


# ====== main ======
def main():
    template = Path(TEMPLATE_FILE).read_text()
    rng = random.Random(SEED)

    for i in range(1, NUM_CASES + 1):
        case_name = f"case_{i:03d}"
        new_text, params = generate_case(template, PARAM_RANGES, rng)
        new_text = new_text.replace("$CASE", case_name)
        assert "$" not in new_text.replace("${fparse", ""), "unfilled placeholder in template"

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
