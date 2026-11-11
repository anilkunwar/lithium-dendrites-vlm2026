#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Hao Tang (Nov 2025)
# Purpose: Generate randomized MOOSE input files from template.i. The applied overpotential
# (POT_LEFT), the Langevin noise amplitude (Noise), the interfacial energy (sigma) and the
# interface width (delta) vary; all other parameters are fixed in the template.
#
# Interface: for F = W*g(eta) + kappa/2*|grad eta|^2 with g = eta^2(1-eta)^2,
#   sigma = sqrt(2*W*kappa)/6,  delta(10-90 %) = 2.2*sqrt(2*kappa/W)
#   => kappa = 3*sigma*delta/2.2,  W = 13.2*sigma/delta
# These ignore the bulk free energy, whose contribution to the interface barrier is small only
# because the template uses As << Al (As = 1e4 J/m^3): the realized sigma and delta then lie
# within ~8 % of the targets. Raising As makes the realized interface sharper and stiffer.
# delta is drawn in elements of the fixed mesh (dx = 0.5 um): >= 5 elements resolves the tanh
# profile, <= 10 elements (5 um) keeps it well below the 10 um initial electrode and the finger
# spacing. kappa and W are written into the template; sigma and delta go to the JSON.
import random
import json
from pathlib import Path

# template file
TEMPLATE_FILE = "template.i"

# output path
OUTPUT_DIR = Path("MooseProject/generated_inputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DX = 0.5e-6                       # m, element size of the template mesh (100 um / 200)

# param range -> (min, max)
# Model time unit is ms (time_scale = 1e3), so Noise is a rate in 1/ms: 0-2e-3 /ms is the
# 0-2 /s benchmarked with L2o = 1e-3 /s (0 gives a planar front, 0.5-2 /s grow fingers).
PARAM_RANGES = {
    "POT_LEFT": (-0.3, -0.1),     # V; below -0.33 V the c >= 0 bound starts to create Li
    "Noise": (0.0, 2e-3),         # 1/ms
    "sigma": (0.3, 0.6),          # J/m^2; around the Li surface energy (~0.5, Chen et al. 2015)
    "width_elements": (5.0, 10.0),  # 10-90 % interface width, in elements
}

NUM_CASES = 1000
SEED = 2025


def interface_params(sigma: float, delta: float):
    """kappa (J/m) and W (J/m^3) for interfacial energy sigma (J/m^2) and 10-90 % width delta (m)."""
    return 3.0 * sigma * delta / 2.2, 13.2 * sigma / delta


# ====== generate and replace ======
def generate_case(template_text: str, param_ranges: dict, rng: random.Random):
    """generate with template"""
    values = {name: rng.uniform(low, high) for name, (low, high) in param_ranges.items()}
    delta = values["width_elements"] * DX
    values["interface_width_um"] = delta * 1e6
    values["kappa"], values["W"] = interface_params(values["sigma"], delta)

    new_text = template_text
    for name in ("POT_LEFT", "Noise", "kappa", "W"):
        new_text = new_text.replace(f"${name}", f"{values[name]:.6g}")
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
            print(f"    {k:18s} = {v:.4e}")
        print("-" * 40)

if __name__ == "__main__":
    main()
