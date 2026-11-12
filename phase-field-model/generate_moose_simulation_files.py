#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Hao Tang (Nov 2025)
# Purpose: Generate randomized MOOSE input files from template.i. The applied overpotential
# (POT_LEFT), the reaction-rate scatter (Noise), the interfacial energy (sigma) and the
# interface width (delta) vary; all other parameters are fixed in the template.
#
# Interface: the template's free energy is fl(c) + W*g(eta) + kappa/2*|grad eta|^2 (fl is not
# weighted by h), so the interface is set by the double well alone, g = eta^2(1-eta)^2:
#   sigma = sqrt(2*W*kappa)/6,  delta(10-90 %) = 2.2*sqrt(2*kappa/W)
#   => kappa = 3*sigma*delta/2.2,  W = 13.2*sigma/delta
# delta is drawn in elements of the fixed mesh (dx = 0.5 um): >= 5 elements resolves the tanh
# profile, <= 10 elements (5 um) keeps it well below the 10 um initial electrode and the finger
# spacing. kappa and W are written into the template; sigma and delta go to the JSON.
#
# Interface mobility: the Butler-Volmer rate grows as exp(alpha*F*|POT_LEFT|/RT), ~870x from
# -0.1 to -0.45 V, while the Allen-Cahn restoring rate L1*W does not depend on the potential. L1o
# is therefore set per case so that their ratio is the same for every case:
#   r = L2o*exp(alpha*F*|POT_LEFT|/RT)*max(h') / (L1o*W),  max(h') = 1.875
# With a reaction-limited front speed ~ L2o*exp(...) and a curvature smoothing speed ~ L1o, this
# also fixes their ratio, so the potential changes the morphology through Li+ transport only.
# r = 2 at -0.45 V (fixed L1o = 2.5e-6) broadened the interface; r = 0.3 at -0.35 V kept it sharp.
import math
import random
import re
import json
from pathlib import Path

# template file
TEMPLATE_FILE = "template.i"

# output path
OUTPUT_DIR = Path("MooseProject/generated_inputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DX = 0.5e-6                       # m, element size of the template mesh (100 um / 200)

# param range -> (min, max)
# Noise scales a frozen random field xi in [-1, 1] on the Butler-Volmer rate, L2*(1 + Noise*xi);
# it must stay < 1 so the local rate constant stays positive.
PARAM_RANGES = {
    "POT_LEFT": (-0.45, -0.1),      # V; planar front above ~-0.25 V, fingers toward -0.35 V and below
    "Noise": (0.0, 0.9),            # dimensionless
    "sigma": (0.3, 0.6),            # J/m^2; around the Li surface energy (~0.5, Chen et al. 2015)
    "width_elements": (5.0, 10.0),  # 10-90 % interface width, in elements
}

BV_TO_AC_RATIO = 0.3              # r
DH_MAX = 1.875                    # max of h'(eta) = 30*eta^2*(1-eta)^2

NUM_CASES = 1000
SEED = 2025


def read_constants(template_text: str) -> dict:
    """name -> value of the numeric GenericConstantMaterial entries in the template"""
    consts = {}
    for names, vals in re.findall(r"prop_names\s*=\s*'([^']*)'\s*\n(?:\s*#.*\n)*\s*prop_values\s*=\s*'([^']*)'",
                                  template_text):
        for n, v in zip(names.split(), vals.split()):
            try:
                consts[n] = float(v)
            except ValueError:
                pass                      # placeholder
    return consts


def interface_mobility(pot_left: float, W: float, consts: dict) -> float:
    """L1o (m^3/(J s)) giving BV/AC rate ratio BV_TO_AC_RATIO at the applied potential"""
    bv_rate = consts["L2o"] * math.exp(consts["alpha"] * consts["Faraday"] * abs(pot_left) / consts["RT"])
    return bv_rate * DH_MAX / (BV_TO_AC_RATIO * W)


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
    values["L1o"] = interface_mobility(values["POT_LEFT"], values["W"], read_constants(template_text))

    # seed of the frozen reaction-rate field xi: a new pattern for every case
    values["seed"] = rng.randrange(1, 2**31 - 1)

    new_text = template_text.replace("$SEED", str(values["seed"]))
    for name in ("POT_LEFT", "Noise", "kappa", "W", "L1o"):
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
