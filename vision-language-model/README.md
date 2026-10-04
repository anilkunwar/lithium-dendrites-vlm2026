# Dendrite analysis

The target output is a cross-case report: shared patterns, contrasts, parameter
associations, and anomalies supported by numerical evidence and figures.
Qwen synthesizes the ensemble; numerical code supplies the measurements.

```bash
python -m pip install -r requirements.txt
python plot_batch.py
python synthesize_results.py
python -m unittest discover -s tests
```

`plot_batch.py` discovers `data/*/npy_files`, prepares each case using its matching
`.i` file, and writes PNG/SVG figures to `data/summary`:

- `growth_trends`: displacement, speed, roughness, and phase fraction over time.
- `case_comparison`: displacement ranking and growth versus roughness.
- `data_quality`: concentration minima, negative values, and nonfinite values.
- `parameter_associations`: exploratory correlations with sampled parameters.

`case_summary.json` stores common-time measurements; `run.json` records the scope;
`errors.json` lists excluded cases. Correlations require at least five valid pairs
and varying values. They are descriptive, not causal or significance tests.
Parameter values come from sampled `.json` files; executed `.i` values may be rounded.

```bash
python plot_batch.py --data data --inputs ../phase-field-model/MooseProject/generated_inputs --time 200
```

Comparisons use the latest shared time by default, with linear interpolation and
no extrapolation. Cases need matching domains, initial times, and physical scales.
A large ensemble uses transparent trajectories; rankings show at most 20 cases.
Only one converted case is currently available, so population conclusions are pending.

## Cross-case synthesis

`synthesize_results.py` sends ensemble distributions, representative extremes,
parameter associations, and comparison figures to Qwen in one synthesis request.
All cases contribute to distributions and eligible correlations. Representative
rows are selected for context, not treated as the full dataset or as clusters.
The request contains aggregate evidence rather than every raw frame.

Outputs in `data/summary`:

- `report.html`: standalone illustrated report with expandable evidence.
- `synthesis.json`: cross-case findings, associations, anomalies, and follow-up checks.
- `synthesis_evidence.json`: traceable numerical evidence used by Qwen.
- `synthesis_status.json`: preparation, failure, or review status.
- `synthesis_request.json` and `synthesis_response.raw.json`: experiment records.

At least two converted cases are required. With one case, the script writes a
pending report without calling Qwen. Use `--prepare-only` to inspect evidence before
inference. The default model is `qwen3-vl:8b-instruct`; `--model` can override it.
Unknown evidence references are rejected. Accepted references do not guarantee
correct interpretation, so generated conclusions remain drafts for scientific review.
Regenerate plots and synthesis after adding results. Historical synthesis files
may remain after a failed run; always check the status and current HTML report.

## Input and metrics

Each `<time>.npy` must have shape `(H,W,3)` with channels `eta,c,pot` and increasing y
along the first axis. Legacy channel order is assumed from step0, not independently
verified. Mesh bounds come from the input file. Units remain simulation units.

The default phase threshold is `eta >= 0.5`. Four-neighbor components touching the
left boundary define the attached phase. Tip position is the largest attached x;
roughness is the standard deviation of row front positions, not interface thickness.
Speed uses actual time differences. Area fraction is a grid estimate, not FE integration.
Use `--threshold` to check sensitivity. Missing frames and rounded timestamps cannot
be recovered; interpolated fields may smooth small structures.

## Optional Qwen analysis

```bash
python analyze_results.py prepare --case data/case_001 --input ../phase-field-model/MooseProject/generated_inputs/case_001.i
python analyze_results.py analyze --case data/case_001 --model qwen3-vl:8b-instruct
```

Ollama must be running with the model installed. Prompts and reports use English.
Each case's `analysis` directory contains metrics, evidence images, request records,
and raw/validated JSON responses. Schema validation does not establish correctness.
Earlier Qwen trials contained factual errors; historical responses are retained as
experiment records and excluded from batch measurements. Repeated runs overwrite
matching output names; archive results when comparing experiments.

Dependencies cover analysis only; step0 also requires `netCDF4`.
