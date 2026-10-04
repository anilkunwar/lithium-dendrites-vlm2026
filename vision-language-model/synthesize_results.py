#!/usr/bin/env python3
"""Use Qwen to synthesize cross-case evidence into an illustrated report."""
import argparse
import base64
from html import escape
import json
from pathlib import Path
import urllib.request

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from analyze_results import ROOT, save_json, sha256

METRICS = ('tip_displacement', 'mean_tip_speed', 'front_std_x', 'attached_grid_fraction')
FIGURES = ('growth_trends', 'case_comparison', 'parameter_associations', 'data_quality')


class Finding(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str
    interpretation: str
    evidence_ids: list[str] = Field(min_length=1)
    limitations: str


class Synthesis(BaseModel):
    model_config = ConfigDict(extra='forbid')
    cross_case_patterns: list[Finding] = Field(min_length=1, max_length=5)
    parameter_associations: list[Finding] = Field(max_length=5)
    anomalies: list[Finding] = Field(max_length=5)
    follow_up_checks: list[str] = Field(max_length=5)


def build_evidence(rows, correlations, run):
    if len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate case IDs')
    evidence = {}
    representatives = set()
    for metric in METRICS:
        valid = sorted((r for r in rows if r[metric] is not None and np.isfinite(r[metric])),
                       key=lambda r: r[metric])
        values = [r[metric] for r in valid]
        if not values:
            continue
        low, high = valid[0], valid[-1]
        representatives.update((low['case_id'], high['case_id']))
        evidence['distribution:' + metric] = dict(
            metric=metric, n=len(values), missing=len(rows)-len(values),
            quantiles=dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                               np.quantile(values, [0, .25, .5, .75, 1]).tolist())),
            minimum_example=low['case_id'], maximum_example=high['case_id'])
    for row in rows:
        if row['case_id'] in representatives:
            evidence['case:' + row['case_id']] = row
    for item in correlations:
        if item['pearson_r'] is not None:
            evidence[f"association:{item['parameter']}:{item['outcome']}"] = item
    return dict(case_count=len(rows), case_ids=[r['case_id'] for r in rows],
                comparison_time=run['comparison_time'], units=run['units'],
                scope='Distributions use all cases; representative rows show metric extremes only.',
                limitations=run['limitations'] + [
                    'Extremes are not automatically outliers or optimal cases.',
                    'Associations use sampled parameters; concurrent changes can confound them.',
                    'No causal, significance, clustering, or physical-unit analysis is provided.'],
                evidence=evidence)


def validate_references(report, packet):
    known = set(packet['evidence'])
    for group in (report.cross_case_patterns, report.parameter_associations, report.anomalies):
        for finding in group:
            if not set(finding.evidence_ids) <= known:
                raise ValueError(f'Unknown evidence reference: {finding.evidence_ids}')


def render_report(out, packet, report=None):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<title>Cross-case dendrite analysis</title>',
             '<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 sans-serif;'
             'color:#172b3a}img{width:100%}pre{white-space:pre-wrap;font-size:12px;background:#f3f6f8;'
             'padding:16px}h2{margin-top:40px}.status{padding:16px;background:#fff3d6}</style>',
             '<h1>Cross-case dendrite analysis</h1>',
             f'<p>{packet["case_count"]} cases; comparison time: {packet["comparison_time"]:.6g}. '
             'All quantities use simulation units.</p>']
    if report is None:
        message = ('At least two converted cases are required.' if packet['case_count'] < 2
                   else 'Evidence is ready; no current Qwen synthesis is available. See synthesis_status.json.')
        parts.append(f'<p class="status">Cross-case synthesis pending. {message}</p>')
    else:
        parts.append('<p class="status">Qwen draft: evidence references were checked; scientific claims still require review.</p>')
        for field, title in [('cross_case_patterns', 'Patterns across cases'),
                             ('parameter_associations', 'Parameter associations'), ('anomalies', 'Anomalies')]:
            parts.append(f'<h2>{title}</h2>')
            findings = getattr(report, field)
            if not findings:
                parts.append('<p>No supported finding was returned.</p>')
            for finding in findings:
                parts.append(f'<h3>{escape(finding.title)}</h3><p>{escape(finding.interpretation)}</p>'
                             f'<p>Limitations: {escape(finding.limitations)}</p><details><summary>Evidence</summary>')
                for key in finding.evidence_ids:
                    parts.append(f'<b>{escape(key)}</b><pre>{escape(json.dumps(packet["evidence"][key], indent=2))}</pre>')
                parts.append('</details>')
        parts.append('<h2>Follow-up checks</h2><ul>')
        parts.extend(f'<li>{escape(item)}</li>' for item in report.follow_up_checks)
        parts.append('</ul>')
    for name in FIGURES:
        path = out / f'{name}.png'
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode()
            parts.append(f'<h2>{escape(name.replace("_", " ").title())}</h2>'
                         f'<img alt="{name}" src="data:image/png;base64,{encoded}">')
    parts.append('<h2>Scope and limitations</h2><ul>')
    parts.extend(f'<li>{escape(item)}</li>' for item in packet['limitations'])
    parts.append('</ul></html>')
    (out / 'report.html').write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT / 'data/summary')
    parser.add_argument('--model', default='qwen3-vl:8b-instruct')
    parser.add_argument('--url', default='http://127.0.0.1:11434')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.summary.resolve()
    rows = json.loads((out / 'case_summary.json').read_text())
    correlations = json.loads((out / 'parameter_associations.json').read_text())
    run = json.loads((out / 'run.json').read_text())
    if set(run['cases']) != {r['case_id'] for r in rows}:
        raise ValueError('Summary and run manifest disagree; rerun plot_batch.py')
    packet = build_evidence(rows, correlations, run)
    save_json(out / 'synthesis_evidence.json', packet)
    if len(rows) < 2 or args.prepare_only:
        render_report(out, packet)
        save_json(out / 'synthesis_status.json', dict(status='insufficient_cases' if len(rows) < 2 else 'prepared',
                                                    case_count=len(rows)))
        print(f'Evidence prepared for {len(rows)} case(s). Cross-case Qwen synthesis not run.')
        return
    prompt = '''Synthesize the entire simulation ensemble in English, not separate case reports.
Use the evidence packet and plots to identify shared patterns, contrasts, parameter
associations, and unresolved anomalies. Explain what the ensemble supports and what
it does not. Every finding must cite exact evidence IDs. Quantitative claims must
come from the packet, not visual estimates. Use at most three sentences per finding.
Do not equate correlation with causality, extrema with outliers, or roughness with
interface thickness. Empty lists are appropriate when evidence is insufficient.
Do not call a case optimal without a defined objective. Do not invent clusters,
significance, units, mechanisms, missing cases, or simulation completion.
Return JSON matching this schema:
'''
    prompt += json.dumps(Synthesis.model_json_schema()) + '\nEvidence:\n' + json.dumps(packet)
    images = [out / f'{name}.png' for name in ('case_comparison', 'parameter_associations')]
    payload = dict(model=args.model, stream=False, format=Synthesis.model_json_schema(),
                   options=dict(temperature=0, num_ctx=32768, num_predict=4096),
                   messages=[dict(role='user', content=prompt,
                                  images=[base64.b64encode(p.read_bytes()).decode() for p in images])])
    save_json(out / 'synthesis_request.json', dict(model=args.model, prompt=prompt, options=payload['options'],
              input_hashes={name: sha256(out / name) for name in ('case_summary.json', 'parameter_associations.json', 'run.json')},
              image_hashes={p.name: sha256(p) for p in images}))
    save_json(out / 'synthesis_status.json', dict(status='running', case_count=len(rows)))
    render_report(out, packet)
    request = urllib.request.Request(args.url.rstrip('/') + '/api/chat', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = json.load(response)
        save_json(out / 'synthesis_response.raw.json', raw)
        report = Synthesis.model_validate_json(raw['message']['content'])
        validate_references(report, packet)
        save_json(out / 'synthesis.json', report.model_dump())
        render_report(out, packet, report)
        save_json(out / 'synthesis_status.json', dict(status='needs_scientific_review', case_count=len(rows)))
    except Exception as error:
        save_json(out / 'synthesis_status.json', dict(status='failed', error=str(error)))
        raise
    print(f'Saved cross-case draft: {out / "report.html"}')


if __name__ == '__main__':
    main()
