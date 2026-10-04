#!/usr/bin/env python3
"""Prepare all cases and plot numerical trends and parameter associations."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from analyze_results import ROOT, plt, prepare, save_json


def interpolate(records, key, time):
    times = np.array([r['time'] for r in records])
    values = np.array([r[key] if r[key] is not None else np.nan for r in records])
    if time < times[0] or time > times[-1]:
        raise ValueError('Comparison time is outside the exported range')
    return float(np.interp(time, times, values))


def summarize(cases, time):
    rows = []
    for case in cases:
        records = case['records']
        start = records[0]
        tip = interpolate(records, 'tip_x', time)
        displacement = tip - start['tip_x'] if start['tip_x'] is not None else np.nan
        duration = time - start['time']
        values = dict(tip_displacement=displacement,
                      mean_tip_speed=displacement / duration if duration > 0 else np.nan,
                      front_std_x=interpolate(records, 'front_std_x', time),
                      attached_grid_fraction=interpolate(records, 'attached_grid_fraction', time))
        rows.append(dict(case_id=case['id'], comparison_time=time,
                         **{k: float(v) if np.isfinite(v) else None for k, v in values.items()},
                         parameters=case['parameters']))
    return rows


def export(fig, out, name):
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'{name}.{suffix}', dpi=180)
    plt.close(fig)


def plot_trends(cases, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specs = [('tip_x', 'Tip displacement (coordinate units)'),
             ('tip_speed', 'Tip speed (coordinate / time)'),
             ('front_std_x', 'Front roughness (coordinate units)'),
             ('attached_grid_fraction', 'Attached phase fraction')]
    for ax, (key, title) in zip(axes.flat, specs):
        for case in cases:
            records = case['records']
            values = np.array([r[key] if r[key] is not None else np.nan for r in records])
            if key == 'tip_x':
                values = values - values[0]
            ax.plot([r['time'] for r in records], values, label=case['id'],
                    alpha=.8 if len(cases) <= 12 else .2, linewidth=1.5)
        ax.set(xlabel='Simulation time', ylabel=title)
        ax.grid(alpha=.2)
    if len(cases) <= 12:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Growth trajectories | {len(cases)} case(s) | exported frames only')
    export(fig, out, 'growth_trends')


def plot_comparison(rows, out, time):
    valid = [r for r in rows if r['tip_displacement'] is not None]
    ranked = sorted(valid, key=lambda r: r['tip_displacement'], reverse=True)[:20]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, .3 * len(ranked) + 2)), constrained_layout=True)
    if ranked:
        axes[0].barh([r['case_id'] for r in ranked], [r['tip_displacement'] for r in ranked], color='#2878a0')
        axes[0].invert_yaxis()
        xs = [r['tip_displacement'] for r in valid]
        ys = [r['front_std_x'] if r['front_std_x'] is not None else np.nan for r in valid]
        axes[1].scatter(xs, ys, s=45, color='#b85731')
        if len(valid) <= 12:
            for r, x, y in zip(valid, xs, ys):
                if np.isfinite(y):
                    axes[1].annotate(r['case_id'], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    else:
        axes[0].text(.5, .5, 'No valid front measurements', ha='center', transform=axes[0].transAxes)
    axes[0].set(xlabel='Tip displacement (coordinate units)', title='Largest displacements (up to 20)')
    axes[1].set(xlabel='Tip displacement (coordinate units)', ylabel='Front roughness (coordinate units)',
                title='Growth and interface roughness')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f'Common-time comparison | t = {time:.6g} | linear interpolation')
    export(fig, out, 'case_comparison')


def plot_quality(cases, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for case in cases:
        records = case['records']
        times = [r['time'] for r in records]
        total = int(np.prod(case['metadata']['shape'][:2]))
        axes[0].plot(times, [r['fields']['c']['min'] for r in records], label=case['id'])
        axes[1].plot(times, [r['fields']['c']['negative_count'] / total for r in records])
        axes[2].plot(times, [sum(r['fields'][v]['nonfinite_count'] for v in ('eta', 'c', 'pot')) / (3 * total)
                             for r in records])
    for ax, title in zip(axes, ['Minimum c', 'Negative c sample fraction', 'Nonfinite field sample fraction']):
        ax.set(xlabel='Simulation time', ylabel=title)
        ax.grid(alpha=.2)
    if len(cases) <= 12:
        axes[0].legend(fontsize=8)
    fig.suptitle('Data quality | diagnostic flags, not physical conclusions')
    export(fig, out, 'data_quality')


def associations(rows):
    names = sorted({k for r in rows for k in r['parameters']})
    outcomes = ['tip_displacement', 'front_std_x', 'attached_grid_fraction']
    result = []
    for name in names:
        for outcome in outcomes:
            pairs = [(r['parameters'].get(name), r[outcome]) for r in rows]
            pairs = [(x, y) for x, y in pairs if isinstance(x, (int, float)) and y is not None
                     and np.isfinite(x) and np.isfinite(y)]
            coefficient = None
            if len(pairs) >= 5:
                x, y = np.array(pairs).T
                if np.ptp(x) > 0 and np.ptp(y) > 0:
                    coefficient = float(np.corrcoef(x, y)[0, 1])
            result.append(dict(parameter=name, outcome=outcome, n=len(pairs), pearson_r=coefficient))
    return result


def plot_associations(rows, out):
    stats = associations(rows)
    save_json(out / 'parameter_associations.json', stats)
    names = sorted({r['parameter'] for r in stats if r['pearson_r'] is not None})
    outcomes = ['tip_displacement', 'front_std_x', 'attached_grid_fraction']
    fig, ax = plt.subplots(figsize=(8, max(4, .35 * len(names) + 2)), constrained_layout=True)
    if names:
        lookup = {(r['parameter'], r['outcome']): r['pearson_r'] for r in stats}
        matrix = np.array([[lookup[(name, outcome)] if lookup[(name, outcome)] is not None else np.nan
                            for outcome in outcomes] for name in names])
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(len(outcomes)))
        ax.set_xticklabels(['Tip displacement', 'Front roughness', 'Phase fraction'])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if np.isfinite(value):
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color='white' if abs(value) > .6 else 'black')
        fig.colorbar(im, ax=ax, label='Pearson r (descriptive)')
    else:
        ax.axis('off')
        ax.text(.5, .5, f'{len(rows)} case(s) available\nAt least 5 valid cases and varying values are required.\n'
                'No parameter association is estimated.', ha='center', va='center', transform=ax.transAxes)
    fig.suptitle('Sampled parameter associations | not causal effects')
    export(fig, out, 'parameter_associations')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--inputs', type=Path, default=ROOT.parent / 'phase-field-model/MooseProject/generated_inputs')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/summary')
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--time', type=float)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('threshold must be in (0, 1)')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases, errors = [], []
    for folder in sorted(args.data.glob('*/npy_files')):
        case = folder.parent
        try:
            prepare(SimpleNamespace(case=case, input=args.inputs / f'{case.name}.i',
                                    threshold=args.threshold, frames=3))
            analysis = case / 'analysis'
            parameters = args.inputs / f'{case.name}.json'
            cases.append(dict(id=case.name, records=json.loads((analysis / 'metrics.json').read_text()),
                              metadata=json.loads((analysis / 'metadata.json').read_text()),
                              parameters=json.loads(parameters.read_text()) if parameters.exists() else {}))
        except (ValueError, OSError, KeyError) as error:
            errors.append(dict(case_id=case.name, error=str(error)))
    save_json(out / 'errors.json', errors)
    if not cases:
        raise SystemExit('No valid cases; inspect errors.json')
    reference = cases[0]['metadata']
    if any(c['metadata']['bounds'] != reference['bounds'] for c in cases):
        raise SystemExit('Mesh bounds differ; compare compatible case groups separately')
    start = max(c['records'][0]['time'] for c in cases)
    end = min(c['records'][-1]['time'] for c in cases)
    time = args.time if args.time is not None else end
    if not np.isfinite(time) or not start <= time <= end:
        raise SystemExit('No shared time range or requested time is outside it')
    if len({c['records'][0]['time'] for c in cases}) != 1:
        raise SystemExit('Initial times differ; displacement requires a shared baseline')
    rows = summarize(cases, time)
    save_json(out / 'case_summary.json', rows)
    save_json(out / 'run.json', dict(cases=[c['id'] for c in cases], comparison_time=time,
              threshold=args.threshold, excluded_cases=errors,
              units='Simulation units; compare only cases with matching physical scales',
              parameters='Sampled JSON values; executed input values may be rounded',
              conclusions='Numerical plots; Qwen text is not used as measurement data',
              limitations=['Exported frames only; completion unknown',
                           'Common-time metrics use linear interpolation, never extrapolation',
                           'Correlations are exploratory, confounded, and not significance tests']))
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    plot_trends(cases, out)
    plot_comparison(rows, out, time)
    plot_quality(cases, out)
    plot_associations(rows, out)
    print(f'Saved PNG/SVG figures for {len(cases)} case(s): {out}')
    if errors:
        print(f'Excluded {len(errors)} case(s); inspect errors.json')


if __name__ == '__main__':
    main()
