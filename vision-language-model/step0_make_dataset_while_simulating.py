#!/usr/bin/env python3
"""Batch export local, inactive MOOSE Exodus results to JPG only.

Dependencies: numpy matplotlib netCDF4. Run on the simulation host.
Existing destination directories are skipped, regardless of their contents.
Process absence is not proof of successful simulation convergence.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

# Loaded only for export, allowing --help and process checks without plotting deps.
np = plt = tri = netCDF4 = None


def load_plotting():
    global np, plt, tri, netCDF4
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import netCDF4


def get_variable_names(dataset):
    """Return a list of nodal variable names."""
    var_names = []
    if 'name_nod_var' in dataset.variables:
        name_var = dataset.variables['name_nod_var']
        for i in range(name_var.shape[0]):
            name_bytes = name_var[i, :].tobytes()
            name = name_bytes.decode('utf-8').strip('\x00').strip()
            if name:
                var_names.append(name)
    return var_names


def get_mesh_data(dataset):
    """Extract node coordinates and element connectivity."""
    x = np.array(dataset.variables['coordx'][:])
    y = np.array(dataset.variables['coordy'][:])

    connectivity_list = []
    for var_name in dataset.variables.keys():
        if var_name.startswith('connect'):
            conn = np.array(dataset.variables[var_name][:])
            conn = conn - 1  # convert 1-based -> 0-based
            connectivity_list.append(conn)

    if connectivity_list:
        connectivity = np.vstack(connectivity_list)
    else:
        raise ValueError("No cell connectivity found.")

    return x, y, connectivity


def get_nodal_variable_data(dataset, var_name, time_step):
    """Read nodal variable at a given time step."""
    var_names = get_variable_names(dataset)
    if var_name not in var_names:
        return None

    var_index = var_names.index(var_name) + 1
    var_key = f'vals_nod_var{var_index}'
    if var_key not in dataset.variables:
        return None

    data = dataset.variables[var_key]
    if time_step < data.shape[0]:
        return np.array(data[time_step, :])
    return None


# ============================================================
# Plotting
# ============================================================
def plot_variable_smooth(
    x, y, connectivity, nodal_values, var_name, time, output_dir, frame_index
):
    """Smooth field plot using triangular interpolation (tripcolor)."""
    if len(nodal_values) != len(x):
        raise ValueError(
            f"nodal_values ({len(nodal_values)}) != len(x) ({len(x)})"
        )

    # Convert to triangles if needed
    if connectivity.shape[1] > 3:
        tris = []
        for elem in connectivity:
            valid = elem[elem >= 0]
            if len(valid) < 3:
                continue
            for i in range(1, len(valid) - 1):
                tris.append([valid[0], valid[i], valid[i + 1]])
        tris = np.array(tris)
    else:
        tris = connectivity

    triang = tri.Triangulation(x, y, tris)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.tripcolor(triang, nodal_values, shading='gouraud', cmap='jet')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_frame_on(False)

    # Save images by variable.
    variable_img_dir = os.path.join(output_dir, var_name)
    os.makedirs(variable_img_dir, exist_ok=True)
    fname = os.path.join(
        variable_img_dir, f"{var_name}_{frame_index:07d}_t{time:.9g}.jpg"
    )
    try:
        fig.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0)
    finally:
        plt.close(fig)

    print(f"  [IMG] {fname}")
    return fname



def discover_cases(data_root):
    """One destination per immediate subdirectory, or root-level Exodus family."""
    cases = {}
    for path in sorted(data_root.rglob('*')):
        if not path.is_file() or not re.search(r'\.e(?:-s\d+)?$', path.name):
            continue
        relative = path.relative_to(data_root)
        key = relative.parts[0] if len(relative.parts) > 1 else re.sub(
            r'\.e(?:-s\d+)?$', '', path.name)
        cases.setdefault(key, []).append(path)
    return cases


def process_snapshot(pattern=None):
    """ps failure must never be treated as an empty process list."""
    result = subprocess.run(
        ['ps', '-ww', '-eo', 'pid=,stat=,args='],
        check=True, capture_output=True, text=True,
    )
    processes = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, state, command = fields
        if int(pid) == os.getpid() or state.startswith('Z'):
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        # Standard MOOSE executables, or a process with a MOOSE-style input.
        is_moose = any(re.search(r'-(?:opt|dbg|devel|oprof)$', t) for t in tokens)
        is_moose |= any(t == '-i' and i + 1 < len(tokens)
                        and tokens[i + 1].endswith('.i')
                        for i, t in enumerate(tokens))
        if pattern is not None:
            is_moose = bool(pattern.search(command))
        if not is_moose:
            continue
        try:
            cwd = Path(f'/proc/{pid}/cwd').resolve(strict=True)
        except OSError:
            cwd = None
        processes.append((pid, command, tokens, cwd))
    return processes


def matching_cases(process, cases, data_root):
    _, _, tokens, cwd = process
    matched = set()
    for key, files in cases.items():
        case_dir = data_root / key
        if case_dir.is_dir() and cwd is not None and (
                cwd == case_dir or case_dir in cwd.parents):
            matched.add(key)
        # Exact path components/stems avoid case1 matching case10.
        aliases = {key}
        aliases.update(re.sub(r'\.e(?:-s\d+)?$', '', f.name) for f in files)
        for token in tokens:
            value = token.split('=', 1)[-1]
            if value.startswith('-'):
                continue
            p = Path(value)
            if p.suffix == '.i' and p.stem in aliases:
                matched.add(key)
            if '=' in token and token.split('=', 1)[0].endswith('file_base'):
                if p.name in aliases:
                    matched.add(key)
            if p.is_absolute() or '/' in value:
                full = p if p.is_absolute() else (cwd / p if cwd else None)
                if full is not None:
                    full = full.resolve()
                    if full in files or (case_dir.is_dir() and
                            (full == case_dir or case_dir in full.parents)):
                        matched.add(key)
    return matched


def running_cases(cases, data_root, pattern):
    active, unknown = set(), []
    for process in process_snapshot(pattern):
        # This batch uses generated_inputs/case_N.i -> results/case_N/.
        # Recognize cases even before their first Exodus file exists.
        tokens = process[2]
        input_cases = set()
        for index, token in enumerate(tokens):
            value = None
            if token == '-i' and index + 1 < len(tokens):
                value = tokens[index + 1]
            elif token.startswith('-i='):
                value = token[3:]
            if value is not None:
                input_path = Path(value)
                if input_path.suffix == '.i' and re.fullmatch(r'case_\d+', input_path.stem):
                    input_cases.add(input_path.stem)
        if input_cases:
            active.update(input_cases)
            continue
        matches = matching_cases(process, cases, data_root)
        if matches:
            active.update(matches)
        else:
            unknown.append(process[0])
    return active, unknown


def file_signature(files):
    return {str(f): (f.stat().st_size, f.stat().st_mtime_ns) for f in files}


def export_all_variables(files, output_root, variables, sample_interval, end_time):
    families = {}
    for path in files:
        family = re.sub(r'-s\d+$', '', str(path))
        families.setdefault(family, []).append(path)
    count = 0
    for family, members in sorted(families.items()):
        members.sort(key=lambda p: int(p.name.rsplit('-s', 1)[1])
                     if re.search(r'-s\d+$', p.name) else -1)
        # Separate families so equal variable names and times cannot overwrite.
        relative = Path(family).relative_to(os.path.commonpath(
            [str(p.parent) for p in files]))
        image_dir = output_root / 'image_files'
        if len(families) > 1:
            image_dir = image_dir / relative.with_suffix('')
        frame = -1
        for filename in members:
            with netCDF4.Dataset(filename, 'r') as ds:
                x, y, connectivity = get_mesh_data(ds)
                if 'time_whole' not in ds.variables:
                    raise ValueError(f'Missing time_whole: {filename}')
                times = np.asarray(ds.variables['time_whole'][:])
                if not len(times):
                    raise ValueError(f'No completed frames: {filename}')
                for t_idx, t_val in enumerate(times):
                    frame += 1
                    if end_time >= 0 and frame > end_time:
                        break
                    if frame % sample_interval:
                        continue
                    for name in variables:
                        values = get_nodal_variable_data(ds, name, t_idx)
                        if values is None:
                            raise ValueError(f'Missing {name}, frame {t_idx}: {filename}')
                        plot_variable_smooth(x, y, connectivity, values, name,
                                             t_val, image_dir, frame)
                        count += 1
            if end_time >= 0 and frame >= end_time:
                break
    if not count:
        raise ValueError('No JPG files generated')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path('../phase-field-model/MooseProject/results'))
    parser.add_argument('--save-root', type=Path, default=Path('data'))
    parser.add_argument('--variables', nargs='+', default=['eta', 'c', 'pot'])
    parser.add_argument('--sample-interval', type=int, default=1)
    parser.add_argument('--end-time', type=int, default=-1,
                        help='Inclusive zero-based frame index per Exodus family')
    parser.add_argument('--process-regex',
                        help='Override MOOSE process detection with a command-line regex')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.sample_interval < 1 or args.end_time < -1:
        parser.error('sample-interval must be >= 1; end-time must be >= -1')
    data_root, save_root = args.data_root.resolve(), args.save_root.resolve()
    if not data_root.is_dir():
        parser.error(f'Input directory does not exist: {data_root}')
    if data_root == save_root or data_root in save_root.parents or save_root in data_root.parents:
        parser.error('Input and output directories must be separate, non-nested directories')
    try:
        pattern = re.compile(args.process_regex) if args.process_regex else None
    except re.error as exc:
        parser.error(str(exc))
    cases = discover_cases(data_root)
    if not cases:
        print('[INFO] No Exodus files found')
        return 0
    failures = 0
    for key, files in sorted(cases.items()):
        destination = save_root / key
        if destination.exists():
            print(f'[SKIP existing] {destination}')
            continue
        temporary = None
        try:
            active, _ = running_cases(cases, data_root, pattern)
            if key in active:
                print(f'[SKIP running] {key}')
                continue
            if args.dry_run:
                print(f'[WOULD EXPORT] {key}: {len(files)} Exodus files')
                continue
            if netCDF4 is None:
                load_plotting()
            before = file_signature(files)
            save_root.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix='.jpg-tmp-', dir=save_root))
            count = export_all_variables(files, temporary, args.variables,
                                         args.sample_interval, args.end_time)
            # Reject results if output changed or the simulation restarted.
            fresh = discover_cases(data_root)
            active, _ = running_cases(fresh, data_root, pattern)
            if (key in active or
                    before != file_signature(fresh.get(key, []))):
                raise RuntimeError('Simulation/output changed during extraction; retry later')
            if destination.exists():
                print(f'[SKIP appeared during export] {destination}')
                continue
            temporary.rename(destination)
            temporary = None
            print(f'[DONE] {key}: {count} JPGs -> {destination}')
        except Exception as exc:
            failures += 1
            print(f'[ERROR] {key}: {exc}')
        finally:
            if temporary is not None:
                shutil.rmtree(temporary)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
