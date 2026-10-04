#!/usr/bin/env python3
"""
Process MOOSE Exodus (.e, .e-s###) :
 - Read all variables across all time steps
 - Save smooth PNG plots
 - Interpolate variables onto a uniform grid and save as (H, W, n) npy arrays
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import netCDF4
import os
import re
from scipy.interpolate import griddata


# ============================================================
# Exodus utilities
# ============================================================
def read_exodus_netcdf(filename):
    """Safely open an Exodus/NetCDF file."""
    try:
        return netCDF4.Dataset(filename, 'r')
    except Exception as e:
        print(f"[ERROR] Cannot read {filename}\n        {e}")
        return None

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
            conn = conn - 1  # convert 1-based → 0-based
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
def plot_variable_smooth(x, y, connectivity, nodal_values, var_name, time, output_dir):
    """Smooth field plot using triangular interpolation (tripcolor)."""
    if len(nodal_values) != len(x):
        raise ValueError(f"nodal_values ({len(nodal_values)}) != len(x) ({len(x)})")

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

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"{var_name}_t{time:.6f}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"  [IMG] {fname}")
    return fname

def extract_s_number(path):
    m = re.search(r'-s(\d+)$', path)
    if m:
        return int(m.group(1))
    else:
        return -1   # 没有 -s 后缀的排最前（也可放成很大数字）

# ============================================================
# Main export
# ============================================================
def export_all_variables(base_exodus_file, output_root, variables=("eta", "c", "pot"), sample_interval=1, grid_size=256, save_images=False):
    """
    Process all .e/.e-s### files, plot variables, and export .npy arrays.
    One .npy file per time step: shape = (H, W, n_variables).
    """
    base_dir = os.path.dirname(base_exodus_file)
    prefix = os.path.splitext(os.path.basename(base_exodus_file))[0]

    # --- locate all related Exodus files
    all_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir)
                 if re.match(fr"{re.escape(prefix)}(\.e|\.e-s\d+)$", f)]
    all_files.sort(key=extract_s_number)
    if not all_files:
        print(f"[ERROR] No Exodus files found for prefix {prefix}")
        return

    # --- output directories
    if save_images:
        img_dir = os.path.join(output_root, "image_files")
        os.makedirs(img_dir, exist_ok=True)
    npy_dir = os.path.join(output_root, "npy_files")
    os.makedirs(npy_dir, exist_ok=True)

    print(f"[INFO] Found {len(all_files)} Exodus files to process.")

    global_time_list = []

    total_ts_index = -1
    for file_idx, filename in enumerate(all_files):

        print(f"\n=== File {file_idx + 1}/{len(all_files)}: {os.path.basename(filename)} ===")
        ds = read_exodus_netcdf(filename)
        if ds is None:
            continue

        try:
            # Always read mesh (some files may differ)
            x, y, connectivity = get_mesh_data(ds)
            print(f"[INFO] Mesh: {len(x)} nodes, {len(connectivity)} elements")

            # Time info
            if 'time_whole' in ds.variables:
                times = np.array(ds.variables['time_whole'][:])
            else:
                times = [0.0]

            # Loop over time steps
            for t_idx, t_val in enumerate(times):

                total_ts_index += 1
                if total_ts_index % sample_interval != 0:
                    continue

                print(f"\n[STEP] t = {t_val:.6e}")
                var_data_list = []
                for vname in variables:
                    nodal_values = get_nodal_variable_data(ds, vname, t_idx)
                    if nodal_values is None:
                        print(f"  [WARN] Could not read variable {vname}")
                        continue

                    # skip mismatched node count
                    if len(nodal_values) != len(x):
                        print(f"  [WARN] Skipping {vname}: node count mismatch ({len(nodal_values)} vs {len(x)})")
                        continue

                    # Plot
                    if save_images:
                        plot_variable_smooth(x, y, connectivity, nodal_values, vname, t_val, img_dir)
                    var_data_list.append(nodal_values)

                if not var_data_list:
                    continue

                # --- Interpolate all variables to uniform grid
                xi = np.linspace(x.min(), x.max(), grid_size)
                yi = np.linspace(y.min(), y.max(), grid_size)
                X, Y = np.meshgrid(xi, yi)

                stacked_vars = []
                for vals in var_data_list:
                    Zi = griddata((x, y), vals, (X, Y), method='linear', fill_value=np.nan)
                    stacked_vars.append(Zi)

                stacked_array = np.stack(stacked_vars, axis=-1)
                npy_name = os.path.join(npy_dir, f"{t_val:.6f}.npy")
                np.save(npy_name, stacked_array)
                global_time_list.append(t_val)
                print(f"  [NPY] Saved {npy_name} shape={stacked_array.shape}")

        finally:
            ds.close()

    print(f"\n[INFO] Completed. Processed {len(global_time_list)} time steps.")
    if save_images:
        print(f"[INFO] PNGs → {img_dir}")
    print(f"[INFO] NPYs → {npy_dir}")


# ============================================================
# Entry point
# ============================================================
if __name__ == '__main__':
    import glob

    grid_size = 256
    save_images = False

    data_root = "data/"
    # for vn in os.listdir(data_root):
    for vn in ["case_001"]:
        main_file = glob.glob(os.path.join(data_root, vn, "exodus_files", "*.e"))[0]
        export_all_variables(
            base_exodus_file=main_file,
            output_root=os.path.join(data_root, vn),
            sample_interval = 1,
            grid_size=grid_size,
            save_images = save_images
        )