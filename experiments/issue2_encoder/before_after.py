"""Compare old (git HEAD) vs new get_running_speed_from_hdf5 on real sessions (gh #2)."""
import sys, json, subprocess, types
import numpy as np, s3fs
import npc_stim.running as new

old_src = subprocess.run(["git", "show", "HEAD:src/npc_stim/running.py"], capture_output=True, text=True, check=True).stdout
old = types.ModuleType("old_running"); exec(compile(old_src, "old_running", "exec"), old.__dict__)

fs = s3fs.S3FileSystem()
rows = []
for root in sys.argv[1:]:
    for p in sorted(fs.glob(f"aind-ephys-data/{root}/behavior/*.hdf5")):
        url = "s3://" + p
        try:
            o = old.get_running_speed_from_hdf5(url)
        except ValueError:
            continue
        n = new.get_running_speed_from_hdf5(url)
        if o.size == 0:
            continue
        both = ~np.isnan(o) & ~np.isnan(n)
        rows.append(dict(file=p.split("/")[-1], n=len(o),
            old_nan=int(np.isnan(o).sum()), new_nan=int(np.isnan(n).sum()),
            old_max=float(np.nanmax(np.abs(o))), new_max=float(np.nanmax(np.abs(n))),
            new_p999=float(np.nanpercentile(np.abs(n), 99.9)),
            unchanged_where_valid=bool(np.array_equal(o[both], n[both])),
            old_filt_min_near_nan=float(np.min(old.lowpass_filter(o)[np.isnan(o)][1:])) if np.isnan(o)[1:].any() else None))
print(json.dumps(rows, indent=1))
