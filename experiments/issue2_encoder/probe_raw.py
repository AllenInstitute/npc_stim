"""Inspect raw rotary encoder data around extreme running-speed samples (gh #2)."""
import sys, json
import numpy as np, h5py, s3fs

fs = s3fs.S3FileSystem()
root = sys.argv[1]
out = {}
for p in sorted(fs.glob(f"{root}/behavior/*.hdf5")):
    with fs.open(p, "rb") as f, h5py.File(f, "r") as d:
        if "rotaryEncoderCount" not in d:
            continue
        name = p.split("/")[-1]
        cnt = d["rotaryEncoderCount"][:]
        idx = d["rotaryEncoderIndex"][:] if "rotaryEncoderIndex" in d else None
        cpr = d["rotaryEncoderCountsPerRev"][()]
        r = d["wheelRadius"][()]
        enc = d["rotaryEncoder"][()]
        scale = 2*np.pi*r*60/cpr
        speed = np.diff(cnt.astype(float))*scale
        bad = np.flatnonzero(np.abs(speed) > 150)
        rec = dict(dtype=str(cnt.dtype), n=len(cnt), cpr=float(cpr), radius_cm=float(r), encoder=str(enc),
                   has_index=idx is not None, cm_s_per_count=float(scale),
                   n_nan=int(np.isnan(cnt.astype(float)).sum()), max_abs_speed=float(np.nanmax(np.abs(speed))) if speed.size else None,
                   n_over_150=len(bad))
        if idx is not None:
            di = np.diff(idx.astype(float))
            vals, cts = np.unique(di[~np.isnan(di)], return_counts=True)
            rec["index_dtype"] = str(idx.dtype)
            rec["index_diff_hist"] = {str(v): int(c) for v, c in zip(vals, cts)}
        ev = []
        for b in bad[:12]:
            s = slice(max(b-2, 0), b+4)
            ev.append(dict(i=int(b), count=cnt[s].tolist(),
                           index=None if idx is None else idx[s].tolist(),
                           speed=np.round(speed[max(b-2,0):b+3], 1).tolist()))
        rec["events"] = ev
        out[name] = rec
print(json.dumps(out, indent=1, default=str))
