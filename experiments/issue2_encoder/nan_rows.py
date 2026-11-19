import sys, numpy as np, h5py, s3fs
fs = s3fs.S3FileSystem()
for p in sys.argv[1:]:
    with fs.open(p, "rb") as f, h5py.File(f, "r") as d:
        c = d["rotaryEncoderCount"][:]; i = d["rotaryEncoderIndex"][:]
        print(p.split("/")[-1], "keys with 'rotary':", [k for k in d if "otary" in k])
        for b in np.flatnonzero(np.isnan(c.astype(float))):
            s = slice(max(b-2,0), b+3); print("  nan@", b, c[s].tolist(), i[s].tolist())
