import subprocess, types, numpy as np
import npc_stim.running as new
old_src = subprocess.run(["git","show","HEAD:src/npc_stim/running.py"],capture_output=True,text=True,check=True).stdout
old = types.ModuleType("o"); exec(compile(old_src,"o","exec"), old.__dict__)
for u in ["s3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5",
          "s3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/OptoTagging_668755_20230831_142538.hdf5"]:
    o, n = old.get_running_speed_from_hdf5(u), new.get_running_speed_from_hdf5(u)
    both = ~np.isnan(o) & ~np.isnan(n)
    print(u.split("/")[-1], "max|old-new| where both valid:", np.max(np.abs(o[both]-n[both])),
          "| newly NaN idx:", np.flatnonzero(np.isnan(n) & ~np.isnan(o)).tolist(),
          "| old values there:", np.round(o[np.isnan(n) & ~np.isnan(o)],1).tolist())
