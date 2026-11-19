# gh #2 / #4: rotary encoder glitches — raw data check (2026-11-18)

Scripts (run with `uv run python <script> ...`, S3 read access to `aind-ephys-data` required):
- `probe_raw.py <bucket/session>` → `raw_<session>.json`: dtype, counts/rev, index-diff histogram, raw count/index around |speed| > 150 cm/s
- `nan_rows.py <hdf5 url>`: count/index around NaN reads
- `before_after.py <session dirs>` → `before_after.json`: HEAD vs working-tree `get_running_speed_from_hdf5`
- `diffcheck.py`: max |old − new| where both valid; values that became NaN

Sessions checked: 668755_2023-08-31, 660023_2023-08-09 (2 dirs), 662892_2023-08-22 (3 of the 7
extreme sessions in the issue). 714753_2024-07-02, 737403_2024-09-26, 742903_2024-10-24 were not
found under `s3://aind-ephys-data/ecephys_*`, so they were **not checked**.

## Observations
- Wheel: 8192 counts/rev, radius 4.69 cm → 0.216 cm/s per count per frame.
- Task (DynamicRouting1), RFMapping, Spontaneous files: int32 count/index, index diff always 1, max |speed| 51–73 cm/s.
- OptoTagging files: float64 count/index, 3–15 NaN reads per file, 1–4 index diffs of 3 per file.
- Every index skip (Δ=3) has a permanent count step of ~2.5k–38k counts (543–6924 cm/s), so it can't be accumulated motion.
- The read **after** a skip is usually elevated too (91–741 cm/s against ~30 cm/s neighbours), so masking only the skip sample is not enough.
- The 1e11–1e12 cm/s values are single garbage count reads (~1.3e13 counts) just before a skip. They are **not** the read-skip steps themselves (the issue attributed everything to skips).
- NaN count reads also have NaN index, with a count step across the gap.

## Implemented masking (`get_running_speed_from_encoder_count`)
NaN where: count is NaN; index diff ≠ 1 (incl. NaN); the sample after any such skip; |speed| > 1.3 × 334 cm/s
(fastest recorded house mouse, Garland et al. 1995, https://doi.org/10.1016/0031-9384(95)00148-c).

## Result (`before_after.json`)
| file type | max abs speed before (cm/s) | after (cm/s) |
|---|---|---|
| OptoTagging (6 files) | 543 – 5.0e12 | 40.8 – 66.9 |
| task / RF / spontaneous (9 files) | 51.6 – 72.5 | unchanged |

Valid samples differ from HEAD by ≤ 2.4e-9 cm/s (float reordering). Cost: the extra masked samples
include about 1 plausible-looking read after each NaN gap (e.g. 61.1, 54.4, 39.9 cm/s); this is deliberate.
