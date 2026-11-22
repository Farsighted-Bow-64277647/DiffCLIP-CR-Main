import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logdir = r"C:\Marcus\G10-Diffusion\DiffCR-main (1)\DiffCLIP-CR-BS8-H100\TB-LOGGER"
outdir = r"C:\Marcus\G10-Diffusion\DiffCR-main (1)\DiffCLIP-CR-BS8-H100\TB-LOGGER"
tags_to_export = ["train/loss", "val/loss"]

ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
ea.Reload()

available = ea.Tags().get("scalars", [])
print("Available scalar tags (first 50):", available[:50])
missing = [t for t in tags_to_export if t not in available]
if missing:
    print("Missing tags:", missing)
    print("Tip: tags are case-sensitive; check available tags above.")
    
for tag in tags_to_export:
    if tag not in available:
        continue
    events = ea.Scalars(tag)
    df = pd.DataFrame([{"wall_time": e.wall_time, "step": e.step, "value": e.value} for e in events])
    outpath = os.path.join(outdir, tag.replace("/", "_") + ".csv")
    df.to_csv(outpath, index=False)
    print(f"Wrote {len(df)} rows -> {outpath}")
PY
