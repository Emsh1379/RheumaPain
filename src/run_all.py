"""Run the whole RheumaPain pipeline (audit -> features -> experiments) and zip the outputs.

On Kaggle start it detached so it survives disconnects:
    nohup python /kaggle/working/rp/run_all.py > /kaggle/working/output/run.log 2>&1 &
"""
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.setdefault("RP_OUT", "/kaggle/working/output")
os.environ.setdefault("RP_DATA", "/kaggle/working")
STAGES = [
    ["s01_audit.py"], ["s01b_audit_detail.py"], ["s01c_batch_check.py"],
    ["s02_features.py", "10", "30", "60"],
    ["s03_experiments.py"],
]

for stage in STAGES:
    t0 = time.time()
    print(f"\n######## {' '.join(stage)}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, stage[0])] + stage[1:])
    print(f"######## {stage[0]} finished in {time.time() - t0:.0f}s, exit code {r.returncode}", flush=True)
    if r.returncode:
        sys.exit(r.returncode)

shutil.make_archive(os.path.join(os.path.dirname(OUT), "rheumapain_output"), "zip", OUT)
print("ALL DONE", flush=True)
