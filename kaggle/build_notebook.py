"""Bundle the pipeline into one Kaggle notebook, so it runs where the competition data already lives.

    python kaggle/build_notebook.py      # writes kaggle/traffic_flow_pipeline.ipynb

On Kaggle: Create > New Notebook > File > Import Notebook, add the competition as input data, Run All.
The upload file lands in /kaggle/working/outputs/submission.csv.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["src/common.py", "src/task1_state.py", "src/task2_queue.py", "src/task4_odme.py", "src/pipeline.py",
         "official/src/merge_submissions.py", "official/config/corridors.json"]


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


cells = [cell("markdown", "# TrafficFlowBench pipeline\n\nWrites the pipeline files, then runs Tasks 1, 2 and 4 "
              "on the attached competition data and builds `outputs/submission.csv`. Task 3 is scored on the Task 1 rows.")]
for f in FILES:
    cells.append(cell("code", f"%%writefile /kaggle/working/tfb/{f}\n" + (ROOT / f).read_text()))
cells.insert(1, cell("code", "import os\nfor d in ('src', 'official/src', 'official/config'):\n"
                     "    os.makedirs(f'/kaggle/working/tfb/{d}', exist_ok=True)"))
cells.append(cell("code", "import glob, os\n"
                  "rel = sorted(glob.glob('/kaggle/input/**/corridors', recursive=True))[0].rsplit('/corridors', 1)[0]\n"
                  "os.environ['TFB_CACHE'] = '/kaggle/temp/cache'\nprint(rel)"))
cells.append(cell("code", "!cd /kaggle/working/tfb && TFB_CACHE=/kaggle/temp/cache python src/pipeline.py "
                  "--release-root \"{rel}\" --output-dir /kaggle/working/outputs"))
cells.append(cell("code", "import pandas as pd\nprint(pd.read_json('/kaggle/working/outputs/report.json', typ='series')['summary'])"))
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "kaggle" / "traffic_flow_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
