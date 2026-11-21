
import csv
import json
import os


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_header = not os.path.exists(path)
        self._f = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=fieldnames)
        if write_header:
            self._writer.writeheader()
            self._f.flush()

    def log(self, row):
        self._writer.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()


def append_json_row(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = json.load(f)
    rows.append(row)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
