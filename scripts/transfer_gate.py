"""Screen paired development retrieval and language outcomes."""

import json
import sys
from pathlib import Path
from statistics import mean


def main():
    directory, baseline, candidate = sys.argv[1:]
    root = Path(directory)

    def read(model, metric):
        return [json.loads(line) for line in (root / f"{model}-{metric}.jsonl").read_text().splitlines()]

    def metrics(model):
        rows = read(model, "16k")
        present = [row for row in rows if not row["absent"]]
        absent = [row for row in rows if row["absent"]]
        return {
            "present": mean(row["correct"] for row in present),
            "absent": mean(row["correct"] for row in absent),
            "nll": mean(row["nll"] for row in read(model, "nll")),
        }

    stock, converted = metrics(baseline), metrics(candidate)
    print(json.dumps({baseline: stock, candidate: converted}), flush=True)
    return int(
        stock["present"] - converted["present"] > 0.05
        or stock["absent"] - converted["absent"] > 0.05
        or converted["nll"] - stock["nll"] > 0.10
    )


if __name__ == "__main__":
    sys.exit(main())
