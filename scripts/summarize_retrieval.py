"""Summarize present-key retrieval and absent-key false answers by distance."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def wilson(successes, count):
    proportion = successes / count
    z = 1.96
    scale = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / scale
    radius = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count**2)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(rows):
    present = [row for row in rows if not row["absent"]]
    absent = [row for row in rows if row["absent"]]
    out = {"present_count": len(present), "absent_count": len(absent)}
    if present:
        correct = sum(row["correct"] for row in present)
        out["retrieval_accuracy"] = correct / len(present)
        out["retrieval_95pct_wilson"] = wilson(correct, len(present))
    if absent:
        out["absent_accuracy"] = sum(row["correct"] for row in absent) / len(absent)
        out["absent_false_answer_rate"] = (
            sum(row["prediction"] not in (None, "ABSENT") for row in absent) / len(absent)
        )
    return out


def main(paths):
    for path in paths:
        by_distance = defaultdict(list)
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                by_distance[row["length"], row["distance"]].append(row)
        print(json.dumps({
            "source": path,
            "by_distance": [
                {"length": length, "distance": distance, **summarize(rows)}
                for (length, distance), rows in sorted(by_distance.items())
            ],
        }))


if __name__ == "__main__":
    main(sys.argv[1:])
