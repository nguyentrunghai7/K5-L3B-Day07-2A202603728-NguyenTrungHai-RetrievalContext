"""Summarize recorded trace reviews using the LAB07 starter grader."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from agent_metrics import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory containing traces.jsonl and reviews.json")
    args = parser.parse_args()
    traces = [json.loads(line) for line in (args.run / "traces.jsonl").read_text(
        encoding="utf-8").splitlines()]
    reviews = json.loads((args.run / "reviews.json").read_text(encoding="utf-8"))
    cases = [{"id": trace["query_id"], "answerable": trace["answerable"]} for trace in traces]
    review_by_id = {row["query_id"]: row for row in reviews}
    # A model can refuse in prose while supplying an excerpt ID. Grade the
    # observed behavior while keeping the raw status and citations in the trace.
    graded_traces = [
        {**trace, "status": "abstain"}
        if trace["status"] == "answered"
        and review_by_id.get(trace["query_id"], {}).get("semantic_abstention")
        else trace
        for trace in traces
    ]
    summary = aggregate(cases, graded_traces, reviews)
    summary["context_grade_counts"] = dict(sorted(Counter(
        row["context_grade"] for row in reviews if "context_grade" in row
    ).items()))
    summary["search_count_distribution"] = dict(sorted(Counter(
        len(trace["tool_calls"]) for trace in traces
    ).items()))
    output = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (args.run / "agent-summary.json").write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
