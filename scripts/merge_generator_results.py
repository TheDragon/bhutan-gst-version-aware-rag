from __future__ import annotations

import json

import pandas as pd

from gst_rag.evaluation import VERSION_SENSITIVE
from gst_rag.paths import RESULTS
from gst_rag.systems import SYSTEMS


def main() -> None:
    paths = sorted(RESULTS.glob("generator_predictions_*_[0-9][0-9][0-9]_[0-9][0-9][0-9].csv"))
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        raise FileNotFoundError("No batched generator prediction files found")
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["system", "question_id"]).any():
        raise ValueError("Duplicate system/question rows found in generator batches")
    if len(combined) != len(SYSTEMS) * 120:
        raise ValueError(f"Expected {len(SYSTEMS) * 120} rows, found {len(combined)}")
    metrics: dict[str, dict] = {}
    for system in SYSTEMS:
        selected = combined[(combined.system == system) & (combined.split == "test")]
        version_rows = selected[selected.category.isin(VERSION_SENSITIVE)]
        by_category: dict[str, dict] = {}
        for category, group in selected.groupby("category"):
            by_category[category] = {
                "questions": int(len(group)),
                "recall_at_5": float(group.recall_at_5.mean()),
                "mrr_at_10": float(group.reciprocal_rank_at_10.mean()),
                "version_selection_accuracy": float(group.version_selected.mean()),
                "citation_accuracy": float(group.citation_correct.mean()),
                "answer_token_f1_proxy": float(group.answer_token_f1_proxy.mean()),
            }
        metrics[system] = {
            "test_questions": int(len(selected)),
            "recall_at_5": float(selected.recall_at_5.mean()),
            "mrr_at_10": float(selected.reciprocal_rank_at_10.mean()),
            "version_sensitive_questions": int(len(version_rows)),
            "version_selection_accuracy": float(version_rows.version_selected.mean()),
            "citation_accuracy": float(selected.citation_correct.mean()),
            "answer_token_f1_proxy": float(selected.answer_token_f1_proxy.mean()),
            "answer_score_proxy_0_2": float(selected.answer_score_proxy_0_2.mean()),
            "abstention_rate": float(selected.abstained.astype(float).mean()),
            "by_category": by_category,
        }

    standard = metrics["standard"]["version_selection_accuracy"]
    date_filtered = metrics["date_filtered"]["version_selection_accuracy"]
    proposed = metrics["version_aware"]["version_selection_accuracy"]
    metrics["success_criteria"] = {
        "target_version_selection_accuracy": 0.85,
        "actual_version_selection_accuracy": proposed,
        "improvement_over_standard_percentage_points": 100 * (proposed - standard),
        "improvement_over_date_filtered_percentage_points": 100 * (proposed - date_filtered),
        "meets_85_percent_target": proposed >= 0.85,
        "meets_15_point_standard_improvement": proposed - standard >= 0.15,
        "meets_5_point_date_filtered_improvement": proposed - date_filtered >= 0.05,
        "result_status": "provisional_until_group_manual_verification",
    }
    combined.to_csv(
        RESULTS / "controlled_comparison_predictions.csv", index=False, encoding="utf-8"
    )
    (RESULTS / "controlled_comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

