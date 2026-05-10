from __future__ import annotations

from sklearn.metrics import classification_report, confusion_matrix


def classification_summary(y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    return {
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
