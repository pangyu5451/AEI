import json
from pathlib import Path

import numpy as np
import pytest

from AGG_FWC.evaluate import (
    compute_classification_metrics,
    evaluate_predictions_file,
    main,
    save_metrics_json,
)


def test_compute_metrics_reports_accuracy_macro_and_per_class():
    labels = [0, 0, 1, 1, 2, 2]
    preds = [0, 1, 1, 1, 2, 0]

    metrics = compute_classification_metrics(labels, preds, num_classes=3)

    expected_macro_f1 = (0.5 + 0.8 + 2 / 3) / 3
    assert metrics["accuracy"] == 4 / 6
    assert len(metrics["per_class"]) == 3
    assert metrics["per_class"][1]["precision"] == pytest.approx(2 / 3)
    assert metrics["per_class"][1]["recall"] == pytest.approx(1.0)
    assert metrics["per_class"][1]["f1"] == pytest.approx(0.8)
    assert metrics["macro_f1"] == pytest.approx(expected_macro_f1)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_save_metrics_json_serializes_numpy_values(tmp_path):
    metrics = {
        "accuracy": np.float64(4 / 6),
        "per_class": np.array([0.5, 2 / 3, 2 / 3]),
        "confusion_matrix": np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]]),
    }
    output_path = tmp_path / "metrics.json"

    saved = save_metrics_json(metrics, output_path)

    assert saved == output_path
    assert output_path.exists()
    json.loads(output_path.read_text(encoding="utf-8"))


def test_save_metrics_json_serializes_paths_numpy_scalars_and_arrays(tmp_path):
    metrics = {
        "config_path": tmp_path / "config.json",
        "checkpoint_path": Path(tmp_path) / "checkpoints" / "final.pt",
        "scalar_float": np.float32(0.25),
        "scalar_int": np.int64(7),
        "scalar_bool": np.bool_(True),
        "array": np.array([[1, 2], [3, 4]], dtype=np.int16),
    }
    output_path = tmp_path / "paths-and-numpy.json"

    save_metrics_json(metrics, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "config_path": str(tmp_path / "config.json"),
        "checkpoint_path": str(tmp_path / "checkpoints" / "final.pt"),
        "scalar_float": 0.25,
        "scalar_int": 7,
        "scalar_bool": True,
        "array": [[1, 2], [3, 4]],
    }


def test_save_metrics_json_rejects_non_finite_numbers(tmp_path):
    with pytest.raises(ValueError, match="finite JSON"):
        save_metrics_json({"score": float("nan")}, tmp_path / "metrics.json")

    assert not (tmp_path / "metrics.json").exists()


def test_save_metrics_json_does_not_overwrite_existing_output(tmp_path):
    output_path = tmp_path / "metrics.json"
    output_path.write_text('{"sentinel": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_metrics_json({"accuracy": 1.0}, output_path)

    assert output_path.read_text(encoding="utf-8") == '{"sentinel": true}\n'


def test_save_metrics_json_cleans_temporary_file_when_atomic_replace_fails(
    tmp_path, monkeypatch
):
    import AGG_FWC.evaluate as evaluate_module

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(evaluate_module.os, "replace", fail_replace)
    output_path = tmp_path / "metrics.json"

    with pytest.raises(OSError, match="replace failure"):
        save_metrics_json({"accuracy": 1.0}, output_path)

    assert not output_path.exists()
    assert list(tmp_path.glob(f".{output_path.name}.*")) == []


def test_compute_metrics_includes_zero_support_class_in_standard_macro_f1():
    metrics = compute_classification_metrics(
        labels=[0, 0, 1],
        predictions=[0, 1, 1],
        num_classes=3,
    )

    assert metrics["per_class"][2] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0.0) / 3)


def test_compute_metrics_uses_standard_f1_formula_for_asymmetric_errors():
    metrics = compute_classification_metrics(
        labels=[0, 0, 0, 1],
        predictions=[0, 1, 1, 1],
        num_classes=2,
    )

    assert metrics["per_class"][0]["f1"] == pytest.approx(0.5)
    assert metrics["per_class"][1]["f1"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(0.5)


def test_evaluate_predictions_file_recomputes_and_saves_metrics_with_metadata(tmp_path):
    predictions_path = tmp_path / "predictions.npz"
    output_path = tmp_path / "recomputed-metrics.json"
    np.savez(predictions_path, labels=np.array([0, 1, 1]), predictions=np.array([0, 0, 1]))

    saved = evaluate_predictions_file(
        predictions_path,
        num_classes=2,
        output_path=output_path,
        metadata={
            "split": "final-target",
            "run_id": "unit",
            "config_path": str(tmp_path / "config.json"),
        },
    )

    assert saved == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["accuracy"] == pytest.approx(2 / 3)
    assert payload["confusion_matrix"] == [[1, 0], [1, 1]]
    assert payload["run_id"] == "unit"
    assert payload["config_path"] == str(tmp_path / "config.json")
    assert payload["metadata"] == {
        "split": "final-target",
        "run_id": "unit",
        "config_path": str(tmp_path / "config.json"),
    }


@pytest.mark.parametrize(
    "metadata",
    [None, {}, {"run_id": "unit"}, {"config_path": "config.json"}],
)
def test_evaluate_predictions_file_requires_run_id_and_config_path_metadata(
    tmp_path, metadata
):
    predictions_path = tmp_path / "predictions.npz"
    np.savez(predictions_path, labels=np.array([0]), predictions=np.array([0]))

    with pytest.raises(ValueError, match="run_id.*config_path"):
        evaluate_predictions_file(
            predictions_path,
            num_classes=1,
            output_path=tmp_path / f"metrics-{len(list(tmp_path.iterdir()))}.json",
            metadata=metadata,
        )


def test_evaluate_predictions_file_rejects_missing_prediction_arrays(tmp_path):
    predictions_path = tmp_path / "incomplete.npz"
    np.savez(predictions_path, labels=np.array([0]))

    with pytest.raises(ValueError, match="labels.*predictions"):
        evaluate_predictions_file(
            predictions_path,
            num_classes=2,
            output_path=tmp_path / "out.json",
            metadata={"run_id": "unit", "config_path": "config.json"},
        )


def test_evaluate_predictions_file_rejects_mismatched_lengths(tmp_path):
    predictions_path = tmp_path / "mismatch.npz"
    np.savez(predictions_path, labels=np.array([0, 1]), predictions=np.array([0]))

    with pytest.raises(ValueError, match="same length"):
        evaluate_predictions_file(
            predictions_path,
            num_classes=2,
            output_path=tmp_path / "out.json",
            metadata={"run_id": "unit", "config_path": "config.json"},
        )


@pytest.mark.parametrize(
    "labels,predictions",
    [([ [0, 1] ], [0, 1]), ([0, 1], [[0, 1]])],
)
def test_compute_metrics_rejects_non_one_dimensional_arrays(labels, predictions):
    with pytest.raises(ValueError, match="strictly one-dimensional"):
        compute_classification_metrics(labels, predictions, num_classes=2)


@pytest.mark.parametrize(
    "labels,predictions",
    [([0, 2], [0, 1]), ([0, 1], [-1, 1]), ([0.5], [0])],
)
def test_compute_metrics_rejects_illegal_class_indices(labels, predictions):
    with pytest.raises(ValueError, match="valid class indices"):
        compute_classification_metrics(labels, predictions, num_classes=2)


def test_compute_metrics_rejects_empty_samples():
    with pytest.raises(ValueError, match="non-empty"):
        compute_classification_metrics([], [], num_classes=2)


@pytest.mark.parametrize("num_classes", [0, -1, 2.0, True])
def test_compute_metrics_rejects_invalid_num_classes(num_classes):
    with pytest.raises(ValueError, match="positive integer"):
        compute_classification_metrics([0], [0], num_classes=num_classes)


def test_evaluate_predictions_cli_entrypoint_saves_output(tmp_path):
    predictions_path = tmp_path / "predictions.npz"
    output_path = tmp_path / "cli-metrics.json"
    np.savez(predictions_path, labels=np.array([0, 1]), predictions=np.array([0, 1]))

    returned = main(
        [
            str(predictions_path),
            "--num-classes",
            "2",
            "--output-path",
            str(output_path),
            "--metadata",
            json.dumps(
                {
                    "source": "cli-test",
                    "run_id": "cli-test",
                    "config_path": str(tmp_path / "config.json"),
                }
            ),
        ]
    )

    assert returned == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "cli-test"
    assert payload["config_path"] == str(tmp_path / "config.json")
    assert payload["metadata"]["source"] == "cli-test"


def test_evaluate_predictions_cli_requires_metadata(tmp_path):
    predictions_path = tmp_path / "predictions.npz"
    np.savez(predictions_path, labels=np.array([0, 1]), predictions=np.array([0, 1]))

    with pytest.raises(SystemExit):
        main(
            [
                str(predictions_path),
                "--num-classes",
                "2",
                "--output-path",
                str(tmp_path / "metrics.json"),
            ]
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_evaluate_predictions_cli_rejects_non_finite_metadata(tmp_path, constant):
    predictions_path = tmp_path / "predictions.npz"
    np.savez(predictions_path, labels=np.array([0]), predictions=np.array([0]))

    with pytest.raises(SystemExit):
        main(
            [
                str(predictions_path),
                "--num-classes",
                "1",
                "--output-path",
                str(tmp_path / "metrics.json"),
                "--metadata",
                f'{{"run_id":"unit","config_path":"config.json","score":{constant}}}',
            ]
        )
