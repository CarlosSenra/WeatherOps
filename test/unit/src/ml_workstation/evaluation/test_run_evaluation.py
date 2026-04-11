from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.ml_workstation.evaluation import run_evaluation


def test_build_parser_reads_expected_args() -> None:
    parser = run_evaluation._build_parser()

    args = parser.parse_args([
        "--run-id",
        "abc123",
        "--target-index",
        "1",
        "--horizon-step",
        "2",
        "--max-points",
        "100",
        "--device",
        "cpu",
        "--output-dir",
        "evaluation_results",
    ])

    assert args.run_id == "abc123"
    assert args.target_index == 1
    assert args.horizon_step == 2
    assert args.max_points == 100


def test_main_calls_evaluate_run(monkeypatch, capsys) -> None:
    called = {}

    def _fake_eval(**kwargs):
        called.update(kwargs)
        return Path("evaluation_results/out.html")

    monkeypatch.setattr(run_evaluation, "evaluate_run", _fake_eval)
    monkeypatch.setattr(
        run_evaluation,
        "_build_parser",
        lambda: _parser_stub(),
    )

    run_evaluation.main()

    out = capsys.readouterr().out
    assert "Grafico salvo em:" in out
    assert called["run_id"] == "abc123"
    assert called["target_index"] == 0


class _ParserStub:
    def parse_args(self):
        return SimpleNamespace(
            run_id="abc123",
            target_index=0,
            horizon_step=0,
            max_points=2000,
            device="cpu",
            output_dir="evaluation_results",
            tracking_uri=None,
        )


def _parser_stub() -> _ParserStub:
    return _ParserStub()
