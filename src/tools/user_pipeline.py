"""Runner de setup para usuario (dados -> treino/promote opcional -> API).

Uso:
    poetry run python -m src.tools.user_pipeline --mode bootstrap --device auto
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AIRFLOW_COMPOSE = PROJECT_ROOT / "docker-compose-airflow.yaml"
API_COMPOSE = PROJECT_ROOT / "src" / "api" / "docker-compose.yml"
ML_COMPOSE = PROJECT_ROOT / "src" / "ml_workstation" / "docker-compose.yml"
INMET_CONFIG = PROJECT_ROOT / "src" / "data_airflow" / "config" / "inmet_scraping.yml"
MODEL_ROOT = PROJECT_ROOT / "src" / "api" / "ml_models"
DEFAULT_EXPERIMENTS = [
    "weather_forecasting_h72",
    "weather_forecasting_h168",
    "weather_forecasting_h336",
]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Comando falhou ({completed.returncode}): {' '.join(cmd)}")


def _capture(cmd: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd or PROJECT_ROOT),
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Comando falhou ({completed.returncode}): {' '.join(cmd)}\n{completed.stderr}"
        )
    return completed.stdout.strip()


def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _validate_prereqs() -> None:
    missing = [name for name in ("docker", "poetry") if not _has_cmd(name)]
    if missing:
        raise RuntimeError(f"Dependencias ausentes: {', '.join(missing)}")
    _run(["docker", "compose", "version"])


def _set_year_range(start_year: int, end_year: int) -> None:
    if start_year > end_year:
        raise ValueError("start_year deve ser <= end_year")
    INMET_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    INMET_CONFIG.write_text(
        yaml.safe_dump(
            {"start_year": int(start_year), "end_year": int(end_year)},
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    print(f"Configuracao de anos atualizada em {INMET_CONFIG}")


def _detect_device(device_arg: str) -> str:
    if device_arg in {"cpu", "cuda"}:
        return device_arg
    if device_arg != "auto":
        raise ValueError("device deve ser cpu|cuda|auto")
    if _has_cmd("nvidia-smi"):
        try:
            _run(["nvidia-smi"])
            return "cuda"
        except RuntimeError:
            return "cpu"
    return "cpu"


def _ensure_model_artifacts(model_root: Path, experiments: list[str]) -> None:
    missing: list[str] = []
    for exp in experiments:
        model_dir = model_root / exp
        checks = [
            model_dir / "MLmodel",
            model_dir / "manifest.json",
            model_dir / "data",
        ]
        if not model_dir.is_dir() or not all(p.exists() for p in checks):
            missing.append(exp)
    if missing:
        raise RuntimeError(
            "Export incompleto em src/api/ml_models para: "
            f"{missing}. Esperado: MLmodel + data/ + manifest.json."
        )


def _has_model_artifacts(model_root: Path, experiments: list[str]) -> bool:
    try:
        _ensure_model_artifacts(model_root, experiments)
        return True
    except RuntimeError:
        return False


def _airflow_up() -> None:
    _run(["docker", "compose", "-f", str(AIRFLOW_COMPOSE), "up", "airflow-init"])
    _run(["docker", "compose", "-f", str(AIRFLOW_COMPOSE), "up", "-d"])


def _trigger_dag(dag_id: str) -> str:
    _capture(
        [
            "docker",
            "compose",
            "-f",
            str(AIRFLOW_COMPOSE),
            "exec",
            "-T",
            "airflow-webserver",
            "airflow",
            "dags",
            "trigger",
            dag_id,
        ]
    )
    # O output de `trigger` pode vir quebrado em mais de uma linha; usa list-runs como fonte de verdade.
    for _ in range(12):
        runs = _list_dag_runs(dag_id)
        if runs:
            return runs[0]["run_id"]
        time.sleep(1)
    raise RuntimeError(f"Nao foi possivel identificar run_id para DAG {dag_id}")


def _list_dag_runs(dag_id: str) -> list[dict[str, str]]:
    raw = _capture(
        [
            "docker",
            "compose",
            "-f",
            str(AIRFLOW_COMPOSE),
            "exec",
            "-T",
            "airflow-webserver",
            "airflow",
            "dags",
            "list-runs",
            "-d",
            dag_id,
            "-o",
            "plain",
            "--no-backfill",
        ]
    )
    rows: list[dict[str, str]] = []
    for line in raw.splitlines():
        match = re.match(
            rf"^{re.escape(dag_id)}\s+(manual__\S+)\s+(\w+)\s+",
            line.strip(),
        )
        if match:
            rows.append({"run_id": match.group(1), "state": match.group(2).lower()})
    return rows


def _dag_run_state(dag_id: str, run_id: str) -> str:
    for row in _list_dag_runs(dag_id):
        if row["run_id"] == run_id:
            return row["state"]
    return ""


def _wait_dag(dag_id: str, run_id: str, timeout_s: int = 1800) -> None:
    start = time.time()
    while True:
        state = _dag_run_state(dag_id, run_id)
        if state == "success":
            print(f"DAG {dag_id} concluida com sucesso ({run_id})")
            return
        if state in {"failed"}:
            raise RuntimeError(f"DAG {dag_id} falhou ({run_id})")
        if not state:
            print(f"Aguardando DAG {dag_id} ({run_id}) aparecer em list-runs...")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timeout aguardando DAG {dag_id} ({run_id})")
        time.sleep(10)


def _run_data_pipeline() -> None:
    for dag_id in ("inmet_download_raw", "data_cleaning", "data_feature_engineering"):
        run_id = _trigger_dag(dag_id)
        _wait_dag(dag_id, run_id)


def _run_training_and_promote(device: str) -> None:
    profile = "train-gpu" if device == "cuda" else "train"
    service = "trainer-gpu" if device == "cuda" else "trainer"
    _run(["docker", "compose", "-f", str(ML_COMPOSE), "--profile", profile, "build", service])
    for exp in DEFAULT_EXPERIMENTS:
        config_name = exp.replace("weather_forecasting_", "tft_") + "_v1.json"
        config_path = f"//app/experiments/tft/{config_name}"
        _run(
            [
                "docker",
                "compose",
                "-f",
                str(ML_COMPOSE),
                "--profile",
                profile,
                "run",
                "--rm",
                service,
                "--config",
                config_path,
            ]
        )
        _run(
            [
                "poetry",
                "run",
                "python",
                "-m",
                "src.ml_workstation.promotion.run_promote",
                "--experiment-name",
                exp,
                "--export-dir",
                "src/api/ml_models",
            ]
        )


def _up_api(device: str) -> None:
    env = os.environ.copy()
    env["DEVICE"] = device
    cmd = ["docker", "compose", "-f", str(API_COMPOSE), "--profile", "api", "up", "--build", "-d"]
    print(f"\n$ {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError("Falha ao subir API")


def _healthcheck_api() -> None:
    _run(
        [
            "docker",
            "compose",
            "-f",
            str(API_COMPOSE),
            "exec",
            "-T",
            "weatherops-api",
            "curl",
            "-fsS",
            "http://localhost:8000/health/ready",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Setup completo para execucao orientada ao usuario.")
    parser.add_argument("--mode", choices=["bootstrap", "train", "full"], default="bootstrap")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Nao sobe a API (util para validacao parcial).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_prereqs()
    _set_year_range(args.start_year, args.end_year)
    device = _detect_device(args.device)
    print(f"Device efetivo: {device}")

    _airflow_up()
    _run_data_pipeline()

    if args.mode == "train":
        _run_training_and_promote(device=device)
        _ensure_model_artifacts(MODEL_ROOT, DEFAULT_EXPERIMENTS)
    elif args.mode == "bootstrap":
        _ensure_model_artifacts(MODEL_ROOT, DEFAULT_EXPERIMENTS)
    else:  # full
        if not _has_model_artifacts(MODEL_ROOT, DEFAULT_EXPERIMENTS):
            print("Modelos bootstrap ausentes/incompletos; executando treino e export...")
            _run_training_and_promote(device=device)
        _ensure_model_artifacts(MODEL_ROOT, DEFAULT_EXPERIMENTS)

    if not args.skip_api:
        _up_api(device=device)
        _healthcheck_api()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERRO: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

