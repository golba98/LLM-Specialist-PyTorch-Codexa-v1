"""Visible 100x22 specialist progress, using measured head step durations."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from llm_specialist.artifacts import REPO_ROOT, atomic_json, namespace_path


def start_viewer(status_path: Path) -> subprocess.Popen:
    """Launch Kitty and require an attached viewer before any training step."""
    if shutil.which("kitty") is None or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("Training requires a visible Kitty viewer and a graphical session.")
    namespace_path(status_path, "logs")
    process = subprocess.Popen([
        "kitty",
        "--override", "remember_window_size=no",
        "--override", "initial_window_width=100c",
        "--override", "initial_window_height=22c",
        "--title", "Specialist training",
        "--working-directory", str(REPO_ROOT),
        sys.executable, "-m", "llm_specialist.cli.specialist",
        "view", "--status", str(status_path),
    ])
    attached = status_path.with_suffix(".attached")
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if attached.is_file():
            return process
        if process.poll() is not None:
            raise RuntimeError("Kitty exited before the specialist viewer attached.")
        time.sleep(0.1)
    raise RuntimeError("Kitty did not attach the visible specialist viewer within 15 seconds.")


def render_progress(status: dict) -> str:
    """Render the complete terminal dashboard without fixed token assumptions."""
    step = status.get("step", 0)
    total = status.get("total_steps", 1)
    percentage = min(100.0, 100.0 * step / max(total, 1))
    filled = int(percentage / 2)
    eta = max(0, int(status.get("eta_seconds", 0)))
    eta_text = f"{eta // 3600:02d}:{eta // 60 % 60:02d}:{eta % 60:02d}"
    train_loss = status.get("training_loss")
    validation_loss = status.get("validation_loss")
    loss = "--" if train_loss is None else f"{train_loss:.5f}"
    validation = "--" if validation_loss is None else f"{validation_loss:.5f}"
    return "\n".join([
        "  SPECIALIST  |  FROZEN EMBEDDING CLASSIFICATION",
        "  " + "-" * 76,
        "",
        f"  {'#' * filled}{'-' * (50 - filled)}  {percentage:5.1f}%  ETA {eta_text}",
        "",
        f"  STATUS      {status.get('state', 'STARTING')}",
        f"  HEAD        {status.get('kind', '--')}",
        f"  STEPS       {step} / {total}",
        f"  SPEED       {status.get('tokens_per_second', 0):.0f} tok/s",
        f"  TRAIN LOSS  {loss}",
        f"  VAL LOSS    {validation}",
        f"  TOKENS      {status.get('total_tokens_seen', 0):,}",
        "",
        "  CHECKPOINT",
        f"  {status.get('checkpoint_path', '--')}",
        "",
        "  FP32 head | tokens represent cached requests presented to the head",
    ])


def view_status(status_path: Path) -> None:
    """Attach to an owned run and keep its last result visible."""
    status_path = namespace_path(status_path, "logs")
    atomic_json(status_path.with_suffix(".attached"), {"pid": os.getpid()})
    while True:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        print("\033[2J\033[H" + render_progress(status), flush=True)
        if status["state"] in ("COMPLETED", "EARLY_STOPPED", "FAILED", "INTERRUPTED"):
            input("\n  Press Enter to close.")
            return
        owner = status.get("pid")
        if owner:
            try:
                os.kill(owner, 0)
            except ProcessLookupError:
                print("\n  Training process exited. Press Enter to close.", flush=True)
                input()
                return
        time.sleep(1.0)
