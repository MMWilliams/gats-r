"""Headless smoke-test: launch Isaac Lab, instantiate the G1 flat-velocity task
with a few parallel envs, run a handful of random-action steps, then exit.

Run via the helper at scripts/run_isaaclab.ps1, or manually with env vars set:

    $env:CARB_APP_PATH = "C:\\isaac-sim\\kit"
    $env:EXP_PATH      = "C:\\isaac-sim\\apps"
    $env:ISAAC_PATH    = "C:\\isaac-sim"
    $env:PYTHONPATH    = "C:\\isaac-sim\\site"
    & "C:\\Users\\reese\\miniconda3\\envs\\isaaclab\\python.exe" `
        C:\\Users\\reese\\Downloads\\robotics_research\\scripts\\isaaclab_smoke.py
"""

from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher

# CLI BEFORE app launch so AppLauncher can read --headless etc.
parser = argparse.ArgumentParser(description="GATS-R Isaac Lab smoke test.")
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--n_steps", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# force headless for a smoke test
args_cli.headless = True
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that depend on the app must happen after AppLauncher.
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  (registers gym IDs)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


REPORT_PATH = r"C:\Users\reese\Downloads\robotics_research\results\isaaclab_smoke_report.txt"


def _log(msg: str) -> None:
    # write to the report file first; print may raise BrokenPipeError when
    # stdout is redirected to nul on Windows.
    try:
        with open(REPORT_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
    except Exception:
        pass
    try:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def main() -> int:
    # truncate the report
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass
    t0 = time.perf_counter()
    _log(f"[smoke] preparing cfg for {args_cli.task} (num_envs={args_cli.num_envs}) ...")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    _log(f"[smoke] device={args_cli.device}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, info = env.reset()
    elapsed = time.perf_counter() - t0
    _log(f"[smoke] env up in {elapsed:.1f}s on {args_cli.device}")
    _log(f"        observation_space={env.observation_space}")
    _log(f"        action_space={env.action_space}")

    n_envs = env.unwrapped.num_envs
    act_shape = env.action_space.shape
    step_times = []
    for step in range(args_cli.n_steps):
        with torch.inference_mode():
            actions = 2 * torch.rand(act_shape, device=env.unwrapped.device) - 1
            t = time.perf_counter()
            obs, r, term, trunc, info = env.step(actions)
            step_times.append(time.perf_counter() - t)
            if step == 0:
                _log(
                    f"        first step: reward shape={tuple(r.shape)}, "
                    f"terminated shape={tuple(term.shape)}, "
                    f"truncated shape={tuple(trunc.shape)}"
                )
    mean_step_ms = 1000.0 * (sum(step_times) / len(step_times))
    fps = n_envs / (sum(step_times) / len(step_times))
    _log(
        f"[smoke] {args_cli.n_steps} steps OK. mean step time = {mean_step_ms:.2f} ms "
        f"({fps:.0f} steps/s aggregate across {n_envs} envs)"
    )

    env.close()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
