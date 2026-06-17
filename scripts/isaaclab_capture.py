"""Capture real RGB frames of the actual Unitree G1 from the Isaac Lab sim.

Runs the chosen method (default GATS-R) on the registered G1 task, headless but
with the RTX renderer on (``--enable_cameras``), aims the viewport camera at
env 0, and writes PNG frames to ``results/frames/``. These are genuine renders
of the actual robot in the actual environment this repo drives — not stylised
art. Used to source the figures in paper/.

Run via the launcher (isaaclab conda env):
    scripts\run_isaaclab.bat scripts\isaaclab_capture.py ^
        --task Isaac-Velocity-Rough-G1-v0 --method gatsr_full ^
        --train_steps 512 --max_steps 140 --every 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

parser = argparse.ArgumentParser(description="Capture real G1 frames from Isaac Lab.")
parser.add_argument("--task", default="Isaac-Velocity-Rough-G1-v0")
parser.add_argument("--method", default="gatsr_full", choices=["zero", "random", "mppi", "gatsr_full"])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--train_steps", type=int, default=512)
parser.add_argument("--max_steps", type=int, default=140)
parser.add_argument("--every", type=int, default=3, help="capture a frame every N control steps")
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=900)
parser.add_argument("--out", default=str(ROOT / "results" / "frames"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# A live viewport is required for capture_viewport_to_file to write frames, so
# we run windowed (not headless) with the RTX renderer on. A window opens
# briefly while frames are captured.
args_cli.headless = False
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gatsr.isaaclab.env import IsaacLabG1Env, IsaacLabG1Config  # noqa: E402
from gatsr.isaaclab.latent import G1EnsembleLatentModel, G1LatentConfig  # noqa: E402
from gatsr.isaaclab.agent import G1Agent, G1AgentConfig  # noqa: E402

# viewport + camera helpers (module paths vary across Isaac Sim versions)
try:
    from isaacsim.core.utils.viewports import set_camera_view
except Exception:  # pragma: no cover
    from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file


OUT = Path(args_cli.out)
OUT.mkdir(parents=True, exist_ok=True)


def pump(n: int) -> None:
    for _ in range(n):
        simulation_app.update()


def robot_pos(env: IsaacLabG1Env):
    robot = env._env.unwrapped.scene["robot"]
    return robot.data.root_pos_w[0].detach().cpu().numpy()


def grab(env: IsaacLabG1Env, path: Path, az: float = 1.5, side: float = 1.5, up: float = 0.75) -> bool:
    p = robot_pos(env)
    target = (float(p[0]), float(p[1]), float(p[2]) + 0.45)
    eye = (float(p[0]) + az, float(p[1]) + side, float(p[2]) + up)
    set_camera_view(eye, target)
    pump(4)
    vp = get_active_viewport()
    try:
        vp.resolution = (args_cli.width, args_cli.height)
    except Exception:
        pass
    pump(3)
    if path.exists():
        path.unlink()
    capture_viewport_to_file(vp, str(path))
    # the capture writes asynchronously over subsequent rendered frames; pump
    # the app until the file actually lands on disk (bounded).
    for _ in range(120):
        simulation_app.update()
        if path.exists() and path.stat().st_size > 0:
            return True
    print(f"[cap] WARN: frame not written: {path.name}", flush=True)
    return False


def collect_random_data(env: IsaacLabG1Env, n_steps: int):
    s, a, sp = [], [], []
    env.reset(seed=0)
    for _ in range(n_steps):
        ps = env.physical_state.clone()
        act = 2 * torch.rand(env.num_envs, env.action_dim, device=env.device) - 1
        env.step(act)
        s.append(ps); a.append(act); sp.append(env.physical_state.clone())
    return torch.cat(s), torch.cat(a), torch.cat(sp)


def make_cost_fn():
    target_vx = 0.5
    def cost_fn(traj, actions, eps):
        vel_err = (traj[..., 0].mean(dim=-1) - target_vx).abs()
        upright = traj[..., 6:8].norm(dim=-1).mean(dim=-1)
        action_cost = actions.pow(2).mean(dim=(-1, -2))
        return vel_err + 1.5 * upright + 0.005 * action_cost + 0.05 * eps.mean(dim=-1)
    return cost_fn


def main() -> int:
    print(f"[cap] {args_cli.task}  method={args_cli.method}  -> {OUT}", flush=True)
    env = IsaacLabG1Env(IsaacLabG1Config(
        task=args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device, seed=0))
    # hide the velocity-command debug arrows so the robot itself is unobstructed
    for mgr in ("command_manager",):
        try:
            getattr(env._env.unwrapped, mgr).set_debug_vis(False)
            print(f"[cap] disabled {mgr} debug vis", flush=True)
        except Exception as e:
            print(f"[cap] note: could not disable {mgr} debug vis: {e}", flush=True)

    agent = cost_fn = None
    if args_cli.method in ("mppi", "gatsr_full"):
        print(f"[cap] fitting L2 on {args_cli.train_steps} random transitions ...", flush=True)
        S, A, SP = collect_random_data(env, args_cli.train_steps)
        model = G1EnsembleLatentModel(G1LatentConfig(
            state_dim=env.physical_dim, action_dim=env.action_dim,
            device=str(env.device), multi_gpu_rollouts=False))
        model.fit(S, A, SP, verbose=False)
        agent = G1Agent(env, model, G1AgentConfig(
            use_mppi=True, use_cbf=args_cli.method == "gatsr_full",
            use_monitor=args_cli.method == "gatsr_full",
            use_recovery=args_cli.method == "gatsr_full",
            horizon=6, n_samples=32, n_iter=1))
        cost_fn = make_cost_fn()

    env.reset(seed=0)
    pump(20)  # let the scene + lighting settle before the first shot
    grab(env, OUT / f"{args_cli.method}_step000.png")

    n_caps = 1
    for step in range(1, args_cli.max_steps + 1):
        with torch.inference_mode():
            ps = env.physical_state
            fallen = bool(env.is_fallen()[0]) if args_cli.method == "gatsr_full" else False
            if args_cli.method == "zero":
                actions = torch.zeros(env.num_envs, env.action_dim, device=env.device)
            elif args_cli.method == "random":
                actions = 2 * torch.rand(env.num_envs, env.action_dim, device=env.device) - 1
            else:
                actions, info = agent.act(ps, cost_fn)
            recovering = False
            if agent is not None and args_cli.method == "gatsr_full":
                recovering = bool(info["recovery_mask"][0])
            if recovering:
                env.recover_step(actions)
            else:
                env.step(actions)
        # capture on a cadence, and always when fallen/recovering (the money shots)
        if step % args_cli.every == 0 or fallen or recovering:
            tag = "rec" if recovering else ("fall" if fallen else "run")
            grab(env, OUT / f"{args_cli.method}_step{step:03d}_{tag}.png")
            n_caps += 1
        if step % 20 == 0:
            print(f"[cap] step {step}/{args_cli.max_steps}  frames={n_caps}", flush=True)

    print(f"[cap] done. {n_caps} frames in {OUT}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
