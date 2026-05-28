"""Isaac Lab + Unitree G1 benchmark for the GATS-R port.

Trains a tiny L2 ensemble from on-policy random rollouts in the Isaac Lab G1
task, then evaluates four agents in head-to-head episodes and writes a single
CSV row per (method, episode, env) plus a small summary table to
``results/isaaclab_raw.csv`` and ``results/isaaclab_summary.csv``.

Methods compared:
    random       — uniform actions
    mppi         — batched MPPI in L2 (no monitor / no recovery / no CBF)
    gatsr_full   — GATS-R with all components on
    gatsr_no_rec — GATS-R minus the recovery dispatcher (ablation)

Run with:
    pwsh scripts/run_isaaclab.ps1 scripts/isaaclab_benchmark.py `
         --num_envs 16 --episodes 4 --max_steps 200 --train_steps 1024

Wall-clock budget: ~3-8 minutes on dual RTX 5090 at num_envs=16.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

parser = argparse.ArgumentParser(description="GATS-R Isaac Lab benchmark.")
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--train_steps", type=int, default=1024)
parser.add_argument("--methods", nargs="+", default=None)
parser.add_argument("--multi_gpu", action="store_true", help="Mirror L2 onto cuda:1 for split rollouts.")
parser.add_argument("--results_dir", type=str, default=str(ROOT / "results"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from gatsr.isaaclab.env import IsaacLabG1Env, IsaacLabG1Config  # noqa: E402
from gatsr.isaaclab.latent import G1EnsembleLatentModel, G1LatentConfig  # noqa: E402
from gatsr.isaaclab.agent import G1Agent, G1AgentConfig  # noqa: E402


REPORT_PATH = ROOT / "results" / "isaaclab_benchmark_report.txt"


def _log(msg: str) -> None:
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_PATH.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
    except Exception:
        pass
    try:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
    except Exception:
        pass


METHODS = ["random", "mppi", "gatsr_no_rec", "gatsr_full"]


def collect_random_data(env: IsaacLabG1Env, n_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll random actions and return (states, actions, next_states)."""
    s_list, a_list, sp_list = [], [], []
    env.reset(seed=0)
    for _ in range(n_steps):
        ps = env.physical_state.clone()
        a = 2 * torch.rand(env.num_envs, env.action_dim, device=env.device) - 1
        env.step(a)
        sp = env.physical_state
        s_list.append(ps)
        a_list.append(a)
        sp_list.append(sp.clone())
    # flatten across (n_steps, num_envs) -> (n_steps * num_envs, ...)
    S = torch.cat(s_list, dim=0)
    A = torch.cat(a_list, dim=0)
    SP = torch.cat(sp_list, dim=0)
    return S, A, SP


def make_cost_fn(env: IsaacLabG1Env):
    """Cost for MPPI: walk forward (lin_vel x > target) and stay upright."""
    target_vx = 0.5  # m/s forward

    def cost_fn(traj: torch.Tensor, actions: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        # traj layout: [lin_vel(3), ang_vel(3), grav(3), jp..., jv...]
        # reward forward velocity, penalize tilt and action magnitude
        lin_vel_x = traj[..., 0]              # (N, K, H)
        grav_xy = traj[..., 6:8]              # (N, K, H, 2)
        tilt = grav_xy.norm(dim=-1)           # (N, K, H)
        vel_err = (lin_vel_x.mean(dim=-1) - target_vx).abs()
        upright = tilt.mean(dim=-1)
        action_cost = actions.pow(2).mean(dim=(-1, -2))
        unc = eps.mean(dim=-1)
        return vel_err + 1.5 * upright + 0.005 * action_cost + 0.05 * unc

    return cost_fn


def build_agent(method: str, env: IsaacLabG1Env, model: G1EnsembleLatentModel) -> G1Agent:
    cfg = G1AgentConfig(
        use_mppi=method != "random",
        use_cbf=method == "gatsr_full",
        use_monitor=method.startswith("gatsr"),
        use_recovery=method == "gatsr_full",
        horizon=6,
        n_samples=32,
        n_iter=1,
    )
    return G1Agent(env, model, cfg)


def run_one(
    method: str,
    env: IsaacLabG1Env,
    model: G1EnsembleLatentModel,
    cost_fn,
    episodes: int,
    max_steps: int,
) -> list[dict]:
    """Returns a list of per-episode dicts (one row per (episode, env))."""
    rows: list[dict] = []
    for ep in range(episodes):
        _log(f"        [{method}] resetting env for ep {ep + 1} ...")
        env.reset(seed=1000 + ep)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        # build a fresh agent each episode to avoid cross-episode CUDA state
        agent = build_agent(method, env, model)
        _log(f"        [{method}] env+agent ready for ep {ep + 1}")
        ep_return = torch.zeros(env.num_envs, device=env.device)
        steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        t_ep = time.perf_counter()
        _log(f"[{method}] ep {ep + 1}/{episodes} starting (max {max_steps} steps) ...")
        for step in range(max_steps):
            ps = env.physical_state
            if method == "random":
                actions = 2 * torch.rand(env.num_envs, env.action_dim, device=env.device) - 1
                _, info = actions, {"plan_ms": 0.0, "cbf_intervened": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)}
                rec_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            else:
                actions, info = agent.act(ps, cost_fn)
                rec_mask = info["recovery_mask"]
            obs, reward, term, trunc, einfo = env.step(actions)
            ep_return = ep_return + alive.float() * reward
            steps = steps + alive.long()
            # an env terminates if term|trunc OR is_crashed
            done = term | trunc
            alive = alive & ~done
            if not alive.any():
                break
            if step > 0 and step % 30 == 0:
                _log(
                    f"        [{method}] ep {ep + 1} step {step}: alive={alive.sum().item()}/{env.num_envs} "
                    f"return_mean={ep_return.mean().item():.2f}"
                )
        ep_secs = time.perf_counter() - t_ep
        # one row per env so seeds aren't conflated
        for i in range(env.num_envs):
            row = dict(
                method=method,
                episode=ep,
                env_idx=i,
                steps=int(steps[i].item()),
                ep_return=float(ep_return[i].item()),
                success=int(steps[i].item() >= max_steps),  # didn't fall = "survived"
                safety_violations=int(agent.safety_violations[i].item()),
                failures_detected=int(agent.failures_detected[i].item()),
                recoveries_attempted=int(agent.recovery.attempts[i].item()),
                recoveries_succeeded=int(agent.recovery.successes[i].item()),
                time_to_recover=float(
                    agent.recovery.time_to_recover_accum[i].item()
                    / max(1, int(agent.recovery.successes[i].item()))
                )
                if int(agent.recovery.successes[i].item()) > 0
                else -1.0,
                planning_ms=float(agent.planning_ms_sum / max(1, agent.planning_steps)),
                wall_clock_s=ep_secs,
            )
            rows.append(row)
        _log(
            f"[{method}] ep {ep + 1}/{episodes}: "
            f"return mean={ep_return.mean().item():.2f} "
            f"steps mean={steps.float().mean().item():.1f} "
            f"alive={alive.sum().item()}/{env.num_envs} "
            f"({ep_secs:.1f}s)"
        )
    return rows


def write_summary(rows: list[dict], path: Path) -> None:
    from collections import defaultdict
    import statistics as st

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["method"]].append(r)
    fields = [
        "method", "n", "success_rate", "return_mean", "return_std",
        "safety_violations_mean", "recoveries_attempted_mean",
        "recovery_success_rate", "time_to_recover_mean", "planning_ms_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method, rs in sorted(groups.items()):
            rets = [r["ep_return"] for r in rs]
            sv = [r["safety_violations"] for r in rs]
            ra = [r["recoveries_attempted"] for r in rs]
            succ = sum(r["recoveries_succeeded"] for r in rs)
            attempts = sum(ra)
            ttrs = [r["time_to_recover"] for r in rs if r["time_to_recover"] >= 0]
            plan = [r["planning_ms"] for r in rs]
            w.writerow(dict(
                method=method,
                n=len(rs),
                success_rate=st.mean(r["success"] for r in rs),
                return_mean=st.mean(rets),
                return_std=st.pstdev(rets) if len(rets) > 1 else 0.0,
                safety_violations_mean=st.mean(sv),
                recoveries_attempted_mean=st.mean(ra),
                recovery_success_rate=(succ / attempts) if attempts else 0.0,
                time_to_recover_mean=st.mean(ttrs) if ttrs else -1.0,
                planning_ms_mean=st.mean(plan),
            ))


def main() -> int:
    REPORT_PATH.unlink(missing_ok=True)
    _log("[bench] launching Isaac Lab env ...")
    env = IsaacLabG1Env(IsaacLabG1Config(
        task=args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device, seed=0,
    ))
    _log(
        f"        task={args_cli.task} num_envs={env.num_envs} "
        f"obs_dim={env.obs_dim} action_dim={env.action_dim} "
        f"physical_dim={env.physical_dim} device={env.device}"
    )

    _log(f"[bench] collecting {args_cli.train_steps} random training transitions ...")
    t0 = time.perf_counter()
    S, A, SP = collect_random_data(env, args_cli.train_steps)
    _log(f"        collected {S.shape[0]} transitions in {time.perf_counter() - t0:.1f}s")
    _log(f"[bench] training L2 ensemble on {S.shape[0]} samples ...")
    model = G1EnsembleLatentModel(G1LatentConfig(
        state_dim=env.physical_dim,
        action_dim=env.action_dim,
        device=str(env.device),
        multi_gpu_rollouts=args_cli.multi_gpu,
    ))
    fit_t0 = time.perf_counter()
    model.fit(S, A, SP, verbose=False)
    _log(f"        training done in {time.perf_counter() - fit_t0:.1f}s")

    cost_fn = make_cost_fn(env)
    methods = args_cli.methods if args_cli.methods else METHODS
    rows: list[dict] = []
    for method in methods:
        _log(f"[bench] running method={method} ...")
        rows.extend(run_one(method, env, model, cost_fn, args_cli.episodes, args_cli.max_steps))

    out_raw = Path(args_cli.results_dir) / "isaaclab_raw.csv"
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    with out_raw.open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    _log(f"[bench] wrote raw -> {out_raw}")

    out_sum = Path(args_cli.results_dir) / "isaaclab_summary.csv"
    write_summary(rows, out_sum)
    _log(f"[bench] wrote summary -> {out_sum}")

    env.close()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
