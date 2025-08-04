from __future__ import annotations
import torch, numpy as np
import hydra
from omegaconf import DictConfig
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim.spawners.single_instance import clone_env
from isaac_world import IsaacSimPathfinder

# --------------------------------------------------------------------- #
# 1.  Config                                                            #
# --------------------------------------------------------------------- #
class DroneLabCfg(DirectRLEnvCfg):
    num_envs = 64
    episode_length_s = 20.0
    sim.dt = 1.0 / 60

    # task-specific knobs go in the `env` dict so they are Hydra-tunable
    env: dict = dict(
        arena_bounds=(( -5, -5, 0), ( 5,  5, 5)),
        num_obstacles=10,
        obstacle_radius=1.0,
        min_goal_clearance=1.5,
        max_steps=100,
    )


# --------------------------------------------------------------------- #
# 2.  Task                                                              #
# --------------------------------------------------------------------- #
class DroneVectorTask(DirectRLEnv):
    """64 independent drones, each controlled by one IsaacSimPathfinder."""

    cfg = DroneLabCfg

    # ------------  scene build (called once) -------------------------- #
    def setup_scene(self):
        # ❶  replicate *everything* under /World/env_i so PhysX steps all at once
        clone_env(self.cfg.num_envs, spacing=12.0)     # Isaac Lab helper

        # ❷  create one Pathfinder per env, pointing at its own cube
        self.ipf: list[IsaacSimPathfinder] = []
        for i in range(self.cfg.num_envs):
            cube_path = f"/World/env_{i}/Cube"         # clone_env keeps the names
            self.ipf.append(IsaacSimPathfinder(cube_path))

        # ❸  Isaac Lab bookkeeping tensors
        self.obs_dim, self.act_dim = 33, 3
        self.observation_space.shape = (self.obs_dim,)   # for SB3 wrappers

        self.rew_buf   = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones (self.num_envs, device=self.device)
        self.obs_buf   = torch.zeros((self.num_envs, self.obs_dim), device=self.device)

        # per-episode targets
        self.goal = torch.zeros((self.num_envs, 3), device=self.device)

    # ------------  reset a subset of envs ----------------------------- #
    def reset_idx(self, env_ids: torch.Tensor):
        cfg = self.cfg.env
        b    = env_ids.numel()
        low  = torch.tensor(cfg["arena_bounds"][0], device=self.device)
        high = torch.tensor(cfg["arena_bounds"][1], device=self.device)

        for idx in env_ids.cpu().numpy():
            ipf = self.ipf[idx]
            ipf.clear_obstacles()

            # start in arena centre
            centre = ((low + high) / 2).cpu().numpy()
            ipf.set_cube_position(*centre)

            # random goal
            while True:
                g = np.random.uniform(low.cpu(), high.cpu())
                if np.linalg.norm(g - centre) > 2.0:
                    break
            self.goal[idx] = torch.tensor(g, device=self.device)

            # random spherical obstacles
            added, tries = 0, 0
            while added < cfg["num_obstacles"] and tries < cfg["num_obstacles"] * 10:
                tries += 1
                c = np.random.uniform(low.cpu(), high.cpu())
                if (
                    np.linalg.norm(c - centre) < cfg["min_goal_clearance"]
                    or np.linalg.norm(c - g)   < cfg["min_goal_clearance"]
                ):
                    continue
                ipf.add_obstacle(*c, radius=cfg["obstacle_radius"])
                added += 1

        # first obs
        self.obs_buf[env_ids]   = self._compute_obs(env_ids)
        self.reset_buf[env_ids] = 0

    # ------------  RL step interface  --------------------------------- #
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply Δx,Δy,Δz from policy → A* path → set final voxel."""
        for i in range(self.num_envs):
            cur = np.array(self.ipf[i].get_cube_position(), dtype=float)
            tgt = cur + actions[i].cpu().numpy().clip(-1, 1)
            path = self.ipf[i].find_path(tuple(cur.astype(int)), tuple(tgt.astype(int)))
            if path:
                self.ipf[i].set_cube_position(*[float(j) for j in path[-1]])

    def post_physics_step(self):
        cfg = self.cfg.env
        for i in range(self.num_envs):
            pos  = np.array(self.ipf[i].get_cube_position(), dtype=float)
            dist_goal = np.linalg.norm(pos - self.goal[i].cpu().numpy())
            dist_safe = self.ipf[i].min_distance_to_spheres(*pos)

            r = -0.1 - 0.05 * dist_goal + 0.05 * min(dist_safe, 5.0)
            self.rew_buf[i] = r

            done = (
                dist_goal < 0.3
                or dist_safe <= 0.0
                or self.progress_buf[i] >= cfg["max_steps"] - 1
            )
            self.reset_buf[i] = done

        self.obs_buf[:] = self._compute_obs()

    # ------------  observation helper  -------------------------------- #
    def _compute_obs(self, env_ids=None):
        ids = env_ids if env_ids is not None else torch.arange(self.num_envs, device=self.device)
        obs = []
        for idx in ids.cpu().numpy():
            pos = np.array(self.ipf[idx].get_cube_position(), dtype=np.float32)
            vox = np.array(self.ipf[idx].local_occupancy_grid(*map(int, pos)), dtype=np.float32)
            obs.append(np.concatenate([pos, self.goal[idx].cpu().numpy(), vox]))
        return torch.as_tensor(np.stack(obs), device=self.device, dtype=torch.float32)


# --------------------------------------------------------------------- #
# 3.  Minimal Hydra entry to smoke-test                                 #
# --------------------------------------------------------------------- #
@hydra.main(version_base=None, config_name=None)
def main(_cfg: DictConfig):
    env = DroneVectorTask(DroneLabCfg)
    print("vectorised env ready:", env.num_envs, "envs")

    obs, _ = env.reset()
    done   = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(256):
        act = torch.empty((env.num_envs, 3), device=env.device).uniform_(-1, 1)
        obs, rew, done, _, _ = env.step(act)
        if done.any():
            env.reset_idx(done.nonzero(as_tuple=True)[0])
    print("✓ smoke-test passed.")


if __name__ == "__main__":
    main()
