import gym
import numpy as np
from isaac_world import IsaacSimPathfinder
import omni

class DronePathEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, 
                 goal_region = ((5,0,1), (5, 0, 1)), 
                 max_steps = 100, 
                 cube_path = "/World/Cube",
                 arena_bounds = ((-5,-5,0), (5, 5, 5)), 
                 num_obstacles = 2,
                 obstacle_radius = 1,
                 min_goal_clearance = 1.0,
                 dist_reward_scale = 0.05
                 viewport = None):
        super().__init__()

        self.stage = omni.usd.get_context().get_stage()
        self.world = IsaacSimPathfinder(self.stage, cube_path)

        self.max_step = max_steps
        self.t = 0

        #Observation Area
        #x.yz of drone + xyz target + 3x3 grid
        low = np.array([-np.inf]*6 + [0]*27, dtype=  np.float32)
        high = np.array([np.inf]*6 + [1]*27, dtype=  np.float32)
        self.observation_space = gym.spaces.Box(low, high)

        #Action Space (delta xyz that we can reach ie +- 1 unit per step)
        self.action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (3,), dtype = np.float32)
    def _random_point(self):
        return np.random.uniform(self.arena_min, self.arena_max)
    def _spawn_random_obstacle(self):
        self.world.clear_obstacles()
        tries = 0
        added = 0
        while added < self.num_obstacles and tries < self.num_obstacles * 10:
            tries = tries + 1
            center = self._random_point()
            if (np.linalg.norm(center - self.start) < self.min_goal_clearance
                or np.linalg.norm(center - self.goal) < self.min_goal_clearance):
                continue
            x, y, z = map(float, center)
            self.world.add_obstacle(x, y, z, radius = self.obstacle_radius)
            added = added + 1

    def _get_obs(self):
        pos = np.array(self.world.get_cube_position(), dtype = np.float32)
        goal = np.array(self.goal, dtype = np.float32)
        occ = self._occupancy_grid_3x3(pos)
        return np.concatenate([pos, goal, occ])
    
    def _occupancy_grid_3x3(self, pos):
        #3x3x3 mask centered around pos
        fx, fy, fz = map(float, pos)
        space = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    space.append(1.0 if self.world.is_position_blocked(fx + dx, fy + dy, fz+ dz)
                                 else 0.0)
        return np.array(space, dtype=np.float32)
    
    def _closest_obstacle_distance(self, pos):
        #distance from pos to the nearest hot zone
        if not self.world.sphericalObstacles:
            return np.inf
        px, py, pz = pos
        dists = [max(0.0, np.linalg.norm([px - cx, py - cy, pz, cz] - r))
                 for cx, cy, cz, r in self.world.sphericalObstacles]
        return min(dists)
    
    def _draw_path_debug(self, path):
        if not path:
            return
        from omni.isaac.debug_draw import _debug_draw
        dd = _debug_draw.acquire_debug_draw_interface()
        pts = [(float(x), float(y), float(z)) for x, y, z in path]
        #label id = 0 line width = 1 px color = RGBA (0,1,0,1)
        dd.draw_line_strip(*pts, color = (0.0, 1.0, 0.0, 1.0), thickness = 1.0)

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.start = (0, 0, 1)
        self.goal = (5, 0, 1)
        self.world.set_cube_position(*self.start)

        self._spawn_random_obstacle()

        self.t = 0
        obs = self._get_obs
        return obs, {}
    
    def step(self, action):
        self.t = self.t + 1

        #action -> waypoint -> path
        target = tuple(np.array(self.world.get_cube_position()) + action)
        path = self.world.find_path(
            start = tuple(map(int, self.world.get_cube_position())), 
            end = tuple(map(int, target))
        )
        if path:
            self.world.set_cube_position(*path[-1]) #move to last step in path
            self._draw_path_debug(path)
        #else stay put

        #Rewards
        pos = np.array(self.world.get_cube_position())
        dist = np.linalg.norm(pos - self.goal)
        #0.01 * (min distance to hot zones capped at 5 units)
        d_safe = self._closest_obstacle_distance(pos)
        dist_bonus = self.dist_reward_scale * min(d_safe, 5)

        reward = -0.1 #step cost
        reward += -0.05 * dist #shaping
        reward = reward + dist_bonus #bonus for avoiding hot zones
        done = False
        #penalty for collisions
        if self.world.is_position_blocked(*pos):
            reward = reward - 10.0
            done = True
        #goal bonus
        if dist < 0.3:
            reward = reward + 10.0

        #timeout (run out of fuel)
        if self.t >= self.max_step:
            done = True
        
        obs = self._get_obs()
        info = {"dist": float(dist)}
        return obs, reward, done, False, info
    
    def _ensure_viewport(self):
        if self.viewport is not None:
            return
        import omni.kit.viewport.utility as vp
        vp_iface = vp.get_viewport_interface()
        self.viewport = vp_iface.create_viewport_window("RL Viewport")
        self.viewport.set_active_camera("/OmniverseKit_Persp") #Default cam
        self.viewport.set_texture_resolution((640, 480))

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array", "Only rgb_array supported"
        self._ensure_viewport()
        tex = self.viewport.get_texture()
        if tex is None:
            return None
        
        img = np.frombuffer(tex.cpu_data, dtype=np.uint8)
        img = img.reshape(tex.height, tex.width, 4)[..., :3]
        return img.copy()

