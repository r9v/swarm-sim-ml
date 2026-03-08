import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box
from drone import Drone


class SwarmTargetEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "swarm_target_v0"}

    def __init__(
        self,
        n_drones=8,
        k_neighbors=5,
        target_pos=None,
        spawn_center=None,
        grid_spacing=2.0,
        dt=0.05,
        max_steps=1000,
        r_hit=1.0,
        r_collision=0.3,
        r_bounds=100.0,
    ):
        super().__init__()
        self.n_drones = n_drones
        self.k_neighbors = k_neighbors
        self.target_pos = np.array(target_pos or [60.0, 5.0, 0.0], dtype=np.float64)
        self.spawn_center = np.array(spawn_center or [0.0, 5.0, 0.0], dtype=np.float64)
        self.grid_spacing = grid_spacing
        self.dt = dt
        self.max_steps = max_steps
        self.r_hit = r_hit
        self.r_collision = r_collision
        self.r_bounds = r_bounds
        self.render_mode = None

        self.possible_agents = [f"drone_{i}" for i in range(n_drones)]
        self.obs_dim = 6 + 3 + 2 + 3 + k_neighbors * 6 + n_drones

        self._obs_space = Box(low=-2.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32)
        self._act_space = Box(low=-6.0, high=6.0, shape=(3,), dtype=np.float32)

        self.drones: dict[str, Drone] = {}
        self.init_dists: dict[str, float] = {}
        self.prev_dists: dict[str, float] = {}
        self.hit_angles: list[np.ndarray] = []
        self.step_count = 0

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self.drones = {}
        self.init_dists = {}
        self.prev_dists = {}
        self.hit_angles = []
        self.step_count = 0

        rng = np.random.default_rng(seed)

        dist = 60.0 + rng.uniform(-5, 5)
        angle = rng.uniform(0, 2 * np.pi)
        self.target_pos = np.array([
            dist * np.cos(angle),
            5.0 + rng.uniform(0, 3),
            dist * np.sin(angle),
        ])

        cols = 4
        for name in self.possible_agents:
            idx = int(name.split("_")[1])
            row = idx // cols
            col = idx % cols
            pos = self.spawn_center + np.array([
                (col - (cols - 1) / 2) * self.grid_spacing,
                0.0,
                (row - 0.5) * self.grid_spacing,
            ])
            pos[1] = max(0.5, pos[1])
            self.drones[name] = Drone(position=pos)
            d = np.linalg.norm(pos - self.target_pos)
            self.init_dists[name] = d
            self.prev_dists[name] = d

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self.step_count += 1

        for agent in self.agents:
            action = actions.get(agent, np.zeros(3))
            self.drones[agent].step(np.array(action, dtype=np.float64), self.dt)

        collision_set = self._find_collisions()

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        newly_dead = set()

        for agent in self.agents:
            drone = self.drones[agent]
            d_curr = np.linalg.norm(drone.position - self.target_pos)
            d_prev = self.prev_dists[agent]

            reward = 0.0
            terminated = False
            truncated = False

            r_approach = 2.0 * (d_prev - d_curr) - 0.01
            reward += r_approach

            nn_dist = self._nearest_neighbor_dist(agent)

            dir_vec = drone.position - self.target_pos
            dir_norm = np.linalg.norm(dir_vec)
            dir_unit = dir_vec / max(dir_norm, 1e-9)

            min_ang_diff = np.pi
            for other_name in self.agents:
                if other_name == agent:
                    continue
                other = self.drones[other_name]
                o_vec = other.position - self.target_pos
                o_unit = o_vec / max(np.linalg.norm(o_vec), 1e-9)
                dot = np.clip(np.dot(dir_unit, o_unit), -1.0, 1.0)
                min_ang_diff = min(min_ang_diff, np.arccos(dot))

            ang_thresh = np.pi / 4
            r_angular = 0.0
            if min_ang_diff < ang_thresh and d_curr > 5.0:
                r_angular = -0.4 * (1.0 - min_ang_diff / ang_thresh)
            reward += r_angular

            r_avoid = 0.0
            if nn_dist < 1.0:
                r_avoid = -1.0 * (1.0 - nn_dist)
            reward += r_avoid

            r_loiter = 0.0
            if d_curr < 5.0 and d_curr > self.r_hit:
                closing_speed = (d_prev - d_curr) / self.dt
                if closing_speed < 1.0:
                    r_loiter = -0.5 * (1.0 - closing_speed)
            reward += r_loiter

            if d_curr < self.r_hit:
                approach_vec = drone.position - self.target_pos
                norm = np.linalg.norm(approach_vec)
                approach_vec = approach_vec / max(norm, 1e-9)
                if self.hit_angles:
                    dots = [abs(np.dot(approach_vec, h)) for h in self.hit_angles]
                    angle_bonus = np.arccos(np.clip(min(dots), -1, 1)) / np.pi
                else:
                    angle_bonus = 1.0
                reward += 100.0 + 50.0 * angle_bonus
                self.hit_angles.append(approach_vec)
                terminated = True

            elif agent in collision_set:
                reward -= 20.0
                terminated = True

            elif drone.position[1] <= 0.0:
                reward -= 20.0
                terminated = True

            elif np.linalg.norm(drone.position) > self.r_bounds:
                reward -= 20.0
                terminated = True

            if not terminated and self.step_count >= self.max_steps:
                truncated = True
                reward -= 10.0

            self.prev_dists[agent] = d_curr
            observations[agent] = self._get_obs(agent)
            rewards[agent] = float(reward)
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {
                "distance_to_target": d_curr,
                "nn_dist": nn_dist,
                "r_approach": r_approach,
                "r_angular": r_angular,
                "r_avoid": r_avoid,
                "r_loiter": r_loiter,
                "min_ang_diff_deg": np.degrees(min_ang_diff),
            }

            if terminated or truncated:
                newly_dead.add(agent)

        self.agents = [a for a in self.agents if a not in newly_dead]

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent):
        drone = self.drones[agent]
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        obs[0:3] = drone.position / self.r_bounds
        obs[3:6] = drone.velocity / drone.max_speed
        rel_target = self.target_pos - drone.position
        obs[6:9] = rel_target / self.r_bounds

        horiz = np.sqrt(rel_target[0]**2 + rel_target[2]**2)
        obs[9] = np.arctan2(rel_target[2], rel_target[0]) / np.pi
        obs[10] = np.arctan2(rel_target[1], max(horiz, 1e-9)) / (np.pi / 2)

        positions = [self.drones[a].position for a in self.agents if a in self.drones]
        centroid = np.mean(positions, axis=0) if positions else drone.position
        obs[11:14] = (drone.position - centroid) / self.r_bounds

        neighbors = self._get_k_nearest(agent)
        nb_pos = 14
        nb_vel = nb_pos + self.k_neighbors * 3
        for j, (rel_pos, rel_vel) in enumerate(neighbors):
            obs[nb_pos + j * 3: nb_pos + j * 3 + 3] = rel_pos / self.r_bounds
            obs[nb_vel + j * 3: nb_vel + j * 3 + 3] = rel_vel / drone.max_speed

        idx = int(agent.split("_")[1])
        obs[-(self.n_drones):] = 0.0
        obs[-(self.n_drones) + idx] = 1.0

        return obs

    def _get_k_nearest(self, agent):
        drone = self.drones[agent]
        dists = []
        for other_name in self.agents:
            if other_name == agent:
                continue
            other = self.drones[other_name]
            rel_pos = other.position - drone.position
            rel_vel = other.velocity - drone.velocity
            d = np.linalg.norm(rel_pos)
            dists.append((d, rel_pos.copy(), rel_vel.copy()))
        dists.sort(key=lambda x: x[0])
        return [(rp, rv) for (_, rp, rv) in dists[: self.k_neighbors]]

    def _nearest_neighbor_dist(self, agent):
        drone = self.drones[agent]
        min_d = float("inf")
        for other_name in self.agents:
            if other_name == agent:
                continue
            d = np.linalg.norm(self.drones[other_name].position - drone.position)
            min_d = min(min_d, d)
        return min_d

    def _find_collisions(self):
        collided = set()
        agents = list(self.agents)
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                d = np.linalg.norm(self.drones[a].position - self.drones[b].position)
                if d < self.r_collision:
                    collided.add(a)
                    collided.add(b)
        return collided

    def render(self):
        pass

    def close(self):
        pass
