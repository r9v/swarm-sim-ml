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
        spawn_radius=6.0,
        dt=0.05,
        max_steps=1000,
        r_hit=1.0,
        r_collision=0.3,
        r_engage=10.0,
        r_bounds=100.0,
    ):
        super().__init__()
        self.n_drones = n_drones
        self.k_neighbors = k_neighbors
        self.target_pos = np.array(target_pos or [60.0, 5.0, 0.0], dtype=np.float64)
        self.spawn_center = np.array(spawn_center or [0.0, 5.0, 0.0], dtype=np.float64)
        self.spawn_radius = spawn_radius
        self.dt = dt
        self.max_steps = max_steps
        self.r_hit = r_hit
        self.r_collision = r_collision
        self.r_engage = r_engage
        self.r_bounds = r_bounds
        self.render_mode = None

        self.possible_agents = [f"drone_{i}" for i in range(n_drones)]
        self.obs_dim = 6 + 3 + k_neighbors * 6 + 3

        self._obs_space = Box(low=-2.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32)
        self._act_space = Box(low=-3.0, high=3.0, shape=(3,), dtype=np.float32)

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

        self._target_angle = rng.uniform(0, 2 * np.pi)
        dist = 60.0 + rng.uniform(-5, 5)
        self.target_pos = np.array([
            dist * np.cos(self._target_angle),
            5.0 + rng.uniform(0, 3),
            dist * np.sin(self._target_angle),
        ])

        for name in self.possible_agents:
            offset = rng.uniform(-1, 1, size=3) * self.spawn_radius
            pos = self.spawn_center + offset
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
            d_init = self.init_dists[agent]

            reward = 0.0
            terminated = False
            truncated = False

            r_approach = 2.0 * (d_prev - d_curr) - 0.01
            reward += r_approach

            r_prox = 0.0
            if d_curr < 5.0:
                r_prox = 1.0 * (1.0 - d_curr / 5.0)
            reward += r_prox

            speed = np.linalg.norm(drone.velocity)
            r_speed = 0.0
            if d_curr < 10.0:
                r_speed = -0.2 * (speed / drone.max_speed) * (1.0 - d_curr / 10.0)
            reward += r_speed

            idx = int(agent.split("_")[1])
            assigned_angle = self._target_angle + np.pi + 2 * np.pi * idx / self.n_drones
            assigned_dir = np.array([np.cos(assigned_angle), 0.0, np.sin(assigned_angle)])
            from_target = drone.position - self.target_pos
            from_target_xz = np.array([from_target[0], 0.0, from_target[2]])
            xz_norm = np.linalg.norm(from_target_xz)
            r_sector = 0.0
            alignment = 0.0
            if xz_norm > 1e-6 and d_curr < 30.0:
                sector_dir = from_target_xz / xz_norm
                alignment = np.dot(sector_dir, assigned_dir)
                closeness = max(0.0, 1.0 - d_curr / 30.0)
                r_sector = 0.3 * closeness * max(-0.3, alignment)
            reward += r_sector

            # Terminal: hit target
            if d_curr < self.r_hit:
                approach_vec = drone.position - self.target_pos
                norm = np.linalg.norm(approach_vec)
                approach_vec = approach_vec / max(norm, 1e-9)
                if self.hit_angles:
                    dots = [abs(np.dot(approach_vec, h)) for h in self.hit_angles]
                    angle_bonus = np.arccos(np.clip(min(dots), -1, 1)) / np.pi
                else:
                    angle_bonus = 1.0
                reward += 30.0 + 10.0 * angle_bonus
                self.hit_angles.append(approach_vec)
                terminated = True

            # Terminal: collision
            elif agent in collision_set:
                reward -= 5.0
                terminated = True

            # Terminal: ground crash
            elif drone.position[1] <= 0.0:
                reward -= 5.0
                terminated = True

            # Terminal: out of bounds
            elif np.linalg.norm(drone.position) > self.r_bounds:
                reward -= 5.0
                terminated = True

            # Truncation: max steps — penalize for not hitting
            if not terminated and self.step_count >= self.max_steps:
                truncated = True
                reward -= 3.0

            self.prev_dists[agent] = d_curr
            observations[agent] = self._get_obs(agent)
            rewards[agent] = float(reward)
            terminations[agent] = terminated
            truncations[agent] = truncated
            nn_dist = self._nearest_neighbor_dist(agent)
            infos[agent] = {
                "distance_to_target": d_curr,
                "nn_dist": nn_dist,
                "alignment": alignment,
                "r_approach": r_approach,
                "r_prox": r_prox,
                "r_speed": r_speed,
                "r_sector": r_sector,
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
        obs[6:9] = (self.target_pos - drone.position) / self.r_bounds

        neighbors = self._get_k_nearest(agent)
        for j, (rel_pos, rel_vel) in enumerate(neighbors):
            obs[9 + j * 3: 9 + j * 3 + 3] = rel_pos / self.r_bounds
            base = 9 + self.k_neighbors * 3
            obs[base + j * 3: base + j * 3 + 3] = rel_vel / drone.max_speed

        idx = int(agent.split("_")[1])
        assigned_angle = self._target_angle + np.pi + 2 * np.pi * idx / self.n_drones
        obs[-3:] = np.array([np.cos(assigned_angle), 0.0, np.sin(assigned_angle)])

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

    def _min_approach_angle(self, agent):
        drone = self.drones[agent]
        u = drone.position - self.target_pos
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-9:
            return np.pi
        u = u / u_norm
        min_angle = np.pi
        for other_name in self.agents:
            if other_name == agent:
                continue
            v = self.drones[other_name].position - self.target_pos
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-9:
                continue
            v = v / v_norm
            angle = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
            min_angle = min(min_angle, angle)
        return min_angle

    def render(self):
        pass

    def close(self):
        pass
