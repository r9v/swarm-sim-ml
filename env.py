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
        spawn_radius=3.0,
        dt=0.05,
        max_steps=1000,
        r_hit=0.5,
        r_collision=0.3,
        r_engage=10.0,
        r_bounds=50.0,
        render_mode=None,
    ):
        super().__init__()
        self.n_drones = n_drones
        self.k_neighbors = k_neighbors
        self.target_pos = np.array(target_pos or [30.0, 5.0, 0.0], dtype=np.float64)
        self.spawn_center = np.array(spawn_center or [0.0, 5.0, 0.0], dtype=np.float64)
        self.spawn_radius = spawn_radius
        self.dt = dt
        self.max_steps = max_steps
        self.r_hit = r_hit
        self.r_collision = r_collision
        self.r_engage = r_engage
        self.r_bounds = r_bounds
        self.render_mode = render_mode

        self.possible_agents = [f"drone_{i}" for i in range(n_drones)]
        self.obs_dim = 6 + 3 + k_neighbors * 6

        self._obs_space = Box(low=-2.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32)
        self._act_space = Box(low=-3.0, high=3.0, shape=(3,), dtype=np.float32)

        self.drones: dict[str, Drone] = {}
        self.init_dists: dict[str, float] = {}
        self.prev_dists: dict[str, float] = {}
        self.hit_angles: list[np.ndarray] = []
        self.step_count = 0
        self._renderer_initialized = False
        self._cam_yaw = 0.0
        self._cam_pitch = 15.0
        self._cam_dist = 30.0
        self._mouse_captured = True
        self._quit_requested = False

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

            # Approach shaping
            reward += 5.0 * (d_prev - d_curr) / max(d_init, 1e-6)

            # Angular spread (within engage radius)
            if d_curr < self.r_engage:
                min_angle = self._min_approach_angle(agent)
                reward += 3.0 * max(0.0, min_angle / np.pi - 0.3)

            # Energy penalty
            act = np.array(actions.get(agent, np.zeros(3)), dtype=np.float64)
            reward -= 0.001 * (np.dot(act, act) / (3.0 ** 2))

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
                reward += 10.0 + 5.0 * angle_bonus
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

            # Truncation: max steps
            if not terminated and self.step_count >= self.max_steps:
                truncated = True

            self.prev_dists[agent] = d_curr
            observations[agent] = self._get_obs(agent)
            rewards[agent] = float(reward)
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {"distance_to_target": d_curr}

            if terminated or truncated:
                newly_dead.add(agent)

        self.agents = [a for a in self.agents if a not in newly_dead]

        if self.render_mode == "human":
            self.render()

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
        if self.render_mode != "human":
            return
        if not self._renderer_initialized:
            self._init_renderer()
        self._render_frame()

    def _init_renderer(self):
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL
        from OpenGL.GL import (
            glEnable, glBlendFunc, GL_DEPTH_TEST, GL_BLEND,
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        )
        from viewer import build_ground_list, build_grid_list

        pygame.init()
        self._display = (1280, 900)
        pygame.display.set_mode(self._display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Swarm Target Env")
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._ground_dl = build_ground_list()
        self._grid_dl = build_grid_list()
        self._clock = pygame.time.Clock()
        self._sim_time = 0.0
        self._renderer_initialized = True
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()

    def _render_frame(self):
        import pygame
        from OpenGL.GL import (
            glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
            glMatrixMode, glLoadIdentity, GL_PROJECTION, GL_MODELVIEW,
            glCallList,
        )
        from OpenGL.GLU import gluPerspective, gluLookAt
        from viewer import (
            draw_sky_gradient, draw_sun, draw_axes, draw_drone,
            draw_trail, draw_target, DRONE_COLORS, TRAIL_COLORS,
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_requested = True
                self.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit_requested = True
                    self.close()
                    return
                if event.key == pygame.K_UP:
                    self._cam_dist = max(5.0, self._cam_dist - 5.0)
                if event.key == pygame.K_DOWN:
                    self._cam_dist = min(80.0, self._cam_dist + 5.0)

        if self._mouse_captured:
            dx, dy = pygame.mouse.get_rel()
            self._cam_yaw -= dx * 0.15
            self._cam_pitch = np.clip(self._cam_pitch + dy * 0.15, -85, 89)

        self._sim_time += self.dt

        all_drones = list(self.drones.values())
        if not all_drones:
            return

        centroid = np.mean([d.position for d in all_drones], axis=0)

        yaw_rad = np.radians(self._cam_yaw)
        pitch_rad = np.radians(self._cam_pitch)
        cam_offset = np.array([
            -np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad),
            np.cos(yaw_rad) * np.cos(pitch_rad),
        ]) * self._cam_dist
        cam_pos = centroid + cam_offset

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self._display[0] / self._display[1], 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            cam_pos[0], cam_pos[1], cam_pos[2],
            centroid[0], centroid[1], centroid[2],
            0, 1, 0,
        )

        draw_sky_gradient()
        draw_sun(cam_pos[0], cam_pos[1], cam_pos[2])
        glCallList(self._ground_dl)
        glCallList(self._grid_dl)
        draw_axes()
        draw_target(self.target_pos)

        for i, name in enumerate(self.possible_agents):
            if name not in self.drones:
                continue
            drone = self.drones[name]
            color = DRONE_COLORS[i % len(DRONE_COLORS)]
            trail_color = TRAIL_COLORS[i % len(TRAIL_COLORS)]
            draw_drone(drone.position, drone.velocity, self._sim_time, color)

        pygame.display.flip()
        self._clock.tick(60)

    def close(self):
        if self._renderer_initialized:
            import pygame
            pygame.quit()
            self._renderer_initialized = False
