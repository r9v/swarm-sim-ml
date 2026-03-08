import numpy as np


class Drone:
    """3D point-mass drone with double-integrator dynamics.

    State: [x, y, z, vx, vy, vz]
    Action: [ax, ay, az] (acceleration command)
    """

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        max_speed: float = 5.0,
        max_accel: float = 6.0,
        drag: float = 0.05,
    ):
        self.position = np.array(position, dtype=np.float64)
        assert self.position.shape == (3,)
        self.velocity = np.array(velocity if velocity is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.drag = drag

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity])

    def step(self, action: np.ndarray, dt: float = 0.05) -> np.ndarray:
        accel = np.array(action, dtype=np.float64)

        accel_mag = np.linalg.norm(accel)
        if accel_mag > self.max_accel:
            accel = accel * (self.max_accel / accel_mag)

        accel = accel - self.drag * self.velocity

        self.velocity = self.velocity + accel * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity * (self.max_speed / speed)

        self.position = self.position + self.velocity * dt

        return self.state

    def reset(self, position: np.ndarray, velocity: np.ndarray | None = None):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity if velocity is not None else [0.0, 0.0, 0.0], dtype=np.float64)
