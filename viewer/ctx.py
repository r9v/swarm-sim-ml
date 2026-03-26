import numpy as np
import pygame
from dataclasses import dataclass, field

DISPLAY_SIZE = (1280, 900)
MOUSE_SENS_MIN = 0.5
MOUSE_SENS_RANGE = 19.5
PITCH_LIMITS = (-85, 89)
MOVE_SPEED = 0.3


@dataclass
class ViewerCtx:
    display: tuple
    hud_surface: pygame.Surface
    font: pygame.font.Font
    ground_dl: int
    grid_dl: int
    hud_tex: int
    clock: pygame.time.Clock
    cam_yaw: float = 0.0
    cam_pitch: float = 15.0
    cam_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 25.0, 60.0]))
    follow_mode: bool = False
    mouse_sens: float = 2.5
    mouse_captured: bool = True
    slider_rect: pygame.Rect = None
    dragging_slider: bool = False
    sim_time: float = 0.0

    def __post_init__(self):
        if self.slider_rect is None:
            self.slider_rect = pygame.Rect(10, self.display[1] - 60, 160, 12)
