import numpy as np
import pygame
from OpenGL.GL import *

from .ctx import ViewerCtx, MOUSE_SENS_MIN, MOUSE_SENS_RANGE


def draw_hud(surface, drones, font, ctx):
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    w = surface.get_width()

    y_offset = 10
    for i, (drone, _trail) in enumerate(drones):
        speed = np.linalg.norm(drone.velocity)
        lines = [
            f"Drone {i}: pos=({drone.position[0]:+.1f}, {drone.position[1]:+.1f}, {drone.position[2]:+.1f})",
            f"  speed={speed:.2f} m/s",
        ]
        for line in lines:
            text = font.render(line, True, (200, 220, 255))
            overlay.blit(text, (10, y_offset))
            y_offset += 18
        y_offset += 4

    slider_y = surface.get_height() - 60
    slider_x = 10
    slider_w = 160
    label = font.render(f"Mouse sens: {ctx.mouse_sens:.2f}", True, (200, 200, 200))
    overlay.blit(label, (slider_x, slider_y - 18))
    pygame.draw.rect(overlay, (80, 80, 80), (slider_x, slider_y, slider_w, 6), border_radius=3)
    knob_t = (ctx.mouse_sens - MOUSE_SENS_MIN) / MOUSE_SENS_RANGE
    knob_x = slider_x + int(knob_t * slider_w)
    pygame.draw.circle(overlay, (220, 220, 220), (knob_x, slider_y + 3), 7)

    fps = ctx.clock.get_fps()
    fps_text = font.render(f"FPS: {fps:.0f}", True, (200, 200, 100))
    overlay.blit(fps_text, (w - 80, 10))

    cam_x, cam_y, cam_z = ctx.cam_pos
    cam_lines = [
        f"Cam: ({cam_x:+.1f}, {cam_y:+.1f}, {cam_z:+.1f})",
        f"Yaw: {ctx.cam_yaw:.1f}\u00b0  Pitch: {ctx.cam_pitch:.1f}\u00b0",
    ]
    cy = 30
    for line in cam_lines:
        t = font.render(line, True, (180, 220, 180))
        overlay.blit(t, (w - t.get_width() - 10, cy))
        cy += 18

    follow_text = "[F] Follow: ON" if ctx.follow_mode else "[F] Follow: OFF"
    help_text = f"[WASD] Move  [Space/Shift] Up/Down  {follow_text}  [ESC] Release mouse / Quit"
    text = font.render(help_text, True, (150, 150, 150))
    overlay.blit(text, (10, surface.get_height() - 25))

    surface.blit(overlay, (0, 0))


def draw_eval_hud(surface, drones_list, font, ctx, step_idx, total_steps, paused, episode_num):
    draw_hud(surface, drones_list, font, ctx)
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    w, h = surface.get_size()

    bar_x, bar_y, bar_w, bar_h = 200, h - 55, w - 220, 10
    pygame.draw.rect(overlay, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
    if total_steps > 0:
        fill_w = int(bar_w * step_idx / max(total_steps, 1))
        pygame.draw.rect(overlay, (0, 180, 220), (bar_x, bar_y, fill_w, bar_h), border_radius=4)
    pygame.draw.rect(overlay, (120, 120, 120), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)

    state_text = "PAUSED" if paused else "PLAYING"
    label = font.render(f"Ep {episode_num}  Step {step_idx}/{total_steps}  [{state_text}]", True, (200, 200, 200))
    overlay.blit(label, (bar_x, bar_y - 16))

    ctrl = font.render("[R] Reset  [Q] Pause  [</>] Step  Click timeline to scrub", True, (150, 150, 150))
    overlay.blit(ctrl, (bar_x, bar_y + 14))

    surface.blit(overlay, (0, 0))


def blit_hud_texture(hud_surface, hud_tex, display):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], display[1], 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    tex_data = pygame.image.tostring(hud_surface, "RGBA", True)
    glBindTexture(GL_TEXTURE_2D, hud_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, display[0], display[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)

    glEnable(GL_TEXTURE_2D)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(0, 0)
    glTexCoord2f(1, 1); glVertex2f(display[0], 0)
    glTexCoord2f(1, 0); glVertex2f(display[0], display[1])
    glTexCoord2f(0, 0); glVertex2f(0, display[1])
    glEnd()
    glDisable(GL_TEXTURE_2D)

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
