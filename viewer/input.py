import numpy as np
import pygame
from pygame.locals import *

from .ctx import ViewerCtx, MOUSE_SENS_MIN, MOUSE_SENS_RANGE, MOVE_SPEED, PITCH_LIMITS


def _slider_value(pos_x, rect_x, rect_w):
    t = np.clip((pos_x - rect_x) / rect_w, 0, 1)
    return MOUSE_SENS_MIN + t * MOUSE_SENS_RANGE


def handle_events(ctx, events):
    running = True
    for event in events:
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            ctx.mouse_captured = not ctx.mouse_captured
            if ctx.mouse_captured:
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
                pygame.mouse.get_rel()
            else:
                pygame.event.set_grab(False)
                pygame.mouse.set_pos(ctx.display[0] // 2, ctx.display[1] // 2)
                pygame.mouse.set_visible(True)
        elif event.type == KEYDOWN and event.key == K_f:
            ctx.follow_mode = not ctx.follow_mode
        elif event.type == MOUSEBUTTONDOWN and not ctx.mouse_captured:
            if event.button == 1 and ctx.slider_rect.collidepoint(event.pos):
                ctx.dragging_slider = True
                ctx.mouse_sens = _slider_value(event.pos[0], ctx.slider_rect.x, ctx.slider_rect.width)
        elif event.type == MOUSEBUTTONUP and not ctx.mouse_captured:
            if event.button == 1:
                ctx.dragging_slider = False
        elif event.type == MOUSEMOTION and not ctx.mouse_captured:
            if ctx.dragging_slider:
                ctx.mouse_sens = _slider_value(event.pos[0], ctx.slider_rect.x, ctx.slider_rect.width)

    if ctx.mouse_captured:
        dx, dy = pygame.mouse.get_rel()
        ctx.cam_yaw -= dx * ctx.mouse_sens * 0.02
        ctx.cam_pitch = np.clip(ctx.cam_pitch + dy * ctx.mouse_sens * 0.02, *PITCH_LIMITS)

    return running


def handle_movement(ctx):
    keys = pygame.key.get_pressed()
    yaw_rad = np.radians(ctx.cam_yaw)
    pitch_rad = np.radians(ctx.cam_pitch)
    forward = np.array([
        np.cos(pitch_rad) * np.sin(yaw_rad),
        np.sin(pitch_rad),
        np.cos(pitch_rad) * np.cos(yaw_rad),
    ])
    right = np.array([np.cos(yaw_rad), 0, -np.sin(yaw_rad)])
    if keys[K_w]:
        ctx.cam_pos -= forward * MOVE_SPEED
    if keys[K_s]:
        ctx.cam_pos += forward * MOVE_SPEED
    if keys[K_a]:
        ctx.cam_pos -= right * MOVE_SPEED
    if keys[K_d]:
        ctx.cam_pos += right * MOVE_SPEED
    if keys[K_SPACE]:
        ctx.cam_pos[1] += MOVE_SPEED
    if keys[K_LSHIFT]:
        ctx.cam_pos[1] = max(0.0, ctx.cam_pos[1] - MOVE_SPEED)


def compute_look_target(cam_pos, cam_yaw, cam_pitch):
    yaw_rad = np.radians(cam_yaw)
    pitch_rad = np.radians(cam_pitch)
    look_x = cam_pos[0] - np.cos(pitch_rad) * np.sin(yaw_rad)
    look_y = cam_pos[1] - np.sin(pitch_rad)
    look_z = cam_pos[2] - np.cos(pitch_rad) * np.cos(yaw_rad)
    return look_x, look_y, look_z


def apply_follow_mode(ctx, drones):
    centroid = np.mean([d.position for d, _ in drones], axis=0)
    look_dir = centroid - ctx.cam_pos
    ctx.cam_yaw = np.degrees(np.arctan2(look_dir[0], look_dir[2]))
    horiz = np.sqrt(look_dir[0]**2 + look_dir[2]**2)
    ctx.cam_pitch = np.clip(np.degrees(np.arctan2(look_dir[1], horiz)), *PITCH_LIMITS)
