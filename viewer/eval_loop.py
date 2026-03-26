import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from drone import Drone

from .ctx import ViewerCtx, DISPLAY_SIZE
from .draw import (
    build_ground_list, build_grid_list, draw_sky_gradient, draw_sun,
    draw_target, draw_axes, draw_drone, draw_trail,
    DRONE_COLORS, TRAIL_COLORS,
)
from .hud import draw_hud, draw_eval_hud, blit_hud_texture
from .input import handle_events, handle_movement, compute_look_target, apply_follow_mode

TRAIL_MAX_LEN = 400


def _init_viewer(title="Drone Swarm 3D Sim"):
    pygame.init()
    pygame.display.set_mode(DISPLAY_SIZE, DOUBLEBUF | OPENGL)
    pygame.display.set_caption(title)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    hud_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, hud_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    pygame.mouse.get_rel()

    return ViewerCtx(
        display=DISPLAY_SIZE,
        hud_surface=pygame.Surface(DISPLAY_SIZE, pygame.SRCALPHA),
        font=pygame.font.SysFont("consolas", 14),
        ground_dl=build_ground_list(),
        grid_dl=build_grid_list(),
        hud_tex=hud_tex,
        clock=pygame.time.Clock(),
    )


def _render_frame(ctx, drones_list, target_pos=None, hud_fn=None):
    if ctx.follow_mode and drones_list:
        apply_follow_mode(ctx, drones_list)

    if ctx.cam_pos[1] < 0.3:
        ctx.cam_pos[1] = 0.3

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, ctx.display[0] / ctx.display[1], 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    cam_x, cam_y, cam_z = ctx.cam_pos
    look_x, look_y, look_z = compute_look_target(ctx.cam_pos, ctx.cam_yaw, ctx.cam_pitch)
    up_y = 1.0 if ctx.cam_pitch > -89 else -1.0
    gluLookAt(cam_x, cam_y, cam_z, look_x, look_y, look_z, 0, up_y, 0)

    draw_sky_gradient()
    draw_sun(cam_x, cam_y, cam_z)
    glCallList(ctx.ground_dl)
    glCallList(ctx.grid_dl)
    draw_axes()

    if target_pos is not None:
        draw_target(target_pos)

    for i, (drone, trail) in enumerate(drones_list):
        draw_trail(trail, TRAIL_COLORS[i % len(TRAIL_COLORS)])
        draw_drone(drone.position, drone.velocity, ctx.sim_time, DRONE_COLORS[i % len(DRONE_COLORS)])

    ctx.hud_surface.fill((0, 0, 0, 0))
    if hud_fn:
        hud_fn(ctx.hud_surface)
    else:
        draw_hud(ctx.hud_surface, drones_list, ctx.font, ctx)
    blit_hud_texture(ctx.hud_surface, ctx.hud_tex, ctx.display)

    pygame.display.flip()
    ctx.clock.tick(60)


def _log_episode(episode_num, ep_log, env):
    print(f"\n{'='*70}")
    print(f"EPISODE {episode_num}  |  steps={ep_log['steps']}  |  target=({env.target_pos[0]:.1f}, {env.target_pos[1]:.1f}, {env.target_pos[2]:.1f})")
    print(f"{'='*70}")

    hits = [a for a, r in ep_log["outcomes"].items() if r == "hit"]
    collisions = [a for a, r in ep_log["outcomes"].items() if r == "collision"]
    crashes = [a for a, r in ep_log["outcomes"].items() if r == "crash"]
    oob = [a for a, r in ep_log["outcomes"].items() if r == "oob"]
    alive = [a for a in env.possible_agents if a not in ep_log["outcomes"]]

    print(f"  hits={len(hits)}  collisions={len(collisions)}  crashes={len(crashes)}  oob={len(oob)}  alive={len(alive)}")
    if hits:
        print(f"  hit drones: {', '.join(hits)}")
        for h in hits:
            print(f"    {h}: hit at step {ep_log['death_step'].get(h, '?')}, final_dist={ep_log['final_dist'].get(h, 0):.2f}")

    print(f"\n  {'drone':<10} {'outcome':<12} {'step':<6} {'final_d':<9} {'min_d':<9} {'total_r':<10}")
    print(f"  {'-'*10} {'-'*12} {'-'*6} {'-'*9} {'-'*9} {'-'*10}")
    for name in env.possible_agents:
        outcome = ep_log["outcomes"].get(name, "alive")
        step = ep_log["death_step"].get(name, ep_log["steps"])
        final_d = ep_log["final_dist"].get(name, 0)
        min_d = ep_log["min_dist"].get(name, 0)
        total_r = ep_log["total_reward"].get(name, 0)
        print(f"  {name:<10} {outcome:<12} {step:<6} {final_d:<9.2f} {min_d:<9.2f} {total_r:<10.2f}")

    total_r = sum(ep_log["total_reward"].values())
    print(f"\n  total_reward={total_r:.2f}  avg_per_drone={total_r/len(env.possible_agents):.2f}")
    print(f"{'='*70}\n")


def _new_ep_log(env):
    return {
        "steps": 0,
        "outcomes": {},
        "death_step": {},
        "final_dist": {},
        "min_dist": {a: float("inf") for a in env.possible_agents},
        "total_reward": {a: 0.0 for a in env.possible_agents},
    }


def _snapshot(env):
    return {
        name: (env.drones[name].position.copy(), env.drones[name].velocity.copy())
        for name in env.possible_agents if name in env.drones
    }


def _trails_at(history, env, idx):
    trail_len = min(idx + 1, TRAIL_MAX_LEN)
    start = max(0, idx + 1 - trail_len)
    trails = {}
    for name in env.possible_agents:
        pts = []
        for frame in history[start:idx + 1]:
            if name in frame:
                pts.append(frame[name][0])
        trails[name] = pts
    return trails


def _scrub_position(pos_x, bar_x, bar_w, max_idx):
    t = np.clip((pos_x - bar_x) / bar_w, 0, 1)
    return int(t * max_idx)


def eval_loop(env, model):
    ctx = _init_viewer("Swarm Eval Viewer")
    obs, infos = env.reset()

    history = [_snapshot(env)]
    step_idx = 0
    paused = False
    episode_num = 1
    scrubbing = False
    key_hold_timer = 0.0
    bar_x, bar_w = 200, ctx.display[0] - 220
    bar_y = ctx.display[1] - 55
    bar_rect = pygame.Rect(bar_x, bar_y, bar_w, 10)
    ep_log = _new_ep_log(env)
    target_pos = env.target_pos.copy()

    def _reset_episode():
        nonlocal obs, infos, history, target_pos, step_idx, paused, episode_num, ep_log
        if ep_log["steps"] > 0:
            _log_episode(episode_num, ep_log, env)
        obs, infos = env.reset()
        history = [_snapshot(env)]
        target_pos = env.target_pos.copy()
        step_idx = 0
        paused = False
        episode_num += 1
        ep_log = _new_ep_log(env)

    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == KEYDOWN and event.key == K_r:
                _reset_episode()
            elif event.type == KEYDOWN and event.key == K_q:
                paused = not paused
            elif event.type == KEYDOWN and event.key == K_RIGHT and paused:
                if step_idx < len(history) - 1:
                    step_idx += 1
                    key_hold_timer = 0.0
            elif event.type == KEYDOWN and event.key == K_LEFT and paused:
                if step_idx > 0:
                    step_idx -= 1
                    key_hold_timer = 0.0
            elif event.type == MOUSEBUTTONDOWN and not ctx.mouse_captured:
                if event.button == 1 and bar_rect.collidepoint(event.pos) and history:
                    scrubbing = True
                    step_idx = _scrub_position(event.pos[0], bar_x, bar_w, len(history) - 1)
                    paused = True
            elif event.type == MOUSEBUTTONUP:
                scrubbing = False
            elif event.type == MOUSEMOTION and scrubbing:
                step_idx = _scrub_position(event.pos[0], bar_x, bar_w, len(history) - 1)

        running = handle_events(ctx, events)
        handle_movement(ctx)

        if paused:
            dt = ctx.clock.get_time() / 1000.0
            keys = pygame.key.get_pressed()
            if keys[K_LEFT] or keys[K_RIGHT]:
                key_hold_timer += dt
                if key_hold_timer > 0.3:
                    step_speed = min(10, 1 + int((key_hold_timer - 0.3) * 8))
                    if keys[K_RIGHT]:
                        step_idx = min(step_idx + step_speed, len(history) - 1)
                    elif keys[K_LEFT]:
                        step_idx = max(step_idx - step_speed, 0)
            else:
                key_hold_timer = 0.0

        if not paused:
            if step_idx < len(history) - 1:
                step_idx += 1
            elif env.agents:
                actions = {}
                for agent in env.agents:
                    o = obs[agent].reshape(1, -1)
                    act, _ = model.predict(o, deterministic=True)
                    actions[agent] = act[0]
                obs, rewards, terms, truncs, infos = env.step(actions)
                ep_log["steps"] = env.step_count

                for agent in list(rewards.keys()):
                    ep_log["total_reward"][agent] = ep_log["total_reward"].get(agent, 0) + rewards[agent]
                    d = infos[agent].get("distance_to_target", 0)
                    ep_log["final_dist"][agent] = d
                    ep_log["min_dist"][agent] = min(ep_log["min_dist"].get(agent, float("inf")), d)

                    if terms.get(agent, False):
                        ep_log["death_step"][agent] = env.step_count
                        if d < env.r_hit:
                            ep_log["outcomes"][agent] = "hit"
                        elif env.drones[agent].position[1] <= 0:
                            ep_log["outcomes"][agent] = "crash"
                        elif np.linalg.norm(env.drones[agent].position) > env.r_bounds:
                            ep_log["outcomes"][agent] = "oob"
                        else:
                            ep_log["outcomes"][agent] = "collision"
                    elif truncs.get(agent, False):
                        ep_log["death_step"][agent] = env.step_count
                        ep_log["outcomes"][agent] = "timeout"

                if env.step_count % 100 == 0:
                    t = env.target_pos
                    print(f"  step {env.step_count:4d} | alive={len(env.agents)} | target=({t[0]:+6.1f},{t[1]:+5.1f},{t[2]:+6.1f})")
                    for a in sorted(rewards.keys()):
                        p = env.drones[a].position
                        spd = np.linalg.norm(env.drones[a].velocity)
                        inf = infos[a]
                        d = inf['distance_to_target']
                        tr = ep_log["total_reward"].get(a, 0)
                        act = actions.get(a, np.zeros(3))
                        print(f"    {a}: pos=({p[0]:+6.1f},{p[1]:+5.1f},{p[2]:+6.1f})  dist={d:5.1f}  spd={spd:4.1f}  nn={inf['nn_dist']:4.1f}  \u0394ang={inf['min_ang_diff_deg']:.0f}\u00b0")
                        print(f"             r_app={inf['r_approach']:+.2f} r_ang={inf['r_angular']:+.2f} r_spr={inf['r_spread']:+.2f} r_loi={inf['r_loiter']:+.2f}  cumR={tr:+.1f}  act=({act[0]:+.1f},{act[1]:+.1f},{act[2]:+.1f})")

                history.append(_snapshot(env))
                step_idx = len(history) - 1
            else:
                _reset_episode()

        frame = history[step_idx]
        trails = _trails_at(history, env, step_idx)
        drones_list = []
        for name in env.possible_agents:
            if name in frame:
                pos, vel = frame[name]
                d = Drone(position=pos.copy())
                d.velocity = vel.copy()
                drones_list.append((d, trails.get(name, [])))

        ctx.sim_time = step_idx * env.dt

        def _eval_hud(surface):
            draw_eval_hud(surface, drones_list, ctx.font, ctx,
                          step_idx=step_idx, total_steps=len(history) - 1,
                          paused=paused, episode_num=episode_num)

        _render_frame(ctx, drones_list, target_pos=target_pos, hud_fn=_eval_hud)

    pygame.quit()
