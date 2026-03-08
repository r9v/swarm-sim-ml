import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from drone import Drone


SUN_DIR = np.array([0.4, 0.8, 0.3])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)
SUN_COLOR = (1.0, 0.95, 0.8)

DRONE_COLORS = [
    (1.0, 0.3, 0.3),
    (0.3, 0.6, 1.0),
    (0.3, 1.0, 0.4),
    (1.0, 0.8, 0.2),
    (0.8, 0.3, 1.0),
]

TRAIL_COLORS = [
    (1.0, 0.4, 0.4),
    (0.4, 0.6, 1.0),
    (0.4, 1.0, 0.5),
    (1.0, 0.9, 0.3),
    (0.9, 0.4, 1.0),
]


# --- Display list builders ---

def build_ground_list(size=30):
    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)
    tile = 2.0
    glBegin(GL_QUADS)
    for x in range(-size, size):
        for z in range(-size, size):
            if (x + z) % 2 == 0:
                glColor3f(0.15, 0.22, 0.12)
            else:
                glColor3f(0.12, 0.18, 0.10)
            glVertex3f(x * tile, -0.01, z * tile)
            glVertex3f((x + 1) * tile, -0.01, z * tile)
            glVertex3f((x + 1) * tile, -0.01, (z + 1) * tile)
            glVertex3f(x * tile, -0.01, (z + 1) * tile)
    glEnd()
    glEndList()
    return dl


def build_grid_list(size=20, spacing=1.0):
    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)
    glColor3f(0.25, 0.35, 0.2)
    glBegin(GL_LINES)
    for i in range(-size, size + 1):
        glVertex3f(i * spacing, 0, -size * spacing)
        glVertex3f(i * spacing, 0, size * spacing)
        glVertex3f(-size * spacing, 0, i * spacing)
        glVertex3f(size * spacing, 0, i * spacing)
    glEnd()
    glEndList()
    return dl


# --- 3D drawing ---

def draw_sky_gradient():
    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glBegin(GL_QUADS)
    glColor3f(0.15, 0.35, 0.65)
    glVertex3f(-1, 1, -0.999)
    glVertex3f(1, 1, -0.999)
    glColor3f(0.55, 0.7, 0.85)
    glVertex3f(1, -1, -0.999)
    glVertex3f(-1, -1, -0.999)
    glEnd()

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glEnable(GL_DEPTH_TEST)


def draw_sun(cam_x, cam_y, cam_z):
    sun_dist = 150.0
    sx = cam_x + SUN_DIR[0] * sun_dist
    sy = cam_y + SUN_DIR[1] * sun_dist
    sz = cam_z + SUN_DIR[2] * sun_dist

    glPushMatrix()
    glTranslatef(sx, sy, sz)

    for radius, alpha in [(8.0, 0.08), (5.0, 0.15), (3.0, 0.3)]:
        glColor4f(1.0, 0.95, 0.7, alpha)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        for a in range(0, 370, 10):
            rad = np.radians(a)
            glVertex3f(radius * np.cos(rad), radius * np.sin(rad), 0)
        glEnd()

    glColor3f(1.0, 1.0, 0.9)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(0, 0, 0)
    for a in range(0, 370, 10):
        rad = np.radians(a)
        glVertex3f(1.5 * np.cos(rad), 1.5 * np.sin(rad), 0)
    glEnd()

    glPopMatrix()


def draw_target(pos, radius=0.5):
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    glColor3f(1.0, 0.2, 0.1)
    glLineWidth(3)
    glBegin(GL_LINES)
    for axis in [(radius * 2, 0, 0), (0, radius * 2, 0), (0, 0, radius * 2)]:
        glVertex3f(-axis[0], -axis[1], -axis[2])
        glVertex3f(axis[0], axis[1], axis[2])
    glEnd()
    glColor4f(1.0, 0.5, 0.0, 0.6)
    glBegin(GL_LINE_LOOP)
    for a in range(0, 360, 10):
        rad = np.radians(a)
        glVertex3f(radius * np.cos(rad), 0, radius * np.sin(rad))
    glEnd()
    glLineWidth(1)
    glPopMatrix()


def draw_axes():
    glLineWidth(2)
    glBegin(GL_LINES)
    glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(2, 0, 0)
    glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, 2, 0)
    glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, 2)
    glEnd()
    glLineWidth(1)


def draw_drone(pos, velocity, sim_time, color=(1.0, 0.2, 0.2), size=0.15):
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])

    speed = np.linalg.norm(velocity[:2])
    if speed > 0.1:
        yaw = np.degrees(np.arctan2(velocity[0], velocity[2]))
        glRotatef(yaw, 0, 1, 0)
        pitch = np.clip(velocity[1] * 5, -20, 20)
        glRotatef(pitch, 1, 0, 0)

    s = size
    h = s * 0.3

    shade = max(0.4, np.dot(np.array([0, 1, 0]), SUN_DIR))
    glColor3f(color[0] * shade, color[1] * shade, color[2] * shade)
    glBegin(GL_QUADS)
    glVertex3f(-s, h, -s * 0.6)
    glVertex3f(s, h, -s * 0.6)
    glVertex3f(s, h, s * 0.6)
    glVertex3f(-s, h, s * 0.6)
    glVertex3f(-s, -h, -s * 0.6)
    glVertex3f(s, -h, -s * 0.6)
    glVertex3f(s, -h, s * 0.6)
    glVertex3f(-s, -h, s * 0.6)
    for (a, b) in [
        ((-s, -h, -s*0.6), (s, -h, -s*0.6)),
        ((s, -h, -s*0.6), (s, -h, s*0.6)),
        ((s, -h, s*0.6), (-s, -h, s*0.6)),
        ((-s, -h, s*0.6), (-s, -h, -s*0.6)),
    ]:
        glVertex3f(a[0], a[1], a[2])
        glVertex3f(b[0], b[1], b[2])
        glVertex3f(b[0], h, b[2])
        glVertex3f(a[0], h, a[2])
    glEnd()

    glColor3f(1.0, 1.0, 0.2)
    glBegin(GL_TRIANGLES)
    glVertex3f(0, h + 0.01, s * 0.6)
    glVertex3f(-s * 0.15, h + 0.01, s * 0.3)
    glVertex3f(s * 0.15, h + 0.01, s * 0.3)
    glEnd()

    arm_len = size * 2.5
    glColor3f(0.6, 0.6, 0.6)
    glLineWidth(2)
    glBegin(GL_LINES)
    for dx, dz in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        glVertex3f(0, 0, 0)
        glVertex3f(dx * arm_len, 0, dz * arm_len)
    glEnd()
    glLineWidth(1)

    rotor_angle = (sim_time * 1200) % 360
    r = size * 0.7
    for i, (dx, dz) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        cx, cz = dx * arm_len, dz * arm_len
        glColor4f(0.2, 0.8, 0.2, 0.5)
        direction = 1 if (i % 2 == 0) else -1
        glPushMatrix()
        glTranslatef(cx, 0.03, cz)
        glRotatef(rotor_angle * direction, 0, 1, 0)
        glBegin(GL_TRIANGLES)
        for blade in range(2):
            a = np.radians(blade * 180)
            glVertex3f(0, 0, 0)
            glVertex3f(r * np.cos(a + 0.15), 0, r * np.sin(a + 0.15))
            glVertex3f(r * np.cos(a - 0.15), 0, r * np.sin(a - 0.15))
        glEnd()
        glColor3f(0.3, 0.7, 0.3)
        glBegin(GL_LINE_LOOP)
        for a in range(0, 360, 15):
            glVertex3f(r * np.cos(np.radians(a)), 0, r * np.sin(np.radians(a)))
        glEnd()
        glPopMatrix()

    glPopMatrix()


def draw_trail(trail, color=(0.2, 0.5, 1.0)):
    if len(trail) < 2:
        return
    glBegin(GL_LINE_STRIP)
    for i, p in enumerate(trail):
        alpha = i / len(trail)
        glColor4f(color[0], color[1], color[2], alpha)
        glVertex3f(p[0], p[1], p[2])
    glEnd()


# --- HUD ---

def draw_hud(surface, drones, font, mouse_sens, follow_mode, fps, cam_pos, cam_yaw, cam_pitch):
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

    # Sensitivity slider
    slider_y = surface.get_height() - 60
    slider_x = 10
    slider_w = 160
    label = font.render(f"Mouse sens: {mouse_sens:.2f}", True, (200, 200, 200))
    overlay.blit(label, (slider_x, slider_y - 18))
    pygame.draw.rect(overlay, (80, 80, 80), (slider_x, slider_y, slider_w, 6), border_radius=3)
    knob_t = (mouse_sens - 0.5) / 19.5
    knob_x = slider_x + int(knob_t * slider_w)
    pygame.draw.circle(overlay, (220, 220, 220), (knob_x, slider_y + 3), 7)

    # Top-right info
    fps_text = font.render(f"FPS: {fps:.0f}", True, (200, 200, 100))
    overlay.blit(fps_text, (w - 80, 10))

    cam_lines = [
        f"Cam: ({cam_pos[0]:+.1f}, {cam_pos[1]:+.1f}, {cam_pos[2]:+.1f})",
        f"Yaw: {cam_yaw:.1f}\u00b0  Pitch: {cam_pitch:.1f}\u00b0",
    ]
    cy = 30
    for line in cam_lines:
        t = font.render(line, True, (180, 220, 180))
        overlay.blit(t, (w - t.get_width() - 10, cy))
        cy += 18

    # Bottom help
    follow_text = "[F] Follow: ON" if follow_mode else "[F] Follow: OFF"
    help_text = f"[WASD] Move  [Space/Shift] Up/Down  {follow_text}  [ESC] Release mouse / Quit"
    text = font.render(help_text, True, (150, 150, 150))
    overlay.blit(text, (10, surface.get_height() - 25))

    surface.blit(overlay, (0, 0))


def draw_eval_hud(surface, drones_list, font, mouse_sens, follow_mode, fps, cam_pos, cam_yaw, cam_pitch,
                  step_idx, total_steps, paused, episode_num):
    draw_hud(surface, drones_list, font, mouse_sens, follow_mode, fps, cam_pos, cam_yaw, cam_pitch)
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    w, h = surface.get_size()

    # Timeline bar
    bar_x, bar_y, bar_w, bar_h = 200, h - 55, w - 220, 10
    pygame.draw.rect(overlay, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
    if total_steps > 0:
        fill_w = int(bar_w * step_idx / max(total_steps, 1))
        pygame.draw.rect(overlay, (0, 180, 220), (bar_x, bar_y, fill_w, bar_h), border_radius=4)
    pygame.draw.rect(overlay, (120, 120, 120), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)

    # Step label
    state_text = "PAUSED" if paused else "PLAYING"
    label = font.render(f"Ep {episode_num}  Step {step_idx}/{total_steps}  [{state_text}]", True, (200, 200, 200))
    overlay.blit(label, (bar_x, bar_y - 16))

    # Controls help
    ctrl = font.render("[R] Reset  [Q] Pause  [</>] Step  Click timeline to scrub", True, (150, 150, 150))
    overlay.blit(ctrl, (bar_x, bar_y + 14))

    surface.blit(overlay, (0, 0))


# --- HUD texture blit ---

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


# --- Input handling ---

def handle_events(events, mouse_captured, follow_mode, dragging_slider, slider_rect, mouse_sens, cam_yaw, cam_pitch, display):
    running = True
    for event in events:
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            mouse_captured = not mouse_captured
            if mouse_captured:
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
                pygame.mouse.get_rel()
            else:
                pygame.event.set_grab(False)
                pygame.mouse.set_pos(display[0] // 2, display[1] // 2)
                pygame.mouse.set_visible(True)
        elif event.type == KEYDOWN and event.key == K_f:
            follow_mode = not follow_mode
        elif event.type == MOUSEBUTTONDOWN and not mouse_captured:
            if event.button == 1 and slider_rect.collidepoint(event.pos):
                dragging_slider = True
                t = (event.pos[0] - slider_rect.x) / slider_rect.width
                mouse_sens = 0.5 + np.clip(t, 0, 1) * 19.5
        elif event.type == MOUSEBUTTONUP and not mouse_captured:
            if event.button == 1:
                dragging_slider = False
        elif event.type == MOUSEMOTION and not mouse_captured:
            if dragging_slider:
                t = (event.pos[0] - slider_rect.x) / slider_rect.width
                mouse_sens = 0.5 + np.clip(t, 0, 1) * 19.5

    if mouse_captured:
        dx, dy = pygame.mouse.get_rel()
        cam_yaw -= dx * mouse_sens * 0.02
        cam_pitch = np.clip(cam_pitch + dy * mouse_sens * 0.02, -85, 89)

    return running, mouse_captured, follow_mode, dragging_slider, mouse_sens, cam_yaw, cam_pitch


def handle_movement(cam_pos, cam_yaw, cam_pitch):
    keys = pygame.key.get_pressed()
    move_speed = 0.3
    yaw_rad = np.radians(cam_yaw)
    pitch_rad = np.radians(cam_pitch)
    forward = np.array([
        np.cos(pitch_rad) * np.sin(yaw_rad),
        np.sin(pitch_rad),
        np.cos(pitch_rad) * np.cos(yaw_rad),
    ])
    right = np.array([np.cos(yaw_rad), 0, -np.sin(yaw_rad)])
    if keys[K_w]:
        cam_pos -= forward * move_speed
    if keys[K_s]:
        cam_pos += forward * move_speed
    if keys[K_a]:
        cam_pos -= right * move_speed
    if keys[K_d]:
        cam_pos += right * move_speed
    if keys[K_SPACE]:
        cam_pos[1] += move_speed
    if keys[K_LSHIFT]:
        cam_pos[1] = max(0.0, cam_pos[1] - move_speed)


# --- Simulation ---

def step_drones(drones, sim_time, dt):
    n = len(drones)
    for i, (drone, trail) in enumerate(drones):
        phase = 2 * np.pi * i / n
        ax = 2.0 * np.cos((0.3 + 0.1 * i) * sim_time + phase)
        ay = 1.2 * np.sin((0.2 + 0.05 * i) * sim_time + phase) + 0.3
        az = 2.0 * np.sin((0.4 + 0.08 * i) * sim_time + phase)
        drone.step(np.array([ax, ay, az]), dt)
        trail.append(drone.position.copy())
        if len(trail) > 400:
            trail.pop(0)


# --- Camera ---

def compute_look_target(cam_pos, cam_yaw, cam_pitch):
    yaw_rad = np.radians(cam_yaw)
    pitch_rad = np.radians(cam_pitch)
    look_x = cam_pos[0] - np.cos(pitch_rad) * np.sin(yaw_rad)
    look_y = cam_pos[1] - np.sin(pitch_rad)
    look_z = cam_pos[2] - np.cos(pitch_rad) * np.cos(yaw_rad)
    return look_x, look_y, look_z


def apply_follow_mode(cam_pos, drones, cam_yaw, cam_pitch):
    centroid = np.mean([d.position for d, _ in drones], axis=0)
    look_dir = centroid - cam_pos
    cam_yaw = np.degrees(np.arctan2(look_dir[0], look_dir[2]))
    horiz = np.sqrt(look_dir[0]**2 + look_dir[2]**2)
    cam_pitch = np.clip(np.degrees(np.arctan2(look_dir[1], horiz)), -85, 89)
    return cam_yaw, cam_pitch


# --- Viewer setup ---

def _init_viewer(title="Drone Swarm 3D Sim"):
    display = (1280, 900)
    pygame.init()
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
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

    return {
        "display": display,
        "hud_surface": pygame.Surface(display, pygame.SRCALPHA),
        "font": pygame.font.SysFont("consolas", 14),
        "ground_dl": build_ground_list(),
        "grid_dl": build_grid_list(),
        "hud_tex": hud_tex,
        "clock": pygame.time.Clock(),
        "cam_yaw": 0.0,
        "cam_pitch": 15.0,
        "cam_pos": np.array([0.0, 25.0, 60.0]),
        "follow_mode": False,
        "mouse_sens": 2.5,
        "mouse_captured": True,
        "slider_rect": pygame.Rect(10, display[1] - 60, 160, 12),
        "dragging_slider": False,
        "sim_time": 0.0,
    }


def _render_frame(ctx, drones_list, target_pos=None):
    display = ctx["display"]

    if ctx["follow_mode"] and drones_list:
        ctx["cam_yaw"], ctx["cam_pitch"] = apply_follow_mode(
            ctx["cam_pos"], drones_list, ctx["cam_yaw"], ctx["cam_pitch"])

    if ctx["cam_pos"][1] < 0.3:
        ctx["cam_pos"][1] = 0.3

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, display[0] / display[1], 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    cam_x, cam_y, cam_z = ctx["cam_pos"]
    look_x, look_y, look_z = compute_look_target(ctx["cam_pos"], ctx["cam_yaw"], ctx["cam_pitch"])
    up_y = 1.0 if ctx["cam_pitch"] > -89 else -1.0
    gluLookAt(cam_x, cam_y, cam_z, look_x, look_y, look_z, 0, up_y, 0)

    draw_sky_gradient()
    draw_sun(cam_x, cam_y, cam_z)
    glCallList(ctx["ground_dl"])
    glCallList(ctx["grid_dl"])
    draw_axes()

    if target_pos is not None:
        draw_target(target_pos)

    for i, (drone, trail) in enumerate(drones_list):
        draw_trail(trail, TRAIL_COLORS[i % len(TRAIL_COLORS)])
        draw_drone(drone.position, drone.velocity, ctx["sim_time"], DRONE_COLORS[i % len(DRONE_COLORS)])

    ctx["hud_surface"].fill((0, 0, 0, 0))
    draw_hud(ctx["hud_surface"], drones_list, ctx["font"], ctx["mouse_sens"], ctx["follow_mode"],
             fps=ctx["clock"].get_fps(), cam_pos=(cam_x, cam_y, cam_z),
             cam_yaw=ctx["cam_yaw"], cam_pitch=ctx["cam_pitch"])
    blit_hud_texture(ctx["hud_surface"], ctx["hud_tex"], display)

    pygame.display.flip()
    ctx["clock"].tick(60)


def _handle_frame_input(ctx):
    running, ctx["mouse_captured"], ctx["follow_mode"], ctx["dragging_slider"], ctx["mouse_sens"], ctx["cam_yaw"], ctx["cam_pitch"] = handle_events(
        pygame.event.get(), ctx["mouse_captured"], ctx["follow_mode"], ctx["dragging_slider"],
        ctx["slider_rect"], ctx["mouse_sens"], ctx["cam_yaw"], ctx["cam_pitch"], ctx["display"],
    )
    handle_movement(ctx["cam_pos"], ctx["cam_yaw"], ctx["cam_pitch"])
    return running


# --- Main ---

def main():
    ctx = _init_viewer("Drone Swarm 3D Sim")

    n_drones = 5
    drones = []
    for i in range(n_drones):
        angle = 2 * np.pi * i / n_drones
        start_pos = [3 * np.cos(angle), 2.0, 3 * np.sin(angle)]
        drones.append((Drone(position=start_pos), []))

    running = True
    while running:
        running = _handle_frame_input(ctx)
        step_drones(drones, ctx["sim_time"], 0.05)
        ctx["sim_time"] += 0.05
        _render_frame(ctx, drones)

    pygame.quit()


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


def eval_loop(env, model):
    ctx = _init_viewer("Swarm Eval Viewer")
    obs, infos = env.reset()

    history = []
    step_idx = 0
    paused = False
    episode_num = 1
    scrubbing = False
    key_hold_timer = 0.0
    bar_x, bar_w = 200, ctx["display"][0] - 220
    bar_y, bar_h = ctx["display"][1] - 55, 10
    bar_rect = pygame.Rect(bar_x, bar_y, bar_w, bar_h)

    def _new_ep_log():
        return {
            "steps": 0,
            "outcomes": {},
            "death_step": {},
            "final_dist": {},
            "min_dist": {a: float("inf") for a in env.possible_agents},
            "total_reward": {a: 0.0 for a in env.possible_agents},
        }

    ep_log = _new_ep_log()

    def _snapshot():
        return {
            name: (env.drones[name].position.copy(), env.drones[name].velocity.copy())
            for name in env.possible_agents if name in env.drones
        }

    def _trails_at(idx):
        trail_len = min(idx + 1, 400)
        start = max(0, idx + 1 - trail_len)
        trails = {}
        for name in env.possible_agents:
            pts = []
            for frame in history[start:idx + 1]:
                if name in frame:
                    pts.append(frame[name][0])
            trails[name] = pts
        return trails

    history.append(_snapshot())
    target_pos = env.target_pos.copy()

    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == KEYDOWN and event.key == K_r:
                if ep_log["steps"] > 0:
                    _log_episode(episode_num, ep_log, env)
                obs, infos = env.reset()
                history = [_snapshot()]
                target_pos = env.target_pos.copy()
                step_idx = 0
                paused = False
                episode_num += 1
                ep_log = _new_ep_log()
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
            elif event.type == MOUSEBUTTONDOWN and not ctx["mouse_captured"]:
                if event.button == 1 and bar_rect.collidepoint(event.pos) and history:
                    scrubbing = True
                    t = (event.pos[0] - bar_x) / bar_w
                    step_idx = int(np.clip(t, 0, 1) * (len(history) - 1))
                    paused = True
            elif event.type == MOUSEBUTTONUP:
                scrubbing = False
            elif event.type == MOUSEMOTION and scrubbing:
                t = (event.pos[0] - bar_x) / bar_w
                step_idx = int(np.clip(t, 0, 1) * (len(history) - 1))

        running, ctx["mouse_captured"], ctx["follow_mode"], ctx["dragging_slider"], ctx["mouse_sens"], ctx["cam_yaw"], ctx["cam_pitch"] = handle_events(
            events, ctx["mouse_captured"], ctx["follow_mode"], ctx["dragging_slider"],
            ctx["slider_rect"], ctx["mouse_sens"], ctx["cam_yaw"], ctx["cam_pitch"], ctx["display"],
        )
        handle_movement(ctx["cam_pos"], ctx["cam_yaw"], ctx["cam_pitch"])

        if paused:
            dt = ctx["clock"].get_time() / 1000.0
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
            else:
                if env.agents:
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
                            print(f"    {a}: pos=({p[0]:+6.1f},{p[1]:+5.1f},{p[2]:+6.1f})  dist={d:5.1f}  spd={spd:4.1f}  nn={inf['nn_dist']:4.1f}  Δang={inf['min_ang_diff_deg']:.0f}°")
                            print(f"             r_app={inf['r_approach']:+.2f} r_ang={inf['r_angular']:+.2f} r_spr={inf['r_spread']:+.2f} r_loi={inf['r_loiter']:+.2f}  cumR={tr:+.1f}  act=({act[0]:+.1f},{act[1]:+.1f},{act[2]:+.1f})")

                    history.append(_snapshot())
                    step_idx = len(history) - 1
                else:
                    _log_episode(episode_num, ep_log, env)
                    obs, infos = env.reset()
                    history = [_snapshot()]
                    target_pos = env.target_pos.copy()
                    step_idx = 0
                    episode_num += 1
                    ep_log = _new_ep_log()

        frame = history[step_idx]
        trails = _trails_at(step_idx)
        drones_list = []
        for name in env.possible_agents:
            if name in frame:
                pos, vel = frame[name]
                d = Drone(position=pos.copy())
                d.velocity = vel.copy()
                drones_list.append((d, trails.get(name, [])))

        ctx["sim_time"] = step_idx * env.dt

        display = ctx["display"]
        if ctx["follow_mode"] and drones_list:
            ctx["cam_yaw"], ctx["cam_pitch"] = apply_follow_mode(
                ctx["cam_pos"], drones_list, ctx["cam_yaw"], ctx["cam_pitch"])
        if ctx["cam_pos"][1] < 0.3:
            ctx["cam_pos"][1] = 0.3

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, display[0] / display[1], 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        cam_x, cam_y, cam_z = ctx["cam_pos"]
        look_x, look_y, look_z = compute_look_target(ctx["cam_pos"], ctx["cam_yaw"], ctx["cam_pitch"])
        up_y = 1.0 if ctx["cam_pitch"] > -89 else -1.0
        gluLookAt(cam_x, cam_y, cam_z, look_x, look_y, look_z, 0, up_y, 0)

        draw_sky_gradient()
        draw_sun(cam_x, cam_y, cam_z)
        glCallList(ctx["ground_dl"])
        glCallList(ctx["grid_dl"])
        draw_axes()
        draw_target(target_pos)

        for i, (drone, trail) in enumerate(drones_list):
            draw_trail(trail, TRAIL_COLORS[i % len(TRAIL_COLORS)])
            draw_drone(drone.position, drone.velocity, ctx["sim_time"], DRONE_COLORS[i % len(DRONE_COLORS)])

        ctx["hud_surface"].fill((0, 0, 0, 0))
        draw_eval_hud(ctx["hud_surface"], drones_list, ctx["font"], ctx["mouse_sens"], ctx["follow_mode"],
                      fps=ctx["clock"].get_fps(), cam_pos=(cam_x, cam_y, cam_z),
                      cam_yaw=ctx["cam_yaw"], cam_pitch=ctx["cam_pitch"],
                      step_idx=step_idx, total_steps=len(history) - 1, paused=paused, episode_num=episode_num)
        blit_hud_texture(ctx["hud_surface"], ctx["hud_tex"], display)

        pygame.display.flip()
        ctx["clock"].tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
