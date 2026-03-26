import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

SUN_DIR = np.array([0.4, 0.8, 0.3])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)

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

SUN_DISTANCE = 150.0


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
    sx = cam_x + SUN_DIR[0] * SUN_DISTANCE
    sy = cam_y + SUN_DIR[1] * SUN_DISTANCE
    sz = cam_z + SUN_DIR[2] * SUN_DISTANCE

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
