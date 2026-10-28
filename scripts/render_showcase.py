"""Path-traced "digital twin" replay of a recorded badminton rollout.

Builds a sports-hall stage (regulation court, net, LED boards, overhead lighting), authors the
recorded robot / racket / shuttle poses as USD time samples, and renders it with the RTX path tracer
through Replicator (motion blur from sub-frame time samples, depth of field from the USD camera).
Physics never runs here: every pose comes from ``scripts/record_rollout.py``.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
        ../IsaacLab/.venv/bin/python scripts/render_showcase.py --rollout rollout.npz --out_dir showcase
    # quick look: one frame per shot, no motion blur
    ... --test
"""

import argparse
import json
import math
import os
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--rollout", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--episodes", type=int, nargs="*", default=None, help="rallies to show (default: successes)")
parser.add_argument("--max_rallies", type=int, default=6)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--spp", type=int, default=8, help="path-tracing samples per motion-blur sub-frame")
parser.add_argument("--subframes", type=int, default=16, help="motion-blur sub-frames per output frame")
parser.add_argument("--test", action="store_true", help="render a few stills per shot instead of the movie")
parser.add_argument("--renderer", default="PathTracing", choices=("PathTracing", "RealTimePathTracing"))
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": args.renderer, "width": args.width, "height": args.height})

import carb  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

# court geometry (BWF), matching the training env: net plane at x = NET_X, robot side towards +x
NET_X = -2.0
NET_HEIGHT = 1.55
HALF_LEN = 6.7
DOUBLES_Y = 3.05
SINGLES_Y = 2.59
SHORT_SERVICE = 1.98
LONG_SERVICE_DOUBLES = 0.76
LINE_W = 0.04
TARGET_RADIUS = 1.0  # env target_success_radius

TEX = os.path.join(args.out_dir, "textures")
os.makedirs(TEX, exist_ok=True)
rng = np.random.default_rng(3)

# ----------------------------------------------------------------------------------------------
# procedural textures
# ----------------------------------------------------------------------------------------------


def save_png(name, arr):
    path = os.path.join(TEX, name)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)
    return path


def wood_texture(n=2048, planks=16):
    """Maple sports floor: planks along u, per-plank tint, staggered butt joints, fine grain."""
    v, u = np.mgrid[0:n, 0:n] / n
    plank = np.floor(v * planks).astype(int)
    tint = rng.uniform(-0.06, 0.06, planks)[plank]
    offset = rng.uniform(0, 1, planks)[plank]
    grain = 0.03 * np.sin(2 * np.pi * (u * 40 + 3 * np.sin(2 * np.pi * (v * planks * 0.5 + offset))))
    grain += 0.02 * rng.standard_normal((n, n)).cumsum(axis=1) / np.sqrt(n)
    base = np.array([0.82, 0.62, 0.40])
    col = base[None, None, :] * (1 + tint + grain)[..., None]
    seam_v = (np.abs((v * planks) % 1) < 0.012)
    seam_u = np.abs(((u + offset) * 2) % 1) < 0.003
    col[seam_v | seam_u] *= 0.6
    return save_png("wood.png", col * 255)


def grid_alpha_texture(name, n, cells, line_px, rgb):
    """RGBA string grid: opaque strings, transparent holes (used for the net and the string bed)."""
    a = np.zeros((n, n), dtype=np.float32)
    step = n / cells
    for k in range(cells):
        c = int(k * step)
        a[:, c : c + line_px] = 1
        a[c : c + line_px, :] = 1
    img = np.zeros((n, n, 4))
    img[..., :3] = np.array(rgb) * 255
    img[..., 3] = a * 255
    return save_png(name, img)


def feather_texture(n=512, feathers=16):
    """Skirt of 16 overlapping feathers: vanes with a dark-ish rachis and thin gaps between them."""
    v, u = np.mgrid[0:n, 0:n] / n
    phase = (u * feathers) % 1
    alpha = np.where(np.abs(phase - 0.5) < 0.46, 0.92, 0.0)
    alpha *= np.where(v < 0.96, 1, 0)
    shade = 0.96 - 0.08 * np.abs(phase - 0.5) - 0.05 * (np.abs(phase - 0.5) < 0.02)
    img = np.zeros((n, n, 4))
    img[..., :3] = (shade * 255)[..., None] * np.array([1.0, 0.99, 0.96])
    img[..., 3] = alpha * 255
    return save_png("feathers.png", img)


def roller_texture(n=256, rollers=12):
    """Mecanum wheel tread: dark rollers set at 45 degrees with lighter gaps."""
    v, u = np.mgrid[0:n, 0:n] / n
    phase = (u * rollers + v) % 1
    shade = np.where(phase < 0.82, 0.07, 0.35) + 0.04 * np.sin(phase * np.pi)
    return save_png("rollers.png", np.repeat(shade[..., None], 3, axis=2) * 255)


def led_texture(n=1024):
    """Perimeter LED board: deep blue to teal gradient with a scanline pattern (no lettering)."""
    u = np.mgrid[0:n // 8, 0:n][1] / n
    v = np.mgrid[0:n // 8, 0:n][0] / (n // 8)
    a = np.array([0.02, 0.10, 0.45])
    b = np.array([0.00, 0.65, 0.70])
    w = (0.5 + 0.5 * np.sin(2 * np.pi * u * 3))[..., None]
    col = a * (1 - w) + b * w
    col *= (0.75 + 0.25 * (np.floor(v * 24) % 2))[..., None]
    stripe = (np.abs(v - 0.5) < 0.06)[..., None]
    col = np.where(stripe, np.array([0.95, 0.95, 1.0]), col)
    return save_png("led.png", col * 255)


# ----------------------------------------------------------------------------------------------
# USD helpers
# ----------------------------------------------------------------------------------------------

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
UsdGeom.Scope.Define(stage, "/World/Looks")


def material(name, color=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0, clearcoat=0.0, texture=None,
             uv_scale=(1.0, 1.0), opacity_texture=False, opacity=1.0, emissive=None, emissive_texture=None,
             texture_gain=1.0, opacity_threshold=0.5):
    """UsdPreviewSurface material; textures read through an `st` primvar."""
    path = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(clearcoat)
    sh.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.05)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    tex_file = texture or emissive_texture
    if tex_file:
        st = UsdShade.Shader.Define(stage, f"{path}/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        xf = UsdShade.Shader.Define(stage, f"{path}/uvScale")
        xf.CreateIdAttr("UsdTransform2d")
        xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*uv_scale))
        tx = UsdShade.Shader.Define(stage, f"{path}/Tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_file)
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf.ConnectableAPI(), "result")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(texture_gain, texture_gain, texture_gain, 1.0))
        rgb_out = tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        if emissive_texture:
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
        if opacity_texture:
            a_out = tx.CreateOutput("a", Sdf.ValueTypeNames.Float)
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(a_out)
            sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(opacity_threshold)
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim() if hasattr(prim, "GetPrim") else prim).Bind(mat)


def mesh(path, points, faces, uvs=None, mat=None, double_sided=False):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in faces]))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for f in faces for i in f]))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(double_sided)
    if uvs is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, uv)) for uv in uvs]))
    if mat is not None:
        bind(m, mat)
    return m


def quad(path, corner, edge_u, edge_v, mat=None, uv=(1.0, 1.0), double_sided=False):
    c, eu, ev = map(np.asarray, (corner, edge_u, edge_v))
    pts = [c, c + eu, c + eu + ev, c + ev]
    return mesh(path, pts, [[0, 1, 2, 3]], [(0, 0), (uv[0], 0), (uv[0], uv[1]), (0, uv[1])], mat, double_sided)


def floor_rect(path, x0, x1, y0, y1, z, mat, uv=(1.0, 1.0)):
    return quad(path, (x0, y0, z), (x1 - x0, 0, 0), (0, y1 - y0, 0), mat, uv)


def box(path, center, size, mat):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(cube, mat)
    return cube


def cylinder(path, center, radius, height, mat, axis="Z"):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(*center))
    bind(cyl, mat)
    return cyl


def sphere(path, center, radius, mat):
    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(radius)
    UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(*center))
    bind(s, mat)
    return s


def ellipse_tube(path, center_z, ax, az, tube, mat, seg=96, ring=12):
    """Racket frame: a tube swept around an ellipse in the x-z plane."""
    pts, faces = [], []
    for i in range(seg):
        t = 2 * math.pi * i / seg
        c = np.array([ax * math.cos(t), 0.0, center_z + az * math.sin(t)])
        n = np.array([math.cos(t) / ax, 0.0, math.sin(t) / az])
        n /= np.linalg.norm(n)
        b = np.array([0.0, 1.0, 0.0])
        for j in range(ring):
            p = 2 * math.pi * j / ring
            pts.append(c + tube * (math.cos(p) * n + math.sin(p) * b))
    for i in range(seg):
        for j in range(ring):
            a, b2 = i * ring + j, ((i + 1) % seg) * ring + j
            faces.append([a, b2, ((i + 1) % seg) * ring + (j + 1) % ring, i * ring + (j + 1) % ring])
    return mesh(path, pts, faces, mat=mat)


def ellipse_disc(path, center_z, ax, az, mat, seg=96):
    pts = [(0.0, 0.0, center_z)] + [(ax * math.cos(2 * math.pi * i / seg), 0.0, center_z + az * math.sin(2 * math.pi * i / seg)) for i in range(seg)]
    uvs = [(0.5, 0.5)] + [(0.5 + 0.5 * math.cos(2 * math.pi * i / seg), 0.5 + 0.5 * az / ax * math.sin(2 * math.pi * i / seg)) for i in range(seg)]
    faces = [[0, 1 + i, 1 + (i + 1) % seg] for i in range(seg)]
    return mesh(path, pts, faces, uvs, mat, double_sided=True)


def frustum(path, z0, r0, z1, r1, mat, seg=48):
    """Open cone (shuttle skirt) from (z0, r0) to (z1, r1) with u around, v along."""
    pts, uvs, faces = [], [], []
    for i in range(seg + 1):
        t = 2 * math.pi * i / seg
        pts += [(r0 * math.cos(t), r0 * math.sin(t), z0), (r1 * math.cos(t), r1 * math.sin(t), z1)]
        uvs += [(i / seg, 0.0), (i / seg, 1.0)]
    for i in range(seg):
        faces.append([2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1])
    return mesh(path, pts, faces, uvs, mat, double_sided=True)


def wheel_tread(path, radius, width, mat, seg=64):
    """Open cylinder along y (the axle) with u around and v across, for the roller texture."""
    pts, uvs, faces = [], [], []
    for i in range(seg + 1):
        t = 2 * math.pi * i / seg
        for side, y in ((0, -width / 2), (1, width / 2)):
            pts.append((radius * math.cos(t), y, radius * math.sin(t)))
            uvs.append((i / seg, side))
    for i in range(seg):
        faces.append([2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2])
    return mesh(path, pts, faces, uvs, mat, double_sided=True)


def annulus(path, r_in, r_out, z, mat, seg=128):
    pts, faces = [], []
    for i in range(seg):
        t = 2 * math.pi * i / seg
        pts += [(r_in * math.cos(t), r_in * math.sin(t), z), (r_out * math.cos(t), r_out * math.sin(t), z)]
    for i in range(seg):
        j = (i + 1) % seg
        faces.append([2 * i, 2 * i + 1, 2 * j + 1, 2 * j])  # counter-clockwise from above: faces +z
    return mesh(path, pts, faces, mat=mat)


def net_mesh(path, x, y0, y1, z0, z1, mat, cell=0.02, strand=0.0015):
    """Badminton net as real string geometry: flat ribbons on a 2 cm square mesh in the x = const plane."""
    pts, faces = [], []

    def ribbon(a, b, c, d):
        k = len(pts)
        pts.extend([a, b, c, d])
        faces.append([k, k + 1, k + 2, k + 3])

    h = strand / 2
    for y in np.arange(y0, y1 + 1e-6, cell):
        ribbon((x, y - h, z0), (x, y + h, z0), (x, y + h, z1), (x, y - h, z1))
    for z in np.arange(z0, z1 + 1e-6, cell):
        ribbon((x, y0, z - h), (x, y1, z - h), (x, y1, z + h), (x, y0, z + h))
    return mesh(path, pts, faces, mat=mat, double_sided=True)


# ----------------------------------------------------------------------------------------------
# the hall
# ----------------------------------------------------------------------------------------------

wood = material("Wood", texture=wood_texture(), uv_scale=(10.0, 10.0), roughness=0.32, clearcoat=0.6)
court_green = material("CourtMat", color=(0.05, 0.30, 0.17), roughness=0.55, clearcoat=0.15)
line_white = material("Lines", color=(0.92, 0.92, 0.9), roughness=0.5)
wall_mat = material("Walls", color=(0.16, 0.20, 0.30), roughness=0.85)
wall_band = material("WallBand", color=(0.55, 0.56, 0.6), roughness=0.6)
ceiling_mat = material("Ceiling", color=(0.25, 0.25, 0.27), roughness=0.9)
fixture_mat = material("Fixture", color=(0.0, 0.0, 0.0), emissive=(6.0, 5.8, 5.4), roughness=0.5)
led = material("LED", emissive_texture=led_texture(), uv_scale=(4.0, 1.0), roughness=0.3, texture_gain=0.35)
board_back = material("BoardBack", color=(0.02, 0.02, 0.025), roughness=0.4)
post_mat = material("Post", color=(0.02, 0.18, 0.10), roughness=0.35, metallic=0.6)
net_mat = material("Net", color=(0.02, 0.02, 0.025), roughness=0.9)
tape_mat = material("NetTape", color=(0.95, 0.95, 0.95), roughness=0.6)
pedestal_mat = material("Pedestal", color=(0.12, 0.12, 0.13), roughness=0.35, metallic=0.8)
plate_mat = material("Plate", color=(0.75, 0.76, 0.78), roughness=0.22, metallic=1.0)

HALL_X, HALL_Y, HALL_Z = (-22.0, 18.0), (-13.0, 13.0), 12.0
floor_rect("/World/Hall/Floor", *HALL_X, *HALL_Y, 0.0, wood, uv=(4.0, 2.6))
floor_rect("/World/Hall/CourtMat", NET_X - HALF_LEN - 2.0, NET_X + HALF_LEN + 2.0, -DOUBLES_Y - 1.5, DOUBLES_Y + 1.5,
           0.002, court_green)
# walls (inward) and ceiling
x0, x1 = HALL_X
y0, y1 = HALL_Y
quad("/World/Hall/WallN", (x0, y1, 0), (x1 - x0, 0, 0), (0, 0, HALL_Z), wall_mat, double_sided=True)
quad("/World/Hall/WallS", (x0, y0, 0), (x1 - x0, 0, 0), (0, 0, HALL_Z), wall_mat, double_sided=True)
quad("/World/Hall/WallE", (x1, y0, 0), (0, y1 - y0, 0), (0, 0, HALL_Z), wall_mat, double_sided=True)
quad("/World/Hall/WallW", (x0, y0, 0), (0, y1 - y0, 0), (0, 0, HALL_Z), wall_mat, double_sided=True)
quad("/World/Hall/Ceiling", (x0, y0, HALL_Z), (x1 - x0, 0, 0), (0, y1 - y0, 0), ceiling_mat, double_sided=True)
# light-grey dado band around the hall at 2.5-3.1 m
for nm, c, eu in (("N", (x0, y1 - 0.01, 2.5), (x1 - x0, 0, 0)), ("S", (x0, y0 + 0.01, 2.5), (x1 - x0, 0, 0)),
                  ("E", (x1 - 0.01, y0, 2.5), (0, y1 - y0, 0)), ("W", (x0 + 0.01, y0, 2.5), (0, y1 - y0, 0))):
    quad(f"/World/Hall/Band{nm}", c, eu, (0, 0, 0.6), wall_band, double_sided=True)

# court lines, painted 2 mm above the mat
z_line = 0.004
xb0, xb1 = NET_X - HALF_LEN, NET_X + HALF_LEN
hw = LINE_W / 2
lines = [
    (xb0, xb1, DOUBLES_Y - LINE_W, DOUBLES_Y), (xb0, xb1, -DOUBLES_Y, -DOUBLES_Y + LINE_W),
    (xb0, xb1, SINGLES_Y - LINE_W, SINGLES_Y), (xb0, xb1, -SINGLES_Y, -SINGLES_Y + LINE_W),
    (xb0, xb0 + LINE_W, -DOUBLES_Y, DOUBLES_Y), (xb1 - LINE_W, xb1, -DOUBLES_Y, DOUBLES_Y),
    (xb0 + LONG_SERVICE_DOUBLES, xb0 + LONG_SERVICE_DOUBLES + LINE_W, -DOUBLES_Y, DOUBLES_Y),
    (xb1 - LONG_SERVICE_DOUBLES - LINE_W, xb1 - LONG_SERVICE_DOUBLES, -DOUBLES_Y, DOUBLES_Y),
    (NET_X - SHORT_SERVICE - LINE_W, NET_X - SHORT_SERVICE, -DOUBLES_Y, DOUBLES_Y),
    (NET_X + SHORT_SERVICE, NET_X + SHORT_SERVICE + LINE_W, -DOUBLES_Y, DOUBLES_Y),
    (xb0, NET_X - SHORT_SERVICE, -hw, hw), (NET_X + SHORT_SERVICE, xb1, -hw, hw),
]
for i, (a, b, c, d) in enumerate(lines):
    floor_rect(f"/World/Court/Line_{i}", a, b, c, d, z_line, line_white)

# net: posts, string mesh (alpha cut-out), white top tape
for s, name in ((1, "PostL"), (-1, "PostR")):
    cylinder(f"/World/Net/{name}", (NET_X, s * (DOUBLES_Y + 0.02), NET_HEIGHT / 2), 0.02, NET_HEIGHT, post_mat)
    cylinder(f"/World/Net/{name}Base", (NET_X, s * (DOUBLES_Y + 0.02), 0.02), 0.12, 0.04, post_mat)
net_mesh("/World/Net/Mesh", NET_X, -DOUBLES_Y, DOUBLES_Y, NET_HEIGHT - 0.76, NET_HEIGHT - 0.075, net_mat)
box("/World/Net/Tape", (NET_X, 0, NET_HEIGHT - 0.0375), (0.006, 2 * DOUBLES_Y, 0.075), tape_mat)

# perimeter LED boards (emissive), ~2 m outside the court
for i, (cx, cy, length, along_x) in enumerate(
    [(NET_X - HALF_LEN - 2.5, 0, 9.0, False), (NET_X + HALF_LEN + 3.0, 0, 9.0, False),
     (NET_X - 4.0, DOUBLES_Y + 2.6, 7.5, True), (NET_X + 4.0, DOUBLES_Y + 2.6, 7.5, True),
     (NET_X - 4.0, -DOUBLES_Y - 2.6, 7.5, True), (NET_X + 4.0, -DOUBLES_Y - 2.6, 7.5, True)]
):
    h = 0.9
    if along_x:
        sgn = -1 if cy > 0 else 1
        quad(f"/World/Boards/Face_{i}", (cx - length / 2, cy, 0.0), (length, 0, 0), (0, 0, h), led, uv=(4.0, 1.0),
             double_sided=True)
        box(f"/World/Boards/Back_{i}", (cx, cy - sgn * 0.06, h / 2), (length, 0.1, h), board_back)
    else:
        sgn = 1 if cx < 0 else -1
        quad(f"/World/Boards/Face_{i}", (cx, cy - length / 2, 0.0), (0, length, 0), (0, 0, h), led, uv=(4.0, 1.0),
             double_sided=True)
        box(f"/World/Boards/Back_{i}", (cx - sgn * 0.06, cy, h / 2), (0.1, length, h), board_back)

data = np.load(args.rollout)
BASE_LINK = list(map(str, data["body_names"])).index("iiwa7_link_0")
base_xy = data["body_pos"][:, BASE_LINK, :2]
MOBILE = bool(np.abs(base_xy).max() > 0.05)
WHEEL_R, WHEEL_W, WHEEL_X, WHEEL_Y = 0.1, 0.08, 0.24, 0.33
if not MOBILE:
    # robot pedestal (the env's 0.3 m column, dressed as a machined base)
    box("/World/Pedestal/Column", (0, 0, 0.3), (0.3, 0.3, 0.6), pedestal_mat)
    box("/World/Pedestal/Plate", (0, 0, 0.605), (0.34, 0.34, 0.012), plate_mat)
    box("/World/Pedestal/Foot", (0, 0, 0.015), (0.5, 0.5, 0.03), pedestal_mat)
else:
    # omnidirectional (mecanum) base carrying the arm mount at 0.6 m, animated from the arm's base link
    chassis_mat = material("Chassis", color=(0.85, 0.42, 0.06), roughness=0.35, clearcoat=0.5)
    cover_mat = material("Cover", color=(0.08, 0.08, 0.09), roughness=0.45)
    strip_mat = material("Strip", color=(0.0, 0.0, 0.0), emissive=(0.0, 1.6, 2.0), roughness=0.4)
    hub_mat = material("Hub", color=(0.7, 0.72, 0.75), roughness=0.25, metallic=1.0)
    tread_mat = material("Tread", texture=roller_texture(), roughness=0.6)
    mobile = UsdGeom.Xform.Define(stage, "/World/MobileBase")
    box("/World/MobileBase/Chassis", (0, 0, 0.2), (0.72, 0.56, 0.2), chassis_mat)
    box("/World/MobileBase/Top", (0, 0, 0.305), (0.70, 0.54, 0.012), cover_mat)
    box("/World/MobileBase/Bumper", (0, 0, 0.115), (0.74, 0.58, 0.03), cover_mat)
    box("/World/MobileBase/LedFront", (-0.3605, 0, 0.24), (0.002, 0.4, 0.015), strip_mat)
    box("/World/MobileBase/LedBack", (0.3605, 0, 0.24), (0.002, 0.4, 0.015), strip_mat)
    box("/World/MobileBase/Column", (0, 0, 0.455), (0.26, 0.26, 0.29), pedestal_mat)
    box("/World/MobileBase/Plate", (0, 0, 0.605), (0.32, 0.32, 0.012), plate_mat)
    mobile_ops = (mobile.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble),
                  mobile.AddOrientOp(UsdGeom.XformOp.PrecisionFloat))
    wheel_ops = {}
    for name, sx, sy in (("FL", -1, 1), ("FR", -1, -1), ("RL", 1, 1), ("RR", 1, -1)):
        wx = UsdGeom.Xform.Define(stage, f"/World/MobileBase/Wheel{name}")
        UsdGeom.Xformable(wx).AddTranslateOp().Set(Gf.Vec3d(sx * WHEEL_X, sy * WHEEL_Y, WHEEL_R))
        wheel_ops[name] = (UsdGeom.Xformable(wx).AddRotateYOp(), sx, sy)
        wheel_tread(f"/World/MobileBase/Wheel{name}/Tread", WHEEL_R, WHEEL_W, tread_mat)
        cylinder(f"/World/MobileBase/Wheel{name}/Hub", (0, 0, 0), WHEEL_R * 0.55, WHEEL_W + 0.01, hub_mat, axis="Y")

# lighting: rows of overhead LED panels plus a faint fill
for i, x in enumerate(np.linspace(NET_X - 7.5, NET_X + 7.5, 6)):
    for j, y in enumerate((-4.5, 0.0, 4.5)):
        lt = UsdLux.RectLight.Define(stage, f"/World/Lights/Panel_{i}_{j}")
        lt.CreateWidthAttr(1.6)
        lt.CreateHeightAttr(0.5)
        lt.CreateIntensityAttr(30000.0)
        lt.CreateColorTemperatureAttr(5200.0)
        lt.CreateEnableColorTemperatureAttr(True)
        UsdGeom.Xformable(lt).AddTranslateOp().Set(Gf.Vec3d(x, y, HALL_Z - 0.4))
        box(f"/World/Lights/Fixture_{i}_{j}", (x, y, HALL_Z - 0.36), (1.7, 0.6, 0.05), fixture_mat)
fill = UsdLux.DomeLight.Define(stage, "/World/Lights/Fill")
fill.CreateIntensityAttr(250.0)
fill.CreateColorAttr(Gf.Vec3f(0.75, 0.82, 1.0))

# ----------------------------------------------------------------------------------------------
# robot (visual only: physics schemas removed), racket and shuttle
# ----------------------------------------------------------------------------------------------

robot = UsdGeom.Xform.Define(stage, "/World/Robot")
robot.GetPrim().GetReferences().AddReference(str(data["robot_usd"]))
robot_xf = UsdGeom.Xformable(robot)
robot_xf.ClearXformOpOrder()
robot_xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "replay").Set(Gf.Vec3d(*map(float, data["robot_pos"])))
for prim in Usd.PrimRange(robot.GetPrim()):
    if prim.IsA(UsdPhysics.Joint):
        prim.SetActive(False)
        continue
    for api in (UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
        if prim.HasAPI(api):
            prim.RemoveAPI(api)
carb.settings.get_settings().set_bool("/app/player/playSimulations", False)

link_xf = {}
for name, rel in zip(data["body_names"], data["body_paths"]):
    prim = stage.GetPrimAtPath(f"/World/Robot{rel}")
    assert prim.IsValid(), rel
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.SetResetXformStack(True)
    link_xf[str(name)] = (xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "replay"),
                          xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat, "replay"))

grip_mat = material("Grip", color=(0.55, 0.04, 0.05), roughness=0.75)
frame_mat = material("Frame", color=(0.03, 0.03, 0.035), roughness=0.25, metallic=0.3, clearcoat=0.8)
accent_mat = material("FrameAccent", color=(0.85, 0.75, 0.1), roughness=0.3, metallic=0.9)
strings_mat = material("Strings", texture=grid_alpha_texture("strings.png", 256, 10, 4, (0.93, 0.93, 0.88)),
                       uv_scale=(2.2, 2.2), opacity_texture=True, roughness=0.4)
shaft_mat = material("Shaft", color=(0.6, 0.62, 0.66), roughness=0.2, metallic=1.0)
racket = UsdGeom.Xform.Define(stage, "/World/Racket")
cylinder("/World/Racket/Grip", (0, 0, 0.0), 0.0155, 0.2, grip_mat)
cylinder("/World/Racket/Butt", (0, 0, -0.103), 0.017, 0.008, frame_mat)
cylinder("/World/Racket/Shaft", (0, 0, 0.225), 0.0045, 0.25, shaft_mat)
ellipse_tube("/World/Racket/Frame", 0.475, 0.097, 0.122, 0.0065, frame_mat)
ellipse_tube("/World/Racket/Bumper", 0.475, 0.103, 0.128, 0.0025, accent_mat, ring=8)
ellipse_disc("/World/Racket/Strings", 0.475, 0.092, 0.117, strings_mat)
racket_ops = (racket.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), racket.AddOrientOp(UsdGeom.XformOp.PrecisionFloat))

cork_mat = material("Cork", color=(0.95, 0.94, 0.9), roughness=0.6)
band_mat = material("Band", color=(0.1, 0.25, 0.75), roughness=0.4)
feather_mat = material("Feathers", texture=feather_texture(), opacity_texture=True, roughness=0.55)
shuttle = UsdGeom.Xform.Define(stage, "/World/Shuttle")
sphere("/World/Shuttle/Cork", (0, 0, 0.012), 0.0125, cork_mat)
cylinder("/World/Shuttle/Band", (0, 0, 0.004), 0.0128, 0.006, band_mat)
frustum("/World/Shuttle/Skirt", 0.002, 0.0135, -0.062, 0.033, feather_mat)
shuttle_ops = (shuttle.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), shuttle.AddOrientOp(UsdGeom.XformOp.PrecisionFloat))

ring_mat = material("TargetRing", color=(0.0, 0.0, 0.0), emissive=(0.0, 2.2, 2.6), roughness=0.4)
hit_mat = material("TargetHit", color=(0.0, 0.0, 0.0), emissive=(0.4, 3.2, 0.6), roughness=0.4)
target = UsdGeom.Xform.Define(stage, "/World/Target")
annulus("/World/Target/Ring", TARGET_RADIUS - 0.035, TARGET_RADIUS, 0.006, ring_mat)
annulus("/World/Target/Dot", 0.0, 0.06, 0.006, ring_mat)
target_hit = UsdGeom.Xform.Define(stage, "/World/Target/Hit")
annulus("/World/Target/Hit/Ring", TARGET_RADIUS - 0.035, TARGET_RADIUS, 0.007, hit_mat)
annulus("/World/Target/Hit/Fill", 0.0, TARGET_RADIUS - 0.035, 0.0065,
        material("TargetFill", color=(0.0, 0.0, 0.0), emissive=(0.05, 0.35, 0.08), roughness=0.4))
target_op = target.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)

cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(36.0 * args.height / args.width)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
cam_op = cam.AddTransformOp()

# ----------------------------------------------------------------------------------------------
# edit: rallies, shots and an action replay, laid out on the stage timeline
# ----------------------------------------------------------------------------------------------

DT = float(data["dt"])
ep = data["episode"]
succ = data["success"].astype(bool)
if args.episodes:
    rallies = args.episodes
elif MOBILE:
    travel = {int(e): float(np.linalg.norm(base_xy[ep == e] - base_xy[ep == e][0], axis=-1).max())
              for e in np.flatnonzero(succ)}
    rallies = sorted(sorted(travel, key=travel.get, reverse=True)[: args.max_rallies])
else:
    rallies = [int(e) for e in np.flatnonzero(succ)][: args.max_rallies]


def q_xyzw_to_gf(q):
    return Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def quat_between_z(v):
    """Rotation taking +z onto v (wxyz)."""
    v = v / np.linalg.norm(v)
    c = v[2]
    axis = np.cross([0.0, 0.0, 1.0], v)
    s = np.linalg.norm(axis)
    if s < 1e-6:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
    axis /= s
    ang = math.atan2(s, c)
    return np.concatenate(([math.cos(ang / 2)], math.sin(ang / 2) * axis))


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_rotate_xyzw(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))


def look_at(eye, target_pt):
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target_pt), Gf.Vec3d(0, 0, 1))
    return m.GetInverse()


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# shots: (eye at start, eye at end, look-at, focal length mm, f-stop); look-at may track the shuttle
SHOTS = {
    "broadcast": dict(eye0=(6.6, 1.2, 3.7), eye1=(6.2, 0.6, 3.5), look=(-3.6, 0.0, 0.5), focal=30.0, fstop=8.0),
    "side": dict(eye0=(2.2, 5.6, 2.0), eye1=(1.7, 5.0, 1.9), look=(-2.6, -0.5, 1.0), focal=26.0, fstop=5.6),
    "far": dict(eye0=(-10.6, -2.4, 2.4), eye1=(-10.2, -1.8, 2.3), look=(-1.2, 0.1, 1.2), focal=45.0, fstop=8.0),
    "replay": dict(offset0=(-1.0, 1.35, 0.30), offset1=(-0.8, 1.15, 0.22), look=None, focal=42.0, fstop=4.0),
}
# USD lens units are tenths of a scene unit (10 cm here) while focal lengths above are in mm,
# so the physical aperture comes out 100x too large without this factor
FSTOP_UNITS = 100.0
rally_shots = ["broadcast", "side", "far", "broadcast", "far", "side", "broadcast", "far"]
if MOBILE:
    SHOTS["broadcast"] = dict(eye0=(8.4, 0.6, 4.6), eye1=(8.0, -0.4, 4.4), look=(-2.6, 0.0, 0.4), focal=24.0, fstop=8.0)
    # follows the base (smoothed) from behind and to the side
    SHOTS["track"] = dict(offset0=(3.0, 3.1, 1.9), offset1=(2.6, 2.7, 1.8), look_offset=(-1.6, -0.3, 0.8),
                          focal=28.0, fstop=5.6)
    rally_shots = ["broadcast", "track", "far", "track", "broadcast", "far", "track", "broadcast"]
# base position low-passed for the tracking camera (+-0.3 s window)
_k = int(0.3 / float(data["dt"]))
base_xy_smooth = np.stack([np.convolve(np.pad(base_xy[:, i], _k, mode="edge"), np.ones(2 * _k + 1) / (2 * _k + 1),
                                       mode="valid") for i in range(2)], axis=-1)

HOLD = 0.6  # s of held poses after the shuttle lands
GAP = 1.0  # s of dead time between segments so motion blur never straddles a cut
segments = []  # dicts: t0, t1 (stage seconds rendered), label info
samples = []  # (stage_time, frame_index, shot, seg_id, u)
t = GAP


def hit_frame(frames):
    v = data["shuttle_vel"][frames, 0]
    flips = np.flatnonzero((v[:-1] > 0) & (v[1:] < -0.5))
    return int(frames[flips[0] + 1]) if len(flips) else int(frames[len(frames) // 2])


replay_ep = (max(rallies, key=lambda e: travel[e]) if MOBILE and not args.episodes else rallies[0]) if rallies else None
for k, e in enumerate(rallies):
    frames = np.flatnonzero(ep == e)
    dur = len(frames) * DT
    shot = rally_shots[k % len(rally_shots)]
    seg = dict(id=len(segments), kind="rally", episode=int(e), shot=shot, t0=t, t1=t + dur + HOLD, frames=frames)
    for i, f in enumerate(frames):
        samples.append((t + i * DT, int(f), seg))
    segments.append(seg)
    t = seg["t1"] + GAP
    if e == replay_ep:
        # action replay at quarter speed around the hit, from a close camera
        h = hit_frame(frames)
        lo, hi = max(frames[0], h - int(0.30 / DT)), min(frames[-1], h + int(0.30 / DT))
        rf = np.arange(lo, hi + 1)
        rq = data["racket_quat"][h]
        head = data["racket_pos"][h] + quat_rotate_xyzw(rq, np.array([0.0, 0.0, 0.475]))
        rseg = dict(id=len(segments), kind="replay", episode=int(e), shot="replay", t0=t,
                    t1=t + len(rf) * DT * 4, frames=rf, hit=h, hit_head=head)
        for i, f in enumerate(rf):
            samples.append((t + i * DT * 4, int(f), rseg))
        segments.append(rseg)
        t = rseg["t1"] + GAP

tcps = 240.0
stage.SetTimeCodesPerSecond(tcps)
stage.SetStartTimeCode(0)
stage.SetEndTimeCode(t * tcps)

spin_phase = 0.0
last_axis = np.array([1.0, 0.0, 0.0])
wheel_angle = {k: 0.0 for k in ("FL", "FR", "RL", "RR")}


def author(time_s, f, seg, u, advance=True):
    """Write all animated attributes for recorded frame f at stage time time_s (advance: wheels roll)."""
    global spin_phase, last_axis
    tc = Usd.TimeCode(time_s * tcps)
    for b, name in enumerate(data["body_names"]):
        tr, ori = link_xf[str(name)]
        tr.Set(Gf.Vec3d(*map(float, data["body_pos"][f, b])), tc)
        ori.Set(q_xyzw_to_gf(data["body_quat"][f, b]), tc)
    racket_ops[0].Set(Gf.Vec3d(*map(float, data["racket_pos"][f])), tc)
    racket_ops[1].Set(q_xyzw_to_gf(data["racket_quat"][f]), tc)
    v = data["shuttle_vel"][f]
    speed = float(np.linalg.norm(v))
    if speed > 0.3:
        last_axis = v / speed
        spin_phase += 2 * math.pi * 6.0 * DT
    q = qmul(quat_between_z(last_axis), np.array([math.cos(spin_phase / 2), 0, 0, math.sin(spin_phase / 2)]))
    shuttle_ops[0].Set(Gf.Vec3d(*map(float, data["shuttle_pos"][f])), tc)
    shuttle_ops[1].Set(Gf.Quatf(*map(float, q)), tc)
    lt = data["land_target"][seg["episode"]]
    target_op.Set(Gf.Vec3d(float(lt[0]), float(lt[1]), 0.0), tc)
    if MOBILE:
        bq = data["body_quat"][f, BASE_LINK]  # xyzw, yaw only
        yaw = 2 * math.atan2(float(bq[2]), float(bq[3]))
        mobile_ops[0].Set(Gf.Vec3d(float(base_xy[f, 0]), float(base_xy[f, 1]), 0.0), tc)
        mobile_ops[1].Set(Gf.Quatf(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), tc)
        # mecanum inverse kinematics from the base velocity (finite difference), for the wheel spin
        f0 = max(f - 1, 0)
        v = (base_xy[f] - base_xy[f0]) / DT if ep[f0] == ep[f] else np.zeros(2)
        bq0 = data["body_quat"][f0, BASE_LINK]
        wz = (yaw - 2 * math.atan2(float(bq0[2]), float(bq0[3]))) / DT if ep[f0] == ep[f] else 0.0
        wz = (wz + math.pi / DT) % (2 * math.pi / DT) - math.pi / DT
        c, s_ = math.cos(yaw), math.sin(yaw)
        vx, vy = c * v[0] + s_ * v[1], -s_ * v[0] + c * v[1]
        for name, (op, sx, sy) in wheel_ops.items():
            rate = (vx - sx * sy * vy - (WHEEL_X + WHEEL_Y) * sy * wz) / WHEEL_R
            if advance:
                wheel_angle[name] += math.degrees(rate * DT)
            op.Set(wheel_angle[name], tc)
    # camera
    s = SHOTS[seg["shot"]]
    w = smooth(u)
    if seg["shot"] == "replay":
        look = seg["hit_head"]
        eye = look + np.array(s["offset0"]) * (1 - w) + np.array(s["offset1"]) * w
    elif seg["shot"] == "track":
        anchor = np.array([base_xy_smooth[f, 0], base_xy_smooth[f, 1], 0.0])
        look = anchor + np.array(s["look_offset"])
        eye = anchor + np.array(s["offset0"]) * (1 - w) + np.array(s["offset1"]) * w
    else:
        look = np.array(s["look"])
        eye = np.array(s["eye0"]) * (1 - w) + np.array(s["eye1"]) * w
    cam_op.Set(look_at(eye, look), tc)
    cam.GetFocalLengthAttr().Set(s["focal"], tc)
    cam.GetFStopAttr().Set(s["fstop"] * FSTOP_UNITS, tc)
    robot_pt = np.array([base_xy[f, 0], base_xy[f, 1], 1.3])
    focus = np.linalg.norm(eye - (look if seg["shot"] in ("replay", "far") else robot_pt))
    if seg["shot"] == "far":
        focus = np.linalg.norm(eye - np.array([lt[0], lt[1], 0.5]))
    cam.GetFocusDistanceAttr().Set(float(focus), tc)


for seg in segments:
    frames = seg["frames"]
    seg_samples = [s_ for s_ in samples if s_[2] is seg]
    # pre-roll: hold the first pose from half the gap before the segment
    author(seg["t0"] - GAP / 2, int(frames[0]), seg, 0.0, advance=False)
    for ts, f, _ in seg_samples:
        author(ts, f, seg, (ts - seg["t0"]) / (seg["t1"] - seg["t0"]))
    # hold the last pose through the landing pause and half the gap
    last = int(frames[-1])
    author(seg["t1"], last, seg, 1.0, advance=False)
    author(seg["t1"] + GAP / 2, last, seg, 1.0, advance=False)

# target ring turns green when the shot lands inside it (rally segments only)
vis = UsdGeom.Imageable(target_hit).GetVisibilityAttr()
vis.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(0))
for seg in segments:
    if seg["kind"] != "rally" or not succ[seg["episode"]]:
        continue
    t_land = seg["t0"] + len(seg["frames"]) * DT
    vis.Set(UsdGeom.Tokens.invisible, Usd.TimeCode((t_land - 0.01) * tcps))
    vis.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(t_land * tcps))
    vis.Set(UsdGeom.Tokens.invisible, Usd.TimeCode((seg["t1"] + 0.01) * tcps))
stage.SetInterpolationType(Usd.InterpolationTypeLinear)

stage.GetRootLayer().Export(os.path.join(args.out_dir, "badminton_twin.usda"))

# ----------------------------------------------------------------------------------------------
# render
# ----------------------------------------------------------------------------------------------

settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", args.renderer)
settings.set_int("/rtx/pathtracing/spp", args.spp)
settings.set_int("/rtx/pathtracing/totalSpp", args.spp * args.subframes)
settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)
settings.set_int("/rtx/pathtracing/maxBounces", 6)
settings.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
settings.set_bool("/rtx/post/histogram/enabled", True)  # auto exposure
settings.set("/rtx/post/tonemap/op", 6)  # ACES
rep.orchestrator.set_capture_on_play(False)

rp = rep.create.render_product(str(cam.GetPath()), (args.width, args.height))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)
timeline = omni.timeline.get_timeline_interface()
frames_dir = os.path.join(args.out_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)

# let assets stream in and the path tracer warm up
for _ in range(30):
    app.update()

meta = []
if args.test:
    shots = []
    for seg in segments:
        span = seg["t1"] - seg["t0"]
        shots += [(seg, seg["t0"] + 0.15 * span), (seg, seg["t0"] + 0.55 * span)]
    for i, (seg, ts) in enumerate(shots):
        timeline.set_current_time(ts)
        t_start = time.time()
        rep.orchestrator.step(delta_time=0.0, rt_subframes=args.subframes)
        img = rgb.get_data()
        Image.fromarray(img[..., :3]).save(os.path.join(frames_dir, f"test_{i:02d}_{seg['shot']}.png"))
        print(f"RENDER test {i} shot={seg['shot']} t={ts:.2f}s took {time.time() - t_start:.1f}s", flush=True)
else:
    settings.set_bool("/omni/replicator/captureMotionBlur", True)
    settings.set_int("/omni/replicator/pathTracedMotionBlurSubSamples", args.subframes)
    k = 0
    t_begin = time.time()
    total = int(sum(round((s["t1"] - s["t0"]) * args.fps) for s in segments))
    for seg in segments:
        n = int(round((seg["t1"] - seg["t0"]) * args.fps))
        timeline.set_current_time(seg["t0"] - 1.0 / args.fps)
        for i in range(n):
            # path tracing spreads its sub-frames over the shutter; real-time PT accumulates them at one instant
            rep.orchestrator.step(delta_time=1.0 / args.fps,
                                  rt_subframes=args.subframes if args.renderer == "RealTimePathTracing" else -1)
            img = rgb.get_data()
            Image.fromarray(img[..., :3]).save(os.path.join(frames_dir, f"{k:05d}.png"))
            meta.append(dict(frame=k, seg=seg["id"], kind=seg["kind"], episode=seg["episode"], shot=seg["shot"],
                             t=timeline.get_current_time() - seg["t0"]))
            k += 1
            if k % 20 == 0:
                el = time.time() - t_begin
                print(f"RENDER {k}/{total} frames, {el / k:.2f} s/frame, eta {(total - k) * el / k / 60:.1f} min",
                      flush=True)

summary = dict(
    rallies=[dict(episode=s["episode"], kind=s["kind"], shot=s["shot"], duration=s["t1"] - s["t0"],
                  land_err=float(np.linalg.norm(data["shuttle_pos"][s["frames"][-1], :2] - data["land_target"][s["episode"]])),
                  success=bool(succ[s["episode"]]), land_time=len(s["frames"]) * DT) for s in segments],
    frames=meta, fps=args.fps,
)
with open(os.path.join(args.out_dir, "render_meta.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print("RENDER done", flush=True)
rep.orchestrator.wait_until_complete()
app.close()
