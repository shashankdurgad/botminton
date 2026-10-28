"""Add broadcast captions to the rendered showcase frames and encode an H.264 mp4.

Usage:
    ../IsaacLab/.venv/bin/python scripts/compose_showcase.py --render_dir showcase --out showcase.mp4 \
        --stats "hit 91% · over the net 82% · in target 50% (2000 eval rallies)"
"""

import argparse
import json
import os

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("--render_dir", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--title", default="Robot badminton, learned with reinforcement learning")
parser.add_argument("--subtitle", default="KUKA iiwa 7 + Allegro hand  ·  PPO in Isaac Lab  ·  rendered with RTX path tracing")
parser.add_argument("--stats", default="")
args = parser.parse_args()

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

meta = json.load(open(os.path.join(args.render_dir, "render_meta.json")))
fps = meta["fps"]
segs = meta["rallies"]
frames = meta["frames"]
W, H = Image.open(os.path.join(args.render_dir, "frames", "00000.png")).size
s = H / 1080
f_big, f_mid, f_small = (ImageFont.truetype(FONT_B, int(54 * s)), ImageFont.truetype(FONT_B, int(30 * s)),
                         ImageFont.truetype(FONT, int(24 * s)))

GREEN, CYAN, AMBER, WHITE = (90, 230, 120), (60, 220, 235), (255, 190, 60), (245, 245, 245)
rally_no = {}
for i, seg in enumerate(segs):
    if seg["kind"] == "rally":
        rally_no[i] = len(rally_no) + 1


def fade(t, t0, t1, ramp=0.35):
    return float(np.clip(min(t - t0, t1 - t) / ramp, 0.0, 1.0))


def panel(draw, xy, text, font, color, alpha):
    if alpha <= 0:
        return
    x, y = xy
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    pad = int(14 * s)
    draw.rounded_rectangle((l - pad, t - pad, r + pad, b + pad), radius=int(8 * s), fill=(8, 12, 20, int(170 * alpha)))
    draw.text((x, y), text, font=font, fill=color + (int(255 * alpha),))


writer = imageio_ffmpeg.write_frames(args.out, (W, H), fps=fps, codec="libx264", quality=None,
                                     output_params=["-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p"])
writer.send(None)
n = len(frames)
total_t = n / fps
for k, fr in enumerate(frames):
    img = Image.open(os.path.join(args.render_dir, "frames", f"{k:05d}.png")).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    seg = segs[fr["seg"]]
    t = fr["t"]  # time within the segment
    seg_len = seg["duration"]
    g = k / fps  # global movie time

    # corner badge
    panel(d, (int(40 * s), int(36 * s)), "SIMULATION  ·  learned policy, no keyframes", f_small, WHITE, 0.9)

    if seg["kind"] == "rally":
        panel(d, (int(40 * s), H - int(90 * s)), f"RALLY {rally_no[fr['seg']]}", f_mid, WHITE, fade(t, 0, seg_len))
        if t >= seg["land_time"]:
            a = fade(t, seg["land_time"], seg_len + 10, ramp=0.15)
            if seg["success"]:
                msg, col = f"IN TARGET  ·  {seg['land_err']:.2f} m from the aim point", GREEN
            else:
                msg, col = f"LANDED {seg['land_err']:.2f} m from the aim point", AMBER
            panel(d, (int(250 * s), H - int(90 * s)), msg, f_mid, col, a)
    else:
        panel(d, (int(40 * s), H - int(90 * s)), "ACTION REPLAY  ·  ¼ speed", f_mid, CYAN, fade(t, 0, seg_len))

    # title over the opening, stats over the close
    if g < 3.2:
        a = fade(g, 0.0, 3.2, ramp=0.5)
        panel(d, (int(40 * s), int(110 * s)), args.title, f_big, WHITE, a)
        panel(d, (int(40 * s), int(200 * s)), args.subtitle, f_small, WHITE, a)
    if args.stats and g > total_t - 3.0:
        panel(d, (int(40 * s), int(110 * s)), "Policy evaluation", f_mid, WHITE, fade(g, total_t - 3.0, total_t + 1, 0.5))
        panel(d, (int(40 * s), int(170 * s)), args.stats, f_small, WHITE, fade(g, total_t - 3.0, total_t + 1, 0.5))

    out = Image.alpha_composite(img, over).convert("RGB")
    writer.send(np.ascontiguousarray(np.asarray(out)))
    if k % 200 == 0:
        print(f"COMPOSE {k}/{n}", flush=True)
writer.close()
print(f"COMPOSE done -> {args.out} ({n} frames, {n / fps:.1f} s)")
