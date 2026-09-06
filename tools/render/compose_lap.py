#!/usr/bin/env python3
# Caption fade for the last 150 frames (3 s at 50 fps): dark bottom gradient
# plus "195,471 / 195,471 ticks identical to the game", same layout as the
# hero still (compose_hero.py) scaled from 3840x2160 to 1920x1080.
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = "/home/adityas/Projects/TMNF-C/viewer/fonts"
SRC = "/tmp/tweet/frames"
DST = "/tmp/tweet/frames_cap"
TOTAL = int(sys.argv[1])
FADE = 150
os.makedirs(DST, exist_ok=True)

W, H = 1920, 1080
band_top = 850
alpha = np.zeros((H, W), dtype=np.float32)
ramp = np.linspace(0.0, 1.0, H - band_top)
alpha[band_top:, :] = (ramp ** 1.6 * 0.52)[:, None]
dark = Image.new("RGB", (W, H), (0x0b, 0x0d, 0x10))
font = ImageFont.truetype(f"{FONTS}/IBMPlexSans-SemiBold.woff2", 28)
text = "195,471 / 195,471 ticks identical to the game"
x, baseline = 64, H - 56

for i in range(TOTAL - FADE, TOTAL):
    t = (i - (TOTAL - FADE) + 1) / FADE
    fade = min(t / 0.6, 1.0)  # fully in after 1.8 s, hold 1.2 s
    fade = fade * fade * (3 - 2 * fade)
    image = Image.open(f"{SRC}/{i:05d}.png").convert("RGB")
    mask = Image.fromarray((alpha * fade * 255).astype(np.uint8), "L")
    image = Image.composite(dark, image, mask)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((x, baseline), text, font=font, fill=(0xe8, 0xec, 0xf1, int(255 * fade)), anchor="ls")
    image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    image.save(f"{DST}/{i:05d}.png", compress_level=1)
for i in range(0, TOTAL - FADE):
    dst = f"{DST}/{i:05d}.png"
    if not os.path.lexists(dst):
        os.symlink(f"{SRC}/{i:05d}.png", dst)
print("composited", FADE, "frames into", DST)
