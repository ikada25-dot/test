"""手描き風アニメーション「てがきの いちにち」を生成するスクリプト。

紙のテクスチャ、ゆらぐ鉛筆線（ラインボイル）、クレヨン風の塗り、
12fps の「2コマ打ち」で、ある家の一日（朝→雨→虹→夜）を描く。

    pip install pillow numpy imageio-ffmpeg
    python hand_drawn_video.py          # -> hand_drawn.mp4
"""

import math
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
SS = 2  # スーパーサンプリング倍率
FPS = 12  # 描画は 12fps（2コマ打ち）、出力は 24fps
DURATION = 25.0
FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
OUT = "hand_drawn.mp4"

PENCIL = (45, 42, 52)
PAPER_RGB = (246, 240, 226)


# ---------------------------------------------------------------- 素材

def make_paper():
    rng = np.random.default_rng(1)
    w, h = W * SS, H * SS
    base = np.ones((h, w, 3), np.float32) * np.array(PAPER_RGB, np.float32)
    fine = rng.normal(0, 5, (h, w, 1)).astype(np.float32)
    coarse = Image.fromarray(rng.integers(0, 255, (h // 16, w // 16), dtype=np.uint8))
    coarse = np.asarray(coarse.resize((w, h), Image.BICUBIC), np.float32)[..., None]
    base += fine + (coarse - 128) * 0.06
    # 紙の繊維
    fib = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(fib)
    for _ in range(900):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        a, l = rng.uniform(0, math.pi), rng.uniform(10, 50)
        d.line([(x, y), (x + math.cos(a) * l, y + math.sin(a) * l)], fill=int(rng.uniform(10, 30)), width=1)
    base -= np.asarray(fib, np.float32)[..., None] * 0.5
    # 周辺減光
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    base *= (1 - 0.12 * np.clip(r - 0.5, 0, 1))[..., None]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def make_grain():
    """クレヨン塗りのかすれ用テクスチャ（L, 0-255）。"""
    rng = np.random.default_rng(2)
    w, h = W * SS, H * SS
    n = rng.random((h, w)).astype(np.float32)
    streak = Image.fromarray((n * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(1))
    streak = streak.resize((w // 2, h // 6)).resize((w, h), Image.BILINEAR)  # 横方向のストローク感
    s = np.asarray(streak, np.float32) / 255
    g = 0.55 + 0.45 * (0.6 * s + 0.4 * n)
    g = np.where(n > 0.93, g * 0.4, g)  # 紙の凹凸で抜ける点
    return Image.fromarray(np.clip(g * 255, 0, 255).astype(np.uint8))


PAPER = make_paper()
GRAIN = make_grain()
FONTS = {}


def font(size):
    if size not in FONTS:
        FONTS[size] = ImageFont.truetype(FONT_PATH, size * SS)
    return FONTS[size]


# ---------------------------------------------------------------- ユーティリティ

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def prog(t, t0, dur):
    return ease((t - t0) / dur)


def lerp(a, b, k):
    return a + (b - a) * k


def lerp3(a, b, k):
    return tuple(lerp(x, y, k) for x, y in zip(a, b))


def resample(pts, step=3.0):
    out = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / step))
        for i in range(1, n + 1):
            out.append((x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n))
    return out


def jitter(pts, seed, amp=1.6):
    r = random.Random(seed)
    ph = [r.uniform(0, 6.3) for _ in range(4)]
    f1, f2 = r.uniform(0.02, 0.04), r.uniform(0.07, 0.12)
    ox, oy = r.uniform(-1, 1) * amp * 0.8, r.uniform(-1, 1) * amp * 0.8
    out, s, prev = [], 0.0, pts[0]
    for x, y in pts:
        s += math.hypot(x - prev[0], y - prev[1])
        prev = (x, y)
        dx = math.sin(s * f1 + ph[0]) + 0.4 * math.sin(s * f2 + ph[1])
        dy = math.sin(s * f1 + ph[2]) + 0.4 * math.sin(s * f2 + ph[3])
        out.append((x + ox + dx * amp, y + oy + dy * amp))
    return out


# ---------------------------------------------------------------- 形

def line(p0, p1, over=5):
    (x0, y0), (x1, y1) = p0, p1
    d = math.hypot(x1 - x0, y1 - y0) or 1
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    return [(x0 - ux * over, y0 - uy * over), (x1 + ux * over, y1 + uy * over)]


def poly(*pts, closed=False):
    pts = list(pts)
    return pts + [pts[0]] if closed else pts


def circle(cx, cy, r, seed=0, a0=None, span=None, wob=0.04, ry=None):
    rnd = random.Random(seed)
    a0 = rnd.uniform(0, 6.3) if a0 is None else a0
    span = 2 * math.pi + 0.35 if span is None else span
    ry = r if ry is None else ry
    n = max(24, int(abs(span) * max(r, ry) / 3))
    k1, k2 = rnd.uniform(0, 6.3), rnd.uniform(0, 6.3)
    out = []
    for i in range(n + 1):
        a = a0 + span * i / n
        m = 1 + wob * math.sin(a * 2 + k1) + wob * 0.5 * math.sin(a * 5 + k2) + wob * 0.6 * (i / n)
        out.append((cx + math.cos(a) * r * m, cy + math.sin(a) * ry * m))
    return out


def blob(cx, cy, rx, ry, bumps, seed):
    """もこもこした雲・木の輪郭。"""
    rnd = random.Random(seed)
    out = []
    for i in range(bumps):
        a0 = 2 * math.pi * i / bumps
        a1 = 2 * math.pi * (i + 1) / bumps
        bump = rnd.uniform(0.18, 0.3)
        for j in range(12):
            a = lerp(a0, a1, j / 12)
            k = 1 + bump * math.sin(math.pi * j / 12)
            out.append((cx + math.cos(a) * rx * k, cy + math.sin(a) * ry * k))
    out.append(out[0])
    return out


def hill(cx, base, half, height, n=60):
    return [(cx + half * (2 * i / n - 1), base - height * math.cos(math.pi / 2 * (2 * i / n - 1)) ** 1.6)
            for i in range(n + 1)]


# ---------------------------------------------------------------- 描画

class Frame:
    def __init__(self, t, boil):
        self.t = t
        self.boil = boil  # 線のゆらぎ用シード（描画ごとに変わる）
        self.img = PAPER.copy()
        self.draw = ImageDraw.Draw(self.img, "RGBA")

    def stroke(self, pts, p=1.0, seed=0, color=PENCIL, width=3.0, alpha=225, amp=1.6, passes=2):
        if p <= 0:
            return
        pts = resample(pts)
        n = max(2, int(len(pts) * clamp(p)))
        for k in range(passes):
            j = jitter(pts, seed * 131 + self.boil * 7 + k * 17, amp * (1 + 0.4 * k))[:n]
            j = [(x * SS, y * SS) for x, y in j]
            w = max(1, int(width * SS * (1 - 0.35 * k)))
            a = int(alpha * (1 - 0.45 * k))
            self.draw.line(j, fill=color + (a,), width=w, joint="curve")

    def _mask(self, pts, seed, blur=2.5, amp=2.5):
        j = jitter(resample(pts, 4), seed * 97 + self.boil * 5, amp)
        j = [(x * SS, y * SS) for x, y in j]
        xs, ys = [p[0] for p in j], [p[1] for p in j]
        pad = 12
        x0, y0 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
        x1, y1 = min(W * SS, int(max(xs)) + pad), min(H * SS, int(max(ys)) + pad)
        if x1 <= x0 or y1 <= y0:
            return None, None
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(m).polygon([(x - x0, y - y0) for x, y in j], fill=255)
        if blur:
            m = m.filter(ImageFilter.GaussianBlur(blur * SS))
        return m, (x0, y0, x1, y1)

    def fill(self, pts, color, alpha=0.6, seed=0, offset=(4, -3)):
        """クレヨン塗り。線から少しずらして塗ると手描き感が出る。"""
        if alpha <= 0:
            return
        pts = [(x + offset[0], y + offset[1]) for x, y in pts]
        m, box = self._mask(pts, seed)
        if m is None:
            return
        m = ImageChops.multiply(m, GRAIN.crop(box))
        m = m.point(lambda v: int(v * clamp(alpha)))
        self.img.paste(Image.new("RGB", m.size, color), box[:2], m)

    def erase(self, pts, seed=0):
        """紙の色で塗りつぶして背後を隠す。"""
        m, box = self._mask(pts, seed, blur=0.6, amp=1.2)
        if m is not None:
            self.img.paste(PAPER.crop(box), box[:2], m)

    def text(self, s, x, y, size, p=1.0, color=PENCIL, alpha=1.0, seed=0, center=False):
        if p <= 0 or alpha <= 0:
            return
        f = font(size)
        widths = [f.getlength(c) / SS for c in s]
        total = sum(widths) * 1.04
        cx = x - total / 2 if center else x
        shown = s[: max(1, int(math.ceil(len(s) * clamp(p))))]
        for i, c in enumerate(shown):
            r = random.Random(seed * 1000 + i * 31 + self.boil)
            if c.strip():
                g = Image.new("L", (size * SS * 2, size * SS * 2), 0)
                ImageDraw.Draw(g).text((size * SS // 2, size * SS // 2), c, font=f, fill=255)
                g = g.rotate(r.uniform(-5, 5), resample=Image.BICUBIC)
                g = ImageChops.multiply(g, GRAIN.crop((0, 0) + g.size).point(lambda v: min(255, v + 60)))
                g = g.point(lambda v: int(v * alpha))
                px = int((cx + r.uniform(-1.5, 1.5)) * SS - size * SS // 2)
                py = int((y + r.uniform(-2, 2) + 3 * math.sin(i * 1.7 + seed)) * SS - size * SS // 2 - size * SS // 2)
                self.img.paste(Image.new("RGB", g.size, color), (px, py), g)
            cx += widths[i] * 1.04

    def tint(self, rgb):
        a = np.asarray(self.img, np.float32) * (np.array(rgb, np.float32) / 255)
        self.img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        self.draw = ImageDraw.Draw(self.img, "RGBA")

    def glow(self, cx, cy, r, color, alpha):
        if alpha <= 0:
            return
        g = Image.new("L", (int(r * 4 * SS), int(r * 4 * SS)), 0)
        ImageDraw.Draw(g).ellipse((r * SS, r * SS, r * 3 * SS, r * 3 * SS), fill=int(255 * alpha))
        g = g.filter(ImageFilter.GaussianBlur(r * SS * 0.5))
        self.img.paste(Image.new("RGB", g.size, color), (int((cx - 2 * r) * SS), int((cy - 2 * r) * SS)), g)

    def result(self):
        return self.img.resize((W, H), Image.LANCZOS)


# ---------------------------------------------------------------- シーン

# 時刻 (秒)
T_TITLE_OUT = 3.4
T_LAND = 3.6
T_RAIN = 11.5
T_RAINBOW = 15.0
T_DUSK = 18.5
T_NIGHT = 20.5
T_END = 23.0

CLOUDS = [(260, 150, 95, 42, 7, 11), (720, 105, 120, 48, 8, 12), (1120, 250, 80, 34, 6, 13)]
STARS = [(120, 80), (330, 200), (470, 60), (610, 170), (780, 50), (860, 210), (990, 110), (1180, 70), (1230, 190), (560, 250)]


def sky_color(t):
    day = (255, 250, 238)
    rain = (212, 218, 228)
    dusk = (255, 208, 170)
    night = (92, 104, 158)
    if t < T_RAIN:
        return day
    if t < T_RAINBOW:
        return lerp3(day, rain, prog(t, T_RAIN, 1.2))
    if t < T_DUSK:
        return lerp3(rain, day, prog(t, T_RAINBOW, 1.5))
    if t < T_NIGHT:
        return lerp3(day, dusk, prog(t, T_DUSK, 1.5))
    return lerp3(dusk, night, prog(t, T_NIGHT, 1.8))


def caption(f, t, s, t0, t1, seed):
    a = prog(t, t0, 0.3) * (1 - prog(t, t1 - 0.4, 0.4))
    f.text(s, 70, 70, 54, p=prog(t, t0, 0.6), alpha=a, seed=seed)
    f.stroke(line((66, 104), (66 + 58 * len(s), 106)), p=prog(t, t0 + 0.4, 0.5) * (a > 0.05),
             seed=seed, color=(200, 80, 60), width=4, alpha=int(200 * a))


def render(t, boil):
    f = Frame(t, boil)

    # ---- タイトル
    if t < T_TITLE_OUT + 0.3:
        a = 1 - prog(t, T_TITLE_OUT - 0.5, 0.6)
        f.text("てがきの いちにち", 640, 330, 96, p=prog(t, 0.2, 1.4), alpha=a, seed=1, center=True)
        f.stroke(poly(*[(390 + i * 10, 398 + 6 * math.sin(i * 0.55)) for i in range(51)]),
                 p=prog(t, 1.5, 0.8), seed=2, color=(210, 90, 60), width=5, alpha=int(210 * a))
        f.text("〜 ある家の一日 〜", 640, 460, 38, p=prog(t, 2.0, 0.8), alpha=a, seed=3, center=True)
        if t < T_LAND:
            return f.result()

    # ---- 空: 太陽・月・星・雲・虹
    # 太陽（朝に昇り、夕方に沈む）
    sun_y = lerp(430, 170, prog(t, 6.6, 2.4))
    sun_y = lerp(sun_y, 640, prog(t, T_DUSK, 2.5))
    sun_x = lerp(1050, 1110, prog(t, T_DUSK, 2.5))
    sun_hide = prog(t, T_RAIN, 1.0) * (1 - prog(t, T_RAINBOW, 1.2))
    if t > 6.6:
        f.fill(circle(sun_x, sun_y, 58, seed=5, span=6.3), (245, 170, 60), alpha=0.85 * (1 - sun_hide), seed=5)
        f.stroke(circle(sun_x, sun_y, 58, seed=5), seed=5, color=(200, 110, 40), width=3.5, alpha=int(230 * (1 - sun_hide)))
        rot = t * 0.4
        for i in range(10):
            a = rot + i * math.pi / 5
            f.stroke(line((sun_x + math.cos(a) * 75, sun_y + math.sin(a) * 75),
                          (sun_x + math.cos(a) * 100, sun_y + math.sin(a) * 100), over=0),
                     p=prog(t, 8.2 + i * 0.06, 0.3), seed=50 + i, color=(215, 130, 40), width=3.5,
                     alpha=int(220 * (1 - sun_hide) * (1 - prog(t, T_DUSK, 1))))

    # 月と星（夜）
    night = prog(t, T_NIGHT, 1.8)

    # 虹
    rb = [(225, 70, 70), (240, 150, 60), (240, 215, 80), (100, 180, 90), (80, 140, 220), (130, 90, 190)]
    rb_fade = 1 - prog(t, T_DUSK + 0.5, 1.5)
    for i, c in enumerate(rb):
        r = 470 - i * 16
        f.stroke(circle(640, 600, r, seed=70, a0=math.pi + 0.05, span=math.pi - 0.1, wob=0.008),
                 p=prog(t, T_RAINBOW + 0.3 + i * 0.25, 1.2), seed=70 + i, color=c, width=9,
                 alpha=int(170 * rb_fade), amp=2.2)

    # 雲
    gray = prog(t, T_RAIN, 1.0) * (1 - prog(t, T_RAINBOW, 1.5))
    for k, (cx, cy, rx, ry, bumps, seed) in enumerate(CLOUDS):
        dx = (t - 9) * (12 + 5 * k)
        grow = 1 + 0.35 * gray
        pts = blob(cx + dx, cy, rx * grow, ry * grow, bumps, seed)
        cp = prog(t, 9.0 + k * 0.4, 1.0)
        ccol = lerp3((170, 200, 235), (120, 125, 140), gray)
        f.fill(pts, tuple(int(v) for v in ccol), alpha=cp * (0.35 + 0.35 * gray) * (1 - night * 0.6), seed=seed)
        f.stroke(pts, p=cp, seed=seed, width=3, alpha=int(225 * (1 - night * 0.7)))

    # 雨
    if T_RAIN + 0.8 < t < T_RAINBOW + 0.6:
        rain_a = prog(t, T_RAIN + 0.8, 0.4) * (1 - prog(t, T_RAINBOW, 0.6))
        rnd = random.Random(99)
        for i in range(70):
            x0 = rnd.uniform(0, W + 150)
            spd = rnd.uniform(500, 700)
            y = (rnd.uniform(0, 700) + t * spd) % 560 + 160
            x = x0 - (y - 160) * 0.25
            f.stroke(line((x, y), (x - 6, y + 24), over=0), seed=200 + i, color=(80, 110, 170),
                     width=2.5, alpha=int(200 * rain_a), passes=1, amp=0.6)

    # ---- 丘
    hills = [(1000, 560, 520, 120, 20, (140, 190, 110)), (300, 560, 480, 160, 21, (110, 170, 90))]
    for i, (cx, base, half, hgt, seed, col) in enumerate(hills):
        p = prog(t, T_LAND + 0.5 + i * 0.4, 1.0)
        if p <= 0:
            continue
        outline = hill(cx, base, half, hgt)
        body = outline + [(cx + half, base + 15), (cx - half, base + 15)]
        f.erase(body, seed=seed)
        f.fill(body, col, alpha=0.7 * prog(t, T_LAND + 1.2 + i * 0.4, 0.8), seed=seed, offset=(0, 4))
        f.stroke(outline, p=p, seed=seed, width=3.5)

    # 地面
    ground = [(x, 560 + 4 * math.sin(x * 0.013) + 2 * math.sin(x * 0.041)) for x in range(-20, 1320, 20)]
    if t > T_DUSK:
        f.erase(ground + [(1320, 740), (-20, 740)], seed=22)  # 沈む太陽を隠す
    f.fill(ground + [(1320, 740), (-20, 740)], (150, 185, 100), alpha=0.75 * prog(t, T_LAND + 0.6, 0.8), seed=22, offset=(0, 6))
    f.stroke(ground, p=prog(t, T_LAND, 0.8), seed=22, width=3.5)
    for i in range(22):  # 草
        gx = 30 + i * 58 + (i * 37 % 23)
        gp = prog(t, T_LAND + 1.4 + i * 0.03, 0.3)
        f.stroke(poly((gx - 6, 600 + i % 3 * 25), (gx, 588 + i % 3 * 25), (gx + 5, 600 + i % 3 * 25)),
                 p=gp, seed=300 + i, color=(70, 120, 60), width=2.5)

    # ---- 家
    hp = lambda t0, d=0.6: prog(t, 4.8 + t0, d)
    wall = poly((480, 400), (700, 400), (700, 560), (480, 560), closed=True)
    if hp(0) > 0:
        f.erase(wall, seed=30)
        f.fill(wall, (235, 200, 150), alpha=0.8 * hp(1.4), seed=30)
        roof = poly((458, 402), (590, 296), (722, 402), closed=True)
        f.erase(roof, seed=31)
        f.fill(roof, (200, 80, 65), alpha=0.8 * hp(1.6), seed=31)
    f.stroke(line((480, 400), (480, 560)), p=hp(0, 0.3), seed=32)
    f.stroke(line((480, 560), (700, 560)), p=hp(0.2, 0.3), seed=33)
    f.stroke(line((700, 560), (700, 400)), p=hp(0.4, 0.3), seed=34)
    f.stroke(poly((455, 402), (590, 296), (725, 402), (455, 402)), p=hp(0.6, 0.6), seed=35)
    f.stroke(poly((630, 328), (630, 305), (660, 305), (660, 352)), p=hp(1.0, 0.3), seed=36)  # 煙突
    f.stroke(poly((560, 560), (560, 480), (608, 480), (608, 560)), p=hp(1.1, 0.4), seed=37)  # ドア
    f.stroke(circle(598, 522, 3, seed=38), p=hp(1.4, 0.1), seed=38, width=2.5)
    win = poly((625, 430), (680, 430), (680, 478), (625, 478), closed=True)
    f.stroke(win, p=hp(1.3, 0.4), seed=39)
    f.stroke(line((652.5, 430), (652.5, 478), over=0), p=hp(1.6, 0.2), seed=40, width=2.5)
    f.stroke(line((625, 454), (680, 454), over=0), p=hp(1.7, 0.2), seed=41, width=2.5)
    f.stroke(circle(530, 450, 20, seed=42), p=hp(1.5, 0.3), seed=42)  # 丸窓
    # 煙
    if t > 7.5:
        for i in range(3):
            ph = (t * 0.5 + i / 3) % 1
            sx, sy = 645 + 20 * ph + 8 * math.sin(ph * 6), 290 - 110 * ph
            f.stroke(circle(sx, sy, 10 + 14 * ph, seed=43 + i, wob=0.1), seed=43 + i, color=(120, 120, 130),
                     width=2.5, alpha=int(180 * math.sin(math.pi * ph) * prog(t, 7.5, 0.5)))

    # ---- 木
    tp = lambda t0, d=0.5: prog(t, 6.2 + t0, d)
    crown = blob(905, 395, 72, 62, 7, 60)
    if tp(0) > 0:
        f.erase(crown, seed=60)
        f.fill(crown, (90, 160, 80), alpha=0.8 * tp(0.9), seed=60)
        f.fill(poly((888, 560), (893, 440), (914, 440), (918, 560), closed=True), (150, 100, 60),
               alpha=0.7 * tp(0.9), seed=61)
    f.stroke(line((888, 560), (893, 462)), p=tp(0, 0.3), seed=62)
    f.stroke(line((918, 560), (914, 462)), p=tp(0.1, 0.3), seed=63)
    f.stroke(crown, p=tp(0.3, 0.7), seed=60)
    for i, (ax, ay) in enumerate([(880, 390), (930, 370), (915, 420)]):  # りんご
        f.fill(circle(ax, ay, 8, seed=64 + i, span=6.3), (215, 60, 50), alpha=0.9 * tp(1.2 + i * 0.1, 0.2), seed=64 + i, offset=(0, 0))

    # ---- 鳥
    if 9.5 < t < 13:
        bx = lerp(-60, 1350, (t - 9.5) / 3.5)
        for k in range(2):
            x, y = bx - k * 70, 230 + k * 30 + 10 * math.sin(t * 3 + k)
            flap = 10 * math.sin(t * 14 + k * 2)
            f.stroke(poly((x - 18, y - flap), (x - 8, y - 6), (x, y), (x + 8, y - 6), (x + 18, y - flap)),
                     seed=90 + k, width=2.5)

    # ---- キャプション
    caption(f, t, "あさ", 5.5, T_RAIN, 101)
    caption(f, t, "あめ", T_RAIN + 0.5, T_RAINBOW, 102)
    caption(f, t, "にじ", T_RAINBOW + 0.6, T_DUSK, 103)
    caption(f, t, "ゆうがた", T_DUSK + 0.3, T_NIGHT + 0.5, 104)

    # ---- 色調（時間帯）
    f.tint(sky_color(t))

    # ---- 夜: 光るもの
    if night > 0:
        f.glow(652, 454, 60, (255, 210, 110), 0.55 * night)
        f.fill(poly((625, 430), (680, 430), (680, 478), (625, 478), closed=True), (255, 225, 130),
               alpha=night, seed=39, offset=(0, 0))
        f.stroke(poly((625, 430), (680, 430), (680, 478), (625, 478), closed=True), seed=39, alpha=int(200 * night))
        mp = prog(t, T_NIGHT + 0.5, 1.0)
        moon = circle(230, 130, 50, seed=80, a0=-1.9, span=3.8) + circle(250, 118, 40, seed=81, a0=1.75, span=-3.5)[::1]
        f.glow(230, 130, 70, (255, 245, 200), 0.35 * mp)
        f.fill(moon, (255, 240, 180), alpha=mp, seed=80, offset=(0, 0))
        f.stroke(moon, p=mp, seed=80, color=(240, 210, 120), width=3)
        for i, (sx, sy) in enumerate(STARS):
            sp = prog(t, T_NIGHT + 0.8 + i * 0.12, 0.3)
            tw = 0.6 + 0.4 * math.sin(t * 5 + i * 1.9)
            s = 9 * tw
            f.stroke(line((sx - s, sy), (sx + s, sy), over=0), p=sp, seed=400 + i, color=(255, 240, 170), width=2.5, passes=1)
            f.stroke(line((sx, sy - s), (sx, sy + s), over=0), p=sp, seed=410 + i, color=(255, 240, 170), width=2.5, passes=1)
        caption(f, t, "よる", T_NIGHT + 0.6, T_END - 0.2, 105)
        # おわり
        ea = prog(t, T_END, 0.6)
        if ea > 0:
            f.text("おやすみなさい", 640, 300, 80, p=prog(t, T_END, 1.2), color=(255, 245, 220), alpha=ea, seed=6, center=True)
            f.text("おしまい", 1120, 660, 36, p=prog(t, T_END + 1.2, 0.5), color=(255, 245, 220),
                   alpha=prog(t, T_END + 1.2, 0.3), seed=7, center=True)

    # フェードアウト
    fo = prog(t, DURATION - 0.6, 0.6)
    img = f.result()
    if fo > 0:
        img = Image.blend(img, Image.new("RGB", img.size, (20, 20, 30)), fo)
    return img


def main():
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    n = int(DURATION * FPS)
    only = [int(a) for a in sys.argv[1:]]  # 引数に秒数を渡すとその時刻の静止画だけ書き出す
    if only:
        for s in only:
            render(float(s), s * FPS).save(f"frame_{s:02d}.png")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            t = i / FPS
            render(t, i).save(os.path.join(tmp, f"{i:05d}.png"))
            if i % FPS == 0:
                print(f"{t:5.1f}s / {DURATION}s", flush=True)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", os.path.join(tmp, "%05d.png"), "-vf", "fps=24,format=yuv420p",
                        "-c:v", "libx264", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", OUT],
                       check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
