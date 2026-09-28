"""手描き風アニメーション「ねこと わたし」を生成するスクリプト。

紙のテクスチャ、ゆらぐ鉛筆線（ラインボイル）、クレヨン風の塗り、
12fps の「2コマ打ち」で、雨の日に子ねこを拾った女の子のお話を描く。

    pip install pillow numpy imageio-ffmpeg
    python hand_drawn_video.py          # -> neko_to_watashi.mp4
    python hand_drawn_video.py 8 20.5   # その時刻の静止画だけ書き出す
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
FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
OUT = "neko_to_watashi.mp4"

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


# ---------------------------------------------------------------- キャラクター

SKIN = (250, 214, 188)
HAIR = (112, 72, 48)
DRESS = (222, 92, 92)
CAT = (242, 168, 88)
CAT_LIGHT = (252, 226, 190)
PINK = (240, 140, 160)
UMBRELLA = (250, 206, 80)


def rot(x, y, a):
    return x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)


def stage(p, i, n):
    """全体の進み具合 p を n 段に分けたときの i 段目の進み具合。"""
    return clamp(p * n - i)


def shape(f, pts, color, p=1.0, seed=0, alpha=0.85, width=3.0, offset=(3, -2)):
    """不透明な塗り + 鉛筆の輪郭。背後のものを隠す。"""
    if p <= 0:
        return
    f.erase(pts, seed=seed)
    f.fill(pts, color, alpha=alpha * ease(p * 1.5 - 0.5), seed=seed, offset=offset)
    f.stroke(pts, p=p, seed=seed, width=width)


def dot(f, x, y, r=4.5, color=PENCIL, alpha=1.0, seed=0):
    if alpha > 0:
        f.fill(circle(x, y, r, seed=seed, span=6.3, wob=0.1), color, alpha=alpha, seed=seed, offset=(0, 0))


def eye(f, x, y, s, kind, p, seed):
    if p <= 0:
        return
    if kind == "open":
        dot(f, x, y, 4.8 * s, alpha=p, seed=seed)
        dot(f, x + 1.5 * s, y - 1.5 * s, 1.4 * s, color=(255, 255, 255), alpha=p, seed=seed + 1)
    elif kind == "sad":
        dot(f, x, y + 2 * s, 4.8 * s, alpha=p, seed=seed)
        d = -1 if x < 0 else 1
        f.stroke(line((x - 7 * s, y - 10 * s + d * 2 * s), (x + 7 * s, y - 10 * s - d * 2 * s), over=0),
                 p=p, seed=seed, width=2)
    elif kind == "happy":  # ∩
        f.stroke(circle(x, y + 3 * s, 7 * s, a0=math.pi + 0.35, span=math.pi - 0.7, wob=0), p=p, seed=seed, width=3)
    else:  # closed ∪
        f.stroke(circle(x, y - 3 * s, 7 * s, a0=0.35, span=math.pi - 0.7, wob=0), p=p, seed=seed, width=3)


def heart(f, cx, cy, size, p=1.0, alpha=1.0, seed=0):
    if p <= 0 or alpha <= 0:
        return
    pts = []
    for i in range(61):
        a = 2 * math.pi * i / 60
        x = 16 * math.sin(a) ** 3
        y = 13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)
        pts.append((cx + x * size / 16, cy - y * size / 16))
    f.fill(pts, PINK, alpha=0.9 * alpha * p, seed=seed, offset=(2, -1))
    f.stroke(pts, p=p, seed=seed, color=(200, 70, 100), width=3, alpha=int(220 * alpha))


def girl_head(f, cx, cy, s=1.0, p=1.0, eyes="open", seed=500):
    P = lambda x, y: (cx + x * s, cy + y * s)
    face = circle(cx, cy, 38 * s, seed=seed, span=6.3, wob=0.03)
    shape(f, face, SKIN, p=stage(p, 0, 3), seed=seed, alpha=0.9)
    hp = stage(p, 1, 3)
    hair = [P(math.cos(a) * 46, math.sin(a) * 46) for a in [math.pi + i * math.pi / 20 for i in range(21)]]
    hair += [P(47, 30), P(34, 32), P(31, 6), P(22, -10), P(10, -4), P(0, -12), P(-10, -4), P(-22, -10),
             P(-31, 6), P(-34, 32), P(-47, 30), hair[0]]
    shape(f, hair, HAIR, p=hp, seed=seed + 1, alpha=0.9)
    bow = poly(P(28, -38), P(48, -52), P(46, -24), P(28, -38), P(12, -56), P(10, -28), closed=True)
    f.fill(bow, PINK, alpha=0.9 * hp, seed=seed + 2, offset=(0, 0))
    f.stroke(bow, p=hp, seed=seed + 2, width=2.5)
    fp = stage(p, 2, 3)
    eye(f, cx - 13 * s, cy + 9 * s, s, eyes, fp, seed + 3)
    eye(f, cx + 13 * s, cy + 9 * s, s, eyes, fp, seed + 5)
    for d in (-1, 1):
        f.fill(circle(cx + d * 24 * s, cy + 20 * s, 7 * s, span=6.3, seed=seed + 7), PINK, alpha=0.45 * fp,
               seed=seed + 7, offset=(0, 0))
    f.stroke(circle(cx, cy + 20 * s, 7 * s, a0=0.5, span=math.pi - 1.0, wob=0), p=fp, seed=seed + 8, width=2.5)


def girl(f, x, y, s=1.0, p=1.0, walk=None, pose="stand", eyes="open", hand=None, seed=500):
    """女の子。(x, y) は足元。hand は右手の位置（スクリーン座標）。"""
    dy = 56 if pose == "sit" else 0
    P = lambda px, py: (x + px * s, y + (py + dy) * s)
    if pose == "stand":
        sw = 14 * math.sin(walk) if walk is not None else 0
        for k, d in enumerate((-1, 1)):
            foot = P(d * 10 + d * sw, -dy)
            f.stroke(line(P(d * 10, -70), foot, over=0), p=stage(p, 0, 5), seed=seed + 20 + k, width=3.5)
            f.fill(circle(foot[0] + d * 4 * s, foot[1] - 4 * s, 9 * s, ry=6 * s, span=6.3, seed=seed + 22 + k),
                   (120, 60, 50), alpha=stage(p, 0, 5), seed=seed + 22 + k, offset=(0, 0))
    else:
        for k, d in enumerate((-1, 1)):
            shape(f, circle(x + d * 26 * s, y - 8 * s, 15 * s, ry=8 * s, span=6.3, seed=seed + 22 + k),
                  (120, 60, 50), p=stage(p, 0, 5), seed=seed + 22 + k)
    dress = poly(P(-20, -150), P(20, -150), P(46, -66), P(-46, -66), closed=True)
    shape(f, dress, DRESS, p=stage(p, 1, 5), seed=seed + 30)
    ap = stage(p, 2, 5)
    f.stroke(line(P(-18, -140), P(-36, -92), over=0), p=ap, seed=seed + 31, width=3.5)
    dot(f, *P(-37, -89), r=6 * s, color=SKIN, alpha=ap, seed=seed + 32)
    hx, hy = hand if hand else P(36, -92)
    f.stroke(line(P(18, -140), (hx, hy), over=0), p=ap, seed=seed + 33, width=3.5)
    dot(f, hx, hy, r=6 * s, color=SKIN, alpha=ap, seed=seed + 34)
    girl_head(f, *P(0, -190), s=s, p=clamp((p * 5 - 3) / 2), eyes=eyes, seed=seed)


def cat(f, x, y, s=1.0, p=1.0, pose="sit", eyes="open", tail=0.0, head=(0, 0), seed=600):
    """ねこ。(x, y) は足元。"""
    P = lambda px, py: (x + px * s, y + py * s)
    if pose == "sit":
        tp = stage(p, 0, 4)
        tl = [P(30 + 45 * u, -8 - 75 * u ** 1.5 + 12 * math.sin(tail + u * 3) * u) for u in [i / 20 for i in range(21)]]
        f.stroke(tl, p=tp, seed=seed, width=13)
        f.stroke(tl, p=tp, seed=seed, color=CAT, width=8, alpha=255, passes=1, amp=0.8)
        body = circle(*P(0, -38), 40 * s, ry=38 * s, span=6.3, seed=seed + 1)
        shape(f, body, CAT, p=stage(p, 1, 4), seed=seed + 1)
        f.fill(circle(*P(0, -30), 20 * s, ry=24 * s, span=6.3, seed=seed + 2), CAT_LIGHT, alpha=0.8 * stage(p, 1, 4),
               seed=seed + 2, offset=(0, 0))
        for k, d in enumerate((-1, 1)):
            shape(f, circle(*P(d * 15, -5), 11 * s, ry=7 * s, span=6.3, seed=seed + 3 + k), CAT_LIGHT,
                  p=stage(p, 1, 4), seed=seed + 3 + k, width=2.5)
        hc = P(head[0], -88 + head[1])
    else:  # 丸まって眠る
        body = circle(*P(0, -30), 64 * s, ry=30 * s, span=6.3, seed=seed + 1)
        shape(f, body, CAT, p=stage(p, 1, 4), seed=seed + 1)
        for k in range(3):
            f.stroke(circle(*P(10 + k * 16, -48), 10 * s, a0=0.3, span=2.2, wob=0), p=stage(p, 1, 4),
                     seed=seed + 5 + k, color=(200, 120, 50), width=3)
        hc = P(-48, -38)
    hp = stage(p, 2, 4)
    hx, hy = hc
    for k, d in enumerate((-1, 1)):
        ear = poly((hx + d * 28 * s, hy - 10 * s), (hx + d * 22 * s, hy - 44 * s), (hx + d * 5 * s, hy - 26 * s), closed=True)
        shape(f, ear, CAT, p=hp, seed=seed + 10 + k)
        f.fill(poly((hx + d * 22 * s, hy - 16 * s), (hx + d * 20 * s, hy - 34 * s), (hx + d * 10 * s, hy - 24 * s), closed=True),
               PINK, alpha=0.7 * hp, seed=seed + 12 + k, offset=(0, 0))
    shape(f, circle(hx, hy, 32 * s, ry=29 * s, span=6.3, seed=seed + 14), CAT, p=hp, seed=seed + 14)
    for k in range(3):  # しま模様
        f.stroke(line((hx + (k - 1) * 9 * s, hy - 28 * s), (hx + (k - 1) * 8 * s, hy - 17 * s), over=0),
                 p=hp, seed=seed + 15 + k, color=(200, 120, 50), width=3)
    fp = stage(p, 3, 4)
    eye(f, hx - 12 * s, hy - 2 * s, s * 0.9, eyes, fp, seed + 20)
    eye(f, hx + 12 * s, hy - 2 * s, s * 0.9, eyes, fp, seed + 22)
    f.fill(poly((hx - 4 * s, hy + 6 * s), (hx + 4 * s, hy + 6 * s), (hx, hy + 11 * s), closed=True), PINK,
           alpha=fp, seed=seed + 24, offset=(0, 0))
    for d in (-1, 1):
        f.stroke(circle(hx + d * 4.5 * s, hy + 12 * s, 4.5 * s, a0=0.2, span=math.pi - 0.4, wob=0), p=fp,
                 seed=seed + 25, width=2)
        for k in range(2):
            f.stroke(line((hx + d * 20 * s, hy + (8 + k * 7) * s), (hx + d * 40 * s, hy + (4 + k * 12) * s), over=0),
                     p=fp, seed=seed + 27 + k, width=1.8, alpha=180, passes=1)


def umbrella(f, cx, cy, a, p=1.0, seed=700):
    if p <= 0:
        return
    R = 95
    T = lambda x, y: (cx + rot(x, y, a)[0], cy + rot(x, y, a)[1])
    hp = stage(p, 0, 2)
    handle = [T(0, 0), T(0, 118)] + [T(-9 + 9 * math.cos(u), 118 + 9 * math.sin(u)) for u in [i * 0.3 for i in range(11)]]
    f.stroke(handle, p=hp, seed=seed, color=(110, 70, 40), width=4)
    top = [T(R * math.cos(u), 0.85 * R * math.sin(u)) for u in [math.pi + i * math.pi / 30 for i in range(31)]]
    scal = []
    for k in range(4):
        x0 = R - k * R / 2
        scal += [T(x0 - R / 2 * j / 8, -12 * math.sin(math.pi * j / 8)) for j in range(9)]
    canopy = top + scal
    shape(f, canopy, UMBRELLA, p=stage(p, 1, 2), seed=seed + 1)
    for k in (1, 2, 3):
        f.stroke(line(T(0, -0.85 * R), T(R - k * R / 2, -2), over=0), p=stage(p, 1, 2), seed=seed + 2 + k, width=2.5)
    f.stroke(line(T(0, -0.85 * R), T(0, -0.85 * R - 14), over=0), p=stage(p, 1, 2), seed=seed + 6, width=4)


def rain(f, t, x0, y0, x1, y1, n, alpha, cover=None, color=(80, 110, 170)):
    """雨粒。cover=(cx, cy) の傘の下には降らない。"""
    if alpha <= 0:
        return
    rnd = random.Random(99)
    h = y1 - y0
    for i in range(n):
        x = rnd.uniform(x0, x1 + 0.25 * h)
        spd = rnd.uniform(520, 700)
        yy = (rnd.uniform(0, h) + t * spd) % h + y0
        xx = x - (yy - y0) * 0.25
        if cover and abs(xx - cover[0]) < 92 and yy > cover[1] - 30:
            continue
        f.stroke(line((xx, yy), (xx - 5, yy + 20), over=0), seed=200 + i, color=color,
                 width=2.5, alpha=int(190 * alpha), passes=1, amp=0.5)


def caption(f, t, s, t0, t1, seed, x=70, y=70, color=PENCIL):
    a = prog(t, t0, 0.3) * (1 - prog(t, t1 - 0.4, 0.4))
    if a <= 0:
        return
    f.text(s, x, y, 50, p=prog(t, t0, 0.2 + 0.08 * len(s)), alpha=a, seed=seed, color=color)
    f.stroke(line((x - 4, y + 32), (x - 4 + 53 * len(s), y + 34)), p=prog(t, t0 + 0.4, 0.5),
             seed=seed, color=(200, 80, 60), width=4, alpha=int(200 * a))


def window(f, x0, y0, x1, y1, p, seed, glass=(200, 220, 240), glass_a=0.5):
    frame = poly((x0, y0), (x1, y0), (x1, y1), (x0, y1), closed=True)
    f.erase(frame, seed=seed)
    f.fill(frame, glass, alpha=glass_a * p, seed=seed, offset=(0, 0))
    return frame


def window_frame(f, x0, y0, x1, y1, p, seed):
    f.stroke(poly((x0, y0), (x1, y0), (x1, y1), (x0, y1), closed=True), p=p, seed=seed, width=4)
    f.stroke(line(((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1), over=0), p=p, seed=seed + 1, width=3.5)
    f.stroke(line((x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2), over=0), p=p, seed=seed + 2, width=3.5)
    f.stroke(line((x0 - 15, y1 + 4), (x1 + 15, y1 + 4)), p=p, seed=seed + 3, width=4)


# ---------------------------------------------------------------- シーン

T_A = 3.4   # 雨の日のであい
T_B = 13.4  # おうちで
T_C = 21.6  # 夜
DURATION = 31.0


def scene_title(f, t):
    a = 1 - prog(t, T_A - 0.6, 0.6)
    f.text("ねこと わたし", 560, 300, 96, p=prog(t, 0.2, 1.2), alpha=a, seed=1, center=True)
    f.stroke(poly(*[(300 + i * 10, 372 + 6 * math.sin(i * 0.55)) for i in range(53)]),
             p=prog(t, 1.3, 0.8), seed=2, color=(210, 90, 60), width=5, alpha=int(210 * a))
    f.text("〜 あめの日の ちいさな であい 〜", 560, 450, 36, p=prog(t, 1.8, 0.9), alpha=a, seed=3, center=True)
    if a > 0.3:
        cat(f, 1000, 470, s=1.1, p=prog(t, 1.2, 1.2), eyes="happy", tail=t * 3, seed=610)


def scene_rain(f, t):
    u = t - T_A
    # 道と塀
    f.fill(poly((-20, 560), (1300, 560), (1300, 740), (-20, 740), closed=True), (170, 170, 175),
           alpha=0.55 * prog(u, 0.4, 0.6), seed=10, offset=(0, 4))
    f.stroke([(x, 560 + 3 * math.sin(x * 0.02)) for x in range(-20, 1320, 20)], p=prog(u, 0, 0.7), seed=11, width=3.5)
    f.fill(poly((-20, 380), (1300, 380), (1300, 560), (-20, 560), closed=True), (215, 180, 140),
           alpha=0.4 * prog(u, 0.5, 0.6), seed=12, offset=(0, 0))
    f.stroke(line((-20, 380), (1300, 382)), p=prog(u, 0.1, 0.7), seed=13, width=3.5)
    for i in range(14):
        bx = 45 + i * 95
        f.stroke(line((bx, 380), (bx + 1, 560), over=0), p=prog(u, 0.3 + i * 0.04, 0.3), seed=14 + i, width=2.5, alpha=170)
    # 街灯
    f.stroke(line((1150, 560), (1152, 230)), p=prog(u, 0.5, 0.5), seed=30, width=5)
    f.stroke(poly((1152, 232), (1130, 215), (1100, 222)), p=prog(u, 0.8, 0.3), seed=31, width=4)
    shape(f, poly((1080, 222), (1120, 222), (1112, 245), (1088, 245), closed=True), (255, 230, 150),
          p=prog(u, 0.9, 0.3), seed=32)
    # 水たまり
    for k, (px, pw) in enumerate([(300, 90), (980, 70)]):
        pp = prog(u, 0.6 + k * 0.1, 0.4)
        f.fill(circle(px, 620 + k * 30, pw, ry=14, span=6.3, seed=40 + k), (150, 180, 220), alpha=0.6 * pp, seed=40 + k)
        f.stroke(circle(px, 620 + k * 30, pw, ry=14, seed=40 + k), p=pp, seed=40 + k, width=2.5)
        rr = (t * 1.3 + k * 0.5) % 1
        f.stroke(circle(px + 20, 620 + k * 30, 10 + 40 * rr, ry=3 + 7 * rr, seed=42 + k, wob=0), seed=42 + k,
                 color=(90, 120, 170), width=2, alpha=int(200 * (1 - rr) * pp), passes=1)

    # 箱の中のねこ
    look = prog(u, 6.6, 0.4)
    cat_eyes = "sad" if u < 6.6 else ("open" if u < 7.2 else "happy")
    shiver = 2.5 * math.sin(t * 40) * (1 - look)
    cat(f, 800 + shiver, 545, s=0.85, p=prog(u, 0.9, 1.0), eyes=cat_eyes, tail=0, head=(0, 10 - 14 * look), seed=620)
    box = poly((735, 490), (865, 490), (865, 562), (735, 562), closed=True)
    bp = prog(u, 0.7, 0.6)
    shape(f, box, (205, 160, 110), p=bp, seed=50)
    f.stroke(poly((735, 490), (712, 462), (740, 460)), p=bp, seed=51, width=3)
    f.stroke(poly((865, 490), (890, 465), (862, 460)), p=bp, seed=52, width=3)
    f.text("ひろってください", 800, 522, 17, p=prog(u, 1.4, 0.5), alpha=bp, seed=53, center=True)

    # 女の子が歩いてくる
    wk = prog(u, 1.5, 3.0)
    gx = lerp(-90, 650, wk)
    walking = 0 < wk < 1
    move = prog(u, 5.6, 1.0)
    if u > 1.5:
        s_c = (gx + 26, 560 - 118 - 130)
        c = (lerp(s_c[0], 790, move), lerp(s_c[1], 345, move))
        a = lerp(0, 0.6, move)
        hand = (c[0] - 118 * math.sin(a), c[1] + 118 * math.cos(a))
        g_eyes = "happy" if u > 7.2 else "open"
        girl(f, gx, 560 - (3 * abs(math.sin(gx * 0.05)) if walking else 0), walk=gx * 0.05 if walking else None,
             eyes=g_eyes, hand=hand, seed=500)
        umbrella(f, *c, a, seed=700)
        cover = c
        if u > 5.8:  # 女の子がぬれる
            for k in range(4):
                ph = (t * 1.5 + k / 4) % 1
                f.stroke(line((gx - 30 + k * 20, 340 + 60 * ph), (gx - 32 + k * 20, 350 + 60 * ph), over=0), seed=80 + k,
                         color=(80, 120, 190), width=3, alpha=int(200 * (1 - ph)), passes=1)
    else:
        cover = None

    f.text("？", gx + 55, 300, 44, p=prog(u, 4.6, 0.2), alpha=1 - prog(u, 5.6, 0.3), seed=60)
    hp = prog(u, 7.0, 0.5)
    heart(f, 800, 400 - 30 * prog(u, 7.0, 2.0), 22, p=hp, alpha=1 - prog(u, 8.6, 0.5), seed=61)

    rain(f, t, -40, 0, 1280, 700, 80, prog(u, 0.2, 0.6), cover=cover)

    caption(f, t, "あめの日", T_A + 0.9, T_A + 3.8, 101)
    caption(f, t, "ひとりぼっち？", T_A + 4.2, T_A + 6.6, 102)
    caption(f, t, "いっしょに かえろう", T_A + 7.0, T_B - 0.3, 103)
    f.tint((212, 218, 230))


def scene_home(f, t):
    u = t - T_B
    rp = prog(u, 0, 1.0)
    # 床とラグ
    f.fill(poly((-20, 560), (1300, 560), (1300, 740), (-20, 740), closed=True), (200, 150, 100),
           alpha=0.45 * rp, seed=10, offset=(0, 4))
    f.stroke(line((-20, 560), (1300, 562)), p=rp, seed=11, width=3.5)
    for i in range(6):
        f.stroke(line((-20, 600 + i * 26), (1300, 600 + i * 26), over=0), p=prog(u, 0.3 + i * 0.05, 0.6), seed=12 + i,
                 color=(160, 110, 70), width=2, alpha=110, passes=1)
    rug = circle(600, 600, 320, ry=48, span=6.3, seed=20)
    f.fill(rug, (170, 200, 230), alpha=0.7 * prog(u, 0.4, 0.6), seed=20)
    f.stroke(circle(600, 600, 320, ry=48, seed=20), p=prog(u, 0.3, 0.6), seed=20, width=3)
    # 窓（外は雨 → 虹）
    wp = prog(u, 0.2, 0.8)
    clear = prog(u, 5.0, 1.0)
    glass = lerp3((170, 180, 200), (185, 215, 245), clear)
    window(f, 840, 110, 1120, 330, wp, 30, glass=tuple(int(v) for v in glass), glass_a=0.6)
    rain(f, t, 840, 110, 1060, 330, 16, wp * (1 - clear))
    for i, c in enumerate([(225, 70, 70), (240, 160, 60), (240, 215, 80), (100, 180, 90), (80, 140, 220)]):
        f.stroke(circle(980, 360, 170 - i * 10, a0=math.pi + 0.15, span=math.pi - 0.3, wob=0.01),
                 p=prog(u, 5.4 + i * 0.15, 0.8), seed=40 + i, color=c, width=7, alpha=180)
    window_frame(f, 840, 110, 1120, 330, wp, 31)
    # 額縁（ねこの絵）
    fp = prog(u, 0.6, 0.6)
    shape(f, poly((200, 170), (330, 170), (330, 270), (200, 270), closed=True), (245, 235, 210), p=fp, seed=50)
    f.stroke(circle(265, 230, 22, seed=51), p=fp, seed=51, color=(230, 140, 70), width=3)
    f.stroke(poly((248, 214), (250, 196), (260, 208)), p=fp, seed=52, color=(230, 140, 70), width=3)
    f.stroke(poly((270, 208), (280, 196), (282, 214)), p=fp, seed=53, color=(230, 140, 70), width=3)

    # 女の子とねこ
    drink = prog(u, 4.0, 0.5) * (1 - prog(u, 5.8, 0.5))
    pet = prog(u, 6.0, 0.4)
    petting = 12 * math.sin(t * 7) * pet
    hand = (lerp(560, 594 + petting, pet), lerp(520, 460, pet))
    g_eyes = "happy" if u > 6.2 else "open"
    girl(f, 500, 585, pose="sit", p=prog(u, 1.2, 1.2), eyes=g_eyes, hand=hand, seed=500)
    shake = 5 * math.sin(t * 50) * (prog(u, 2.4, 0.2) - prog(u, 3.4, 0.2))
    c_eyes = "closed" if 2.4 < u < 3.4 or drink > 0.5 else ("happy" if u > 6.2 else "open")
    cat(f, 605 + shake, 585, s=0.9, p=prog(u, 1.4, 1.2), eyes=c_eyes, tail=t * 2.5,
        head=(45 * drink, 42 * drink), seed=620)
    for k in range(6):  # ぶるぶる
        if 2.4 < u < 3.5:
            a = k * math.pi / 3 + 0.3
            f.stroke(line((605 + math.cos(a) * 60, 520 + math.sin(a) * 60), (605 + math.cos(a) * 75, 520 + math.sin(a) * 75),
                          over=0), seed=70 + k, color=(80, 130, 200), width=3, passes=1)
    bp = prog(u, 3.4, 0.4)
    bowl = poly(*[(690 + 34 * math.cos(a), 562 + 16 * math.sin(a)) for a in [i * math.pi / 12 for i in range(13)]],
                (656, 562), closed=True)
    shape(f, bowl, (120, 160, 220), p=bp, seed=80)
    f.fill(circle(690, 562, 32, ry=6, span=6.3, seed=81), (255, 255, 250), alpha=bp, seed=81, offset=(0, 0))
    f.stroke(circle(690, 562, 34, ry=7, seed=81), p=bp, seed=81, width=2.5)
    if drink > 0.5:
        f.text("ぺろ ぺろ", 720, 480, 26, alpha=drink, seed=82 + int(t * 2) % 2)
    for k in range(3):
        st = 6.5 + k * 0.5
        hp = prog(u, st, 0.3)
        ph = clamp((u - st) / 1.6)
        heart(f, 560 + k * 40 + 10 * math.sin(t * 3 + k), 430 - 110 * ph, 16 + 3 * k, p=hp, alpha=1 - ph, seed=90 + k)

    caption(f, t, "ぬれちゃったね", T_B + 2.0, T_B + 4.6, 104)
    caption(f, t, "あったかいね", T_B + 5.6, T_C - 0.3, 105)
    f.tint((255, 246, 228))


def scene_night(f, t):
    u = t - T_C
    rp = prog(u, 0, 1.0)
    f.fill(poly((-20, 560), (1300, 560), (1300, 740), (-20, 740), closed=True), (200, 150, 100),
           alpha=0.45 * rp, seed=10, offset=(0, 4))
    f.stroke(line((-20, 560), (1300, 562)), p=rp, seed=11, width=3.5)
    wp = prog(u, 0.2, 0.8)
    window(f, 150, 110, 430, 330, wp, 30, glass=(60, 70, 120), glass_a=0.7)
    window_frame(f, 150, 110, 430, 330, wp, 31)
    # ベッド
    bp = lambda d: prog(u, 0.4 + d, 0.6)
    shape(f, poly((560, 330), (610, 330), (610, 575), (560, 575), closed=True), (180, 120, 80), p=bp(0), seed=40)
    shape(f, poly((600, 470), (1110, 470), (1110, 540), (600, 540), closed=True), (180, 120, 80), p=bp(0.2), seed=41)
    for lx in (620, 1090):
        f.stroke(line((lx, 540), (lx, 575)), p=bp(0.4), seed=42 + lx, width=4)
    shape(f, blob(680, 440, 62, 24, 5, 43), (250, 250, 245), p=bp(0.5), seed=43)
    # 眠る女の子とねこ
    girl_head(f, 690, 412, s=0.95, p=prog(u, 1.2, 1.0), eyes="closed", seed=500)
    blanket = [(700 + i * 20, 438 + 5 * math.sin(i * 0.8)) for i in range(21)] + [(1105, 450), (1110, 505), (705, 505)]
    blanket.append(blanket[0])
    shape(f, blanket, (240, 180, 190), p=bp(0.9), seed=44)
    for i in range(10):
        dot(f, 740 + (i % 5) * 75 + (i // 5) * 35, 460 + (i // 5) * 25, 6, color=(255, 240, 245),
            alpha=prog(u, 1.8, 0.4), seed=45 + i)
    cat(f, 930, 445, s=0.8, p=prog(u, 1.6, 1.2), pose="curl", eyes="closed", seed=620)

    night = prog(u, 0.6, 1.8)
    f.tint(lerp3((255, 246, 228), (95, 105, 165), night))

    # 月と星（窓の中）
    mp = prog(u, 2.2, 0.8)
    f.glow(290, 220, 90, (200, 210, 255), 0.25 * mp)
    moon = circle(250, 180, 38, seed=80, a0=-1.9, span=3.8) + circle(266, 170, 30, seed=81, a0=1.75, span=-3.5)
    f.fill(moon, (255, 240, 180), alpha=mp, seed=80, offset=(0, 0))
    f.stroke(moon, p=mp, seed=80, color=(240, 210, 120), width=3)
    for i, (sx, sy) in enumerate([(340, 150), (390, 250), (200, 280), (320, 290), (400, 140)]):
        sp = prog(u, 2.6 + i * 0.15, 0.3)
        s = 7 * (0.6 + 0.4 * math.sin(t * 5 + i * 1.9))
        for k, (dx, dy) in enumerate(((s, 0), (0, s))):
            f.stroke(line((sx - dx, sy - dy), (sx + dx, sy + dy), over=0), p=sp, seed=400 + i * 2 + k,
                     color=(255, 240, 170), width=2.5, passes=1)
    f.glow(690, 420, 110, (255, 220, 170), 0.12 * mp)
    # Zzz
    for k in range(4):
        st = 3.0 + k * 0.7
        ph = ((u - st) / 2.8) % 1 if u > st else -1
        if ph >= 0:
            sz = 22 + 10 * ph
            f.text("z", 740 + 60 * ph + (k % 2) * 180, 360 - 90 * ph - (k % 2) * 10, int(sz),
                   alpha=math.sin(math.pi * ph), seed=410 + k, color=(230, 230, 250))

    light = (250, 244, 225)
    caption(f, t, "おやすみ", T_C + 3.0, T_C + 5.8, 106, color=light)
    ea = prog(u, 6.0, 0.6)
    f.text("ずっと いっしょ", 820, 200, 72, p=prog(u, 6.0, 1.0), alpha=ea, seed=6, center=True, color=light)
    f.text("おしまい", 1130, 660, 34, p=prog(u, 7.4, 0.5), alpha=prog(u, 7.4, 0.3), seed=7, center=True, color=light)


def render(t, boil):
    f = Frame(t, boil)
    if t < T_A:
        scene_title(f, t)
    elif t < T_B:
        scene_rain(f, t)
    elif t < T_C:
        scene_home(f, t)
    else:
        scene_night(f, t)
    img = f.result()
    # 場面の切りかえ: 紙にフェード
    for end in (T_B, T_C):
        k = prog(t, end - 0.5, 0.5) if t < end else 0
        if k > 0:
            img = Image.blend(img, PAPER.resize((W, H)), k)
    fo = prog(t, DURATION - 0.7, 0.7)
    if fo > 0:
        img = Image.blend(img, Image.new("RGB", img.size, (20, 20, 30)), fo)
    return img


def main():
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    n = int(DURATION * FPS)
    only = [float(a) for a in sys.argv[1:]]  # 引数に秒数を渡すとその時刻の静止画だけ書き出す
    if only:
        for s in only:
            render(s, int(s * FPS)).save(f"frame_{s:04.1f}.png")
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
