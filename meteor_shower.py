"""セル画風アニメーション「流星の夜」を生成するスクリプト（葬送のフリーレン ファンアート）。

丘の上に並んで座るフリーレン・フェルン・シュタルクを後ろから描き、
夜空いっぱいの流星群をいっしょに見上げる。空からゆっくりティルトダウンして
一行に寄っていく 1 カット構成。

    pip install pillow numpy imageio-ffmpeg
    python meteor_shower.py          # -> frieren_meteor_shower.mp4
    python meteor_shower.py 8 14.6   # その時刻の静止画だけ書き出す
"""

import math
import subprocess
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720  # 出力解像度
WW, WH = 2400, 2000  # ワールド（背景美術）の大きさ
FPS = 24
DURATION = 20.0
FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
OUT = "frieren_meteor_shower.mp4"

LINE = (22, 18, 38)
GOLD = (220, 180, 90)
SKIN = (246, 216, 200)

# 一行の配置（ワールド座標、腰の位置）
PARTY_BOX = (700, 1300, 1800, 1720)
FERN_POS, FRIEREN_POS, STARK_POS = (950, 1650), (1185, 1652), (1440, 1652)
CHAR_S = 1.25


# ---------------------------------------------------------------- 補助

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def prog(t, t0, dur):
    return clamp((t - t0) / dur)


def lerp(a, b, k):
    return a + (b - a) * k


def crest(x):
    """手前の丘の稜線の y。"""
    return 1640 + 0.00012 * (x - 1200) ** 2


def ridge(rng, y0, amp, n=2400, octaves=6):
    xs = np.linspace(0, WW, n)
    y = np.full(n, float(y0))
    for o in range(octaves):
        f = (o + 1) ** 1.7 / WW * 2 * math.pi
        y -= amp / (o + 1) ** 1.2 * np.abs(np.sin(xs * f + rng.uniform(0, 6.3)))
    return list(zip(xs, y))


# ---------------------------------------------------------------- 背景美術

def make_sky():
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:WH, 0:WW].astype(np.float32)
    k = (yy / 1500)[..., None].clip(0, 1)
    top, mid, low = (np.array(c, np.float32) for c in ((6, 9, 30), (18, 34, 84), (86, 92, 160)))
    sky = np.where(k < 0.6, lerp(top, mid, k / 0.6), lerp(mid, low, np.clip((k - 0.6) / 0.4, 0, 1) ** 1.6))
    # 地平線の紫がかった残光
    glow = np.exp(-((yy - 1480) / 260) ** 2)[..., None] * np.array([40, 22, 50], np.float32)
    sky += glow

    # 天の川: 斜めの帯 + ぼかしたノイズ
    small = rng.random((WH // 20, WW // 20)).astype(np.float32)
    noise = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((WW, WH), Image.BICUBIC), np.float32) / 255
    fine = rng.random((WH // 6, WW // 6)).astype(np.float32)
    fine = np.asarray(Image.fromarray((fine * 255).astype(np.uint8)).resize((WW, WH), Image.BICUBIC), np.float32) / 255
    d = (yy - (1350 - xx * 0.55)) / 230
    band = np.exp(-d ** 2) * (0.55 + 0.6 * noise * fine)
    dust = np.exp(-(d / 0.35) ** 2) * (noise > 0.55) * 0.35  # 暗黒帯
    sky += band[..., None] * np.array([70, 75, 120], np.float32)
    sky += (band ** 2)[..., None] * np.array([60, 40, 70], np.float32)
    sky -= dust[..., None] * np.array([40, 40, 60], np.float32)

    img = Image.fromarray(np.clip(sky, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(4200):
        x, y = rng.uniform(0, WW), rng.uniform(0, 1550) ** 1.0
        b = rng.uniform(0.25, 1) ** 2.2
        near_band = math.exp(-((y - (1350 - x * 0.55)) / 260) ** 2)
        if rng.random() > 0.35 + 0.65 * near_band and b < 0.3:
            continue
        c = tuple(int(lerp(sky[int(y), int(x), i], v, b)) for i, v in enumerate((235, 240, 255)))
        r = 0.6 + 1.4 * b
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
    return img


def make_stars(n=170):
    """またたく明るい星（出力フレームに毎回描く）。"""
    rng = np.random.default_rng(11)
    tint = [(255, 250, 240), (200, 220, 255), (255, 230, 200)]
    return [(rng.uniform(0, WW), rng.uniform(0, 1350), rng.uniform(1.2, 2.6), rng.uniform(0.8, 3.2),
             rng.uniform(0, 6.3), tint[rng.integers(3)]) for _ in range(n)]


def make_land_back():
    """遠景の山並みと手前の丘（キャラより奥）。"""
    rng = np.random.default_rng(3)
    img = Image.new("RGBA", (WW, WH), (0, 0, 0, 0))
    layers = [(1470, 170, (44, 52, 100)), (1540, 120, (30, 38, 76)), (1600, 80, (20, 26, 54))]
    for y0, amp, col in layers:
        pts = ridge(rng, y0, amp)
        lay = Image.new("RGBA", (WW, WH), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.polygon(pts + [(WW, WH), (0, WH)], fill=col + (255,))
        # 稜線のうっすらした縁取り
        d.line(pts, fill=tuple(min(255, c + 40) for c in col) + (160,), width=2)
        img.alpha_composite(lay)
    # 遠くの森
    d = ImageDraw.Draw(img)
    for x in np.arange(0, WW, 7):
        top = 1630 + 30 * math.sin(x / 190) + rng.uniform(-6, 6)
        h = rng.uniform(20, 55)
        d.polygon([(x - 9, top + 40), (x, top - h + 40), (x + 9, top + 40)], fill=(16, 22, 44, 255))
    # 丘
    xs = np.arange(0, WW + 1, 4)
    hill = [(x, crest(x)) for x in xs]
    d.polygon(hill + [(WW, WH), (0, WH)], fill=(14, 20, 38, 255))
    rim = Image.new("RGBA", (WW, WH), (0, 0, 0, 0))
    ImageDraw.Draw(rim).line(hill, fill=(120, 150, 220, 200), width=4)
    img.alpha_composite(rim.filter(ImageFilter.GaussianBlur(2)))
    grass(d, rng, 3000, 0, WW, 14, 34)
    # 丘の下へ向かって暗く
    yy = np.arange(WH, dtype=np.float32)
    shade = np.clip((yy - 1680) / 320, 0, 1)[:, None] * 120
    a = np.asarray(img).copy()
    a[..., :3] = np.clip(a[..., :3].astype(np.float32) - shade[..., None] * 0.1, 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def grass(d, rng, n, x0, x1, hmin, hmax, dy=6):
    for _ in range(n):
        x = rng.uniform(x0, x1)
        y = crest(x) + dy + rng.uniform(-2, 8)
        h = rng.uniform(hmin, hmax)
        lean = rng.uniform(-8, 8)
        d.polygon([(x - 3, y), (x + lean, y - h), (x + 3, y)], fill=(14, 20, 38, 255))
        d.line([(x + lean * 0.6, y - h * 0.6), (x + lean, y - h)], fill=(70, 95, 160, 255), width=1)


def make_land_front():
    """キャラの腰を隠す手前の草。"""
    rng = np.random.default_rng(5)
    img = Image.new("RGBA", (WW, WH), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(x, crest(x) + 14) for x in range(0, WW + 1, 4)] + [(WW, WH), (0, WH)], fill=(12, 17, 32, 255))
    grass(d, rng, 1800, 700, 1800, 12, 30, dy=16)
    return img


# ---------------------------------------------------------------- キャラクター（背中ごし・セル画風）

class Pen:
    """2 倍解像度のパッチに、ワールド座標でキャラを描く。"""

    def __init__(self):
        x0, y0, x1, y1 = PARTY_BOX
        self.x0, self.y0 = x0, y0
        self.img = Image.new("RGBA", ((x1 - x0) * 2, (y1 - y0) * 2), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def at(self, origin, s=CHAR_S):
        ox, oy = origin
        return lambda px, py: ((ox - self.x0 + px * s) * 2, (oy - self.y0 + py * s) * 2)

    def poly(self, pts, fill, line=True, width=5):
        self.d.polygon(pts, fill=fill + (255,))
        if line:
            self.d.line(pts + [pts[0]], fill=LINE + (255,), width=width, joint="curve")

    def ellipse(self, c, rx, ry, fill, line=True, width=5, n=40):
        rx, ry = rx * CHAR_S * 2, ry * CHAR_S * 2  # 半径はキャラのローカル単位
        pts = [(c[0] + rx * math.cos(a), c[1] + ry * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]
        self.poly(pts, fill, line, width)

    def stroke(self, pts, fill, width=4):
        self.d.line(pts, fill=fill + (255,), width=width, joint="curve")


def strand(P, root, length, wmax, tip_dx, sway, n=18):
    """根元が細く、中ほどで膨らみ、先が尖る髪の房。"""
    left, right = [], []
    for i in range(n + 1):
        v = i / n
        cx = root[0] + tip_dx * v ** 1.5 + sway * v ** 2
        cy = root[1] + length * v
        w = 3 + wmax * math.sin(math.pi * min(1.0, 0.15 + v * 0.9)) ** 0.7 * (1 - 0.35 * v)
        left.append(P(cx - w, cy))
        right.append(P(cx + w, cy))
    return left + [P(root[0] + tip_dx + sway, root[1] + length + wmax * 0.5)] + right[::-1]


def hair_lines(pen, P, c, rx, ry, col, k=5, spread=0.8):
    for i in range(k):
        a = (i - (k - 1) / 2) / k * spread
        pts = [P(c[0] + rx * math.sin(a) * v * 1.1, c[1] - ry * 0.85 + ry * 1.7 * v) for v in np.linspace(0, 1, 8)]
        pen.stroke(pts, col, width=3)


def frieren(pen, t, look=0.0):
    P = pen.at(FRIEREN_POS)
    sway = 5 * math.sin(t * 1.4) + 3 * math.sin(t * 2.3 + 1)
    # 杖（肩に立てかける）
    pen.stroke([P(38, 8), P(72, -232)], LINE, width=15)
    pen.stroke([P(38, 8), P(72, -232)], (126, 84, 52), width=9)
    ring = P(76, -250)
    for r, col, w in ((17, LINE, 14), (17, GOLD, 8)):
        r *= CHAR_S * 2
        pen.d.ellipse([ring[0] - r, ring[1] - r * 1.1, ring[0] + r, ring[1] + r * 1.1], outline=col + (255,), width=w)
    pen.ellipse(ring, 6, 6, (210, 50, 60), width=4)
    # からだ（白いワンピース）とケープ
    pen.poly([P(-50, 6), P(-45, -40), P(-36, -80), P(-26, -98), P(26, -98), P(36, -80), P(45, -40), P(50, 6)],
             (236, 236, 242))
    pen.stroke([P(-47, -20), P(47, -20)], GOLD, width=6)
    cape = [P(-42, -68), P(-40, -88), P(-30, -101), P(-14, -108), P(14, -108), P(30, -101), P(40, -88), P(42, -68),
            P(21, -62), P(0, -60), P(-21, -62)]
    pen.poly(cape, (248, 248, 252))
    pen.stroke(cape[-4:] + [cape[0]], GOLD, width=6)
    pen.stroke([P(0, -108), P(0, -60)], GOLD, width=4)
    # ツインテール（結び目は頭の陰）
    hc = (0, -136 - 4 * look)
    for d in (-1, 1):
        root = (d * 24, hc[1] - 10)
        pen.poly(strand(P, root, 126, 13, d * 16, sway * (1 + 0.3 * d)), (226, 229, 238))
        pen.stroke(strand(P, root, 126, 5, d * 16, sway * (1 + 0.3 * d))[4:15], (190, 196, 214), 3)
    # エルフの耳と耳飾り
    for d in (-1, 1):
        pen.poly([P(d * 22, hc[1] - 4), P(d * 64, hc[1] - 20 - 4 * look), P(d * 26, hc[1] + 14)], SKIN)
        pen.ellipse(P(d * 33, hc[1] + 16), 3, 3, (215, 45, 55), width=3)
    # 頭
    pen.ellipse(P(*hc), 31, 33, (228, 230, 238))
    hair_lines(pen, P, hc, 30, 32, (192, 197, 214))


def fern(pen, t, look=0.0):
    P = pen.at(FERN_POS)
    sway = 3 * math.sin(t * 1.2 + 0.7)
    pen.poly([P(-60, 8), P(-54, -50), P(-46, -112), P(-36, -134), P(36, -134), P(46, -112), P(54, -50), P(60, 8)],
             (40, 36, 48))
    pen.stroke([P(-58, -10), P(58, -10)], (70, 64, 84), width=4)
    hc = (4, -174 - 4 * look)
    hair = [P(-30, -182), P(-37, -150), P(-41, -100), P(-43, -44), P(-38 + sway, -12), P(0 + sway * 1.3, -4),
            P(38 + sway, -12), P(43, -44), P(41, -100), P(37, -150), P(30, -182)]
    pen.poly(hair, (126, 90, 162))
    for x in (-24, -8, 8, 24):
        pen.stroke([P(x * 0.8, -150), P(x, -90), P(x * 1.05 + sway * 0.8, -18)], (98, 66, 130), width=3)
    pen.ellipse(P(*hc), 31, 34, (126, 90, 162))
    hair_lines(pen, P, hc, 30, 33, (98, 66, 130))
    # 蝶の髪飾り
    for d in (-1, 1):
        pen.poly([P(hc[0], hc[1] + 6), P(hc[0] + d * 14, hc[1] - 4), P(hc[0] + d * 12, hc[1] + 14)], (236, 232, 244),
                 width=3)


def stark(pen, t, look=0.0):
    P = pen.at(STARK_POS)
    # 斧（地面に置く）
    pen.stroke([P(80, 4), P(200, -18)], LINE, width=16)
    pen.stroke([P(80, 4), P(200, -18)], (110, 76, 48), width=10)
    pen.poly([P(178, -14), P(170, -58), P(214, -70), P(236, -36), P(222, 6), P(196, 0)], (150, 158, 176))
    pen.stroke([P(176, -52), P(212, -64), P(230, -36)], (205, 212, 228), width=4)
    # からだ
    pen.poly([P(-68, 8), P(-63, -60), P(-68, -144), P(-50, -164), P(50, -164), P(68, -144), P(63, -60), P(68, 8)],
             (52, 60, 88))
    pen.stroke([P(0, -160), P(0, 4)], (38, 44, 66), width=4)
    # 後ろ手をついた腕
    for d in (-1, 1):
        pen.stroke([P(d * 60, -148), P(d * 86, -70), P(d * 92, 2)], LINE, width=38)
        pen.stroke([P(d * 60, -148), P(d * 86, -70), P(d * 92, 2)], (60, 70, 100), width=28)
        pen.ellipse(P(d * 93, 4), 7, 5, SKIN, width=4)
    pen.poly([P(-14, -176), P(14, -176), P(12, -158), P(-12, -158)], SKIN, width=4)
    hc = (0, -198 - 5 * look)
    spikes = []
    for i in range(15):
        a = math.pi * (0.95 + 1.1 * i / 14)
        r = 38 if i % 2 else 30
        spikes.append(P(hc[0] + r * math.cos(a) * 1.05, hc[1] + r * math.sin(a)))
    base = [P(hc[0] + 33 * math.cos(a), hc[1] + 30 * math.sin(a)) for a in np.linspace(0.1, math.pi - 0.1, 12)]
    pen.poly(base + spikes, (188, 66, 50))
    for d in (-1, 0, 1):
        pen.stroke([P(d * 10, hc[1] - 30), P(d * 16, hc[1] + 6)], (150, 46, 38), width=3)


def light_patch(img, flash):
    """夜の色味に沈め、空側（上）から青いリムライトを入れる。"""
    a = np.asarray(img).astype(np.float32)
    rgb, al = a[..., :3], a[..., 3:] / 255
    rgb = rgb * np.array([0.52, 0.58, 0.82]) + np.array([4, 8, 22])
    up = np.zeros_like(al)
    up[10:] = al[:-10]
    rim = np.clip(al - up, 0, 1)
    rim = np.asarray(Image.fromarray((rim[..., 0] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),
                     np.float32)[..., None] / 255
    rgb += rim * np.array([120, 150, 230]) * (0.8 + flash)
    rgb += flash * 60 * al
    out = np.concatenate([np.clip(rgb, 0, 255), al * 255], -1).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def party(t, flash, look):
    pen = Pen()
    fern(pen, t, look)
    frieren(pen, t, look)
    stark(pen, t, look)
    x0, y0, x1, y1 = PARTY_BOX
    return light_patch(pen.img, flash).resize((x1 - x0, y1 - y0), Image.LANCZOS)


# ---------------------------------------------------------------- 流星

RADIANT = np.array([2300.0, -500.0])


def make_meteors():
    rng = np.random.default_rng(21)
    ms, t = [], 0.3
    while t < DURATION:
        rate = 1.6 + 3.2 * prog(t, 3, 9)  # 時間とともに数が増える
        t += rng.exponential(1 / rate)
        # そのときの画面に入る空から降らせる
        x0, y0, w = camera(min(t, DURATION))
        h = w * H / W
        start = np.array([rng.uniform(x0 - 0.1 * w, x0 + 1.1 * w), rng.uniform(y0 - 0.15 * h, y0 + 0.5 * h)])
        d = start - RADIANT
        d /= np.linalg.norm(d)
        sc = w / WW  # 寄るほど流星も小さく
        ms.append(dict(t0=t, p=start, d=d, v=rng.uniform(1400, 2600) * sc, dur=rng.uniform(0.35, 0.8),
                       L=rng.uniform(160, 420) * sc, b=rng.uniform(0.45, 1.0), w=rng.uniform(1.0, 2.0) * sc ** 0.5))
    # クライマックスの大火球
    ms.append(dict(t0=14.2, p=np.array([1900.0, 930.0]), d=np.array([-0.9, 0.36]), v=420, dur=2.2,
                   L=700, b=1.6, w=4.5))
    return ms


def draw_meteors(t, ms, cam):
    x0, y0, sc = cam
    layer = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    flash = 0.0
    for m in ms:
        age = t - m["t0"]
        if not 0 <= age <= m["dur"]:
            continue
        k = age / m["dur"]
        env = math.sin(math.pi * k) ** 0.6
        head = m["p"] + m["d"] * m["v"] * age
        L = m["L"] * clamp(age * m["v"] / m["L"]) * (0.4 + 0.6 * env)
        n = 16
        for i in range(n):
            a, b = i / n, (i + 1) / n
            p0 = head - m["d"] * L * (1 - a)
            p1 = head - m["d"] * L * (1 - b)
            br = env * m["b"] * b ** 1.8
            col = tuple(int(clamp(br) * c) for c in (lerp(150, 255, b), lerp(200, 250, b), 255))
            d.line([((p0[0] - x0) * sc, (p0[1] - y0) * sc), ((p1[0] - x0) * sc, (p1[1] - y0) * sc)], fill=col,
                   width=max(1, int(m["w"] * sc * (0.5 + b) * 1.4)))
        hx, hy = (head[0] - x0) * sc, (head[1] - y0) * sc
        r = m["w"] * sc * 1.6 * env
        d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=(255, 252, 240))
        if m["b"] > 1:
            flash = max(flash, 0.35 * env)
    glow = layer.filter(ImageFilter.GaussianBlur(7))
    wide = layer.filter(ImageFilter.GaussianBlur(22))
    return ImageChops.add(ImageChops.add(layer, glow), wide), flash


def draw_twinkles(img, t, stars, cam):
    x0, y0, sc = cam
    d = ImageDraw.Draw(img)
    for x, y, r, f, ph, col in stars:
        k = 0.55 + 0.45 * math.sin(t * f + ph)
        px, py = (x - x0) * sc, (y - y0) * sc
        if not (-10 < px < W + 10 and -10 < py < H + 10):
            continue
        rr = r * sc * (0.7 + 0.5 * k)
        c = tuple(int(v * (0.5 + 0.5 * k)) for v in col)
        d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=c)
        if r > 2.1:  # 光条
            s = rr * 4 * k
            d.line([(px - s, py), (px + s, py)], fill=c, width=1)
            d.line([(px, py - s), (px, py + s)], fill=c, width=1)


# ---------------------------------------------------------------- 撮影

def camera(t):
    """(左上 x, 左上 y, 表示幅)。空 → ティルトダウン → 一行へ寄る。"""
    tilt = ease(prog(t, 2.5, 5.5))
    zoom = ease(prog(t, 9.0, 8.0))
    w = lerp(WW, 1500, zoom)
    h = w * H / W
    y_top = lerp(0, WH - 60 - h, tilt)
    cx = lerp(WW / 2, 1195, zoom)
    y_bot = lerp(y_top + h, 1790, zoom)
    return cx - w / 2, y_bot - h, w


SUBS = [
    (5.6, 8.2, "フェルン", "……きれいですね"),
    (8.8, 11.4, "シュタルク", "すげぇ……こんなにたくさん降ってくるのかよ"),
    (12.2, 14.0, "フリーレン", "……うん。"),
    (15.4, 18.2, "フリーレン", "前にも見たんだ。みんなと"),
]


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def subtitle(img, t):
    d = ImageDraw.Draw(img)
    for t0, t1, who, s in SUBS:
        if t0 <= t <= t1:
            a = min(prog(t, t0, 0.25), 1 - prog(t, t1 - 0.25, 0.25))
            c = int(255 * a)
            text = f"{who}「{s}」"
            f = font(34)
            w = d.textlength(text, font=f)
            d.text(((W - w) / 2, H - 78), text, font=f, fill=(c, c, c), stroke_width=3,
                   stroke_fill=(int(10 * a), int(10 * a), int(20 * a)))


# ---------------------------------------------------------------- 合成

SKY = BACK = FRONT = STARS = METEORS = None


def setup():
    global SKY, BACK, FRONT, STARS, METEORS
    print("背景美術を準備中…", flush=True)
    SKY, BACK, FRONT = make_sky(), make_land_back(), make_land_front()
    STARS, METEORS = make_stars(), make_meteors()


def render(t):
    x0, y0, w = camera(t)
    h = w * H / W
    sc = W / w
    box = (int(x0), int(y0), int(x0 + w), int(y0 + h))
    frame = SKY.crop(box).resize((W, H), Image.LANCZOS)
    cam = (box[0], box[1], sc)
    draw_twinkles(frame, t, STARS, cam)
    met, flash = draw_meteors(t, METEORS, cam)
    frame = ImageChops.add(frame, met)

    look = ease(prog(t, 14.0, 1.2)) * (1 - ease(prog(t, 17.5, 1.5)))
    land = BACK.crop(box)
    px0, py0 = PARTY_BOX[:2]
    land.alpha_composite(party(t, flash, look), (px0 - box[0], py0 - box[1]))
    land.alpha_composite(FRONT.crop(box))
    frame = frame.convert("RGBA")
    frame.alpha_composite(land.resize((W, H), Image.LANCZOS))
    frame = frame.convert("RGB")

    if flash > 0:
        frame = Image.blend(frame, ImageChops.screen(frame, Image.new("RGB", (W, H), (70, 80, 120))), flash)
    subtitle(frame, t)

    # 黒フェード + タイトル
    fi = 1 - prog(t, 0, 1.2)
    fo = prog(t, DURATION - 2.0, 0.8)
    k = max(fi, fo)
    if k > 0:
        frame = Image.blend(frame, Image.new("RGB", (W, H)), k)
    if t > DURATION - 1.8:
        a = prog(t, DURATION - 1.8, 0.5)
        d = ImageDraw.Draw(frame)
        for s, size, y, c in (("流星の夜", 56, H / 2 - 50, 235), ("葬送のフリーレン ファンアート", 22, H / 2 + 30, 170)):
            f = font(size)
            tw = d.textlength(s, font=f)
            v = int(c * a)
            d.text(((W - tw) / 2, y), s, font=f, fill=(v, v, int(v * 1.05) if v < 243 else 255))
    return frame


def main():
    import imageio_ffmpeg

    setup()
    only = [float(a) for a in sys.argv[1:]]
    if only:
        for s in only:
            render(s).save(f"frame_{s:04.1f}.png")
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-vf", "format=yuv420p",
                             "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-movflags", "+faststart", OUT],
                            stdin=subprocess.PIPE)
    n = int(DURATION * FPS)
    for i in range(n):
        t = i / FPS
        proc.stdin.write(render(t).tobytes())
        if i % FPS == 0:
            print(f"{t:5.1f}s / {DURATION}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
