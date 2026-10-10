"""注文票（JSON）から、手描き風のメッセージ動画を1本つくるテンプレート。

hand_drawn_video.py の画材（紙・ゆらぐ鉛筆線・クレヨン塗り・キャラクター）を使い、
タイトル → テンプレートごとの場面 → 写真とメッセージ → 「〜より」の4場面を組み立てる。
名前・メッセージ・写真・色を注文票に書くだけで、1本あたり数分で書き出せる。

    pip install pillow numpy
    python order_video.py orders/birthday.json            # 動画を書き出す
    python order_video.py orders/birthday.json --preview  # 確認用の静止画（5場面を1枚に）
    python order_video.py orders/birthday.json --sample   # 「SAMPLE」の透かし入り（入金前の確認用）

注文票の書き方は orders/README.md を参照。
"""

import argparse
import json
import math
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

import hand_drawn_video as hv
from hand_drawn_video import (FONT_PATH, FPS, H, PAPER, SS, W, Frame, caption, cat, circle, clamp, dot, girl,
                              heart, hill, lerp, line, poly, prog, rot, shape, sparkle, stage)

TEMPLATES = ("birthday", "wedding", "pet", "shop")

HAIR_COLORS = {
    "brown": (112, 72, 48),
    "black": (55, 48, 52),
    "blonde": (226, 186, 104),
    "pink": (232, 140, 160),
    "gray": (160, 160, 168),
}
DRESS_COLORS = {
    "red": (222, 92, 92),
    "blue": (92, 132, 214),
    "yellow": (240, 196, 72),
    "green": (96, 170, 110),
    "pink": (240, 150, 175),
    "purple": (160, 116, 200),
    "white": (246, 244, 236),
}
# (からだ, おなか・足先, しま模様)
CAT_COLORS = {
    "chatora": ((242, 168, 88), (252, 226, 190), (200, 120, 50)),   # 茶トラ
    "kuro": ((96, 92, 106), (130, 126, 140), (70, 66, 80)),         # 黒
    "shiro": ((252, 250, 246), (255, 255, 255), (226, 220, 212)),   # 白
    "haiiro": ((168, 168, 178), (218, 218, 226), (122, 122, 134)),  # グレー
    "kijitora": ((170, 136, 96), (236, 222, 196), (110, 84, 56)),   # キジトラ
}
CONFETTI = [(240, 120, 140), (250, 210, 80), (120, 180, 230), (130, 200, 130), (200, 150, 230)]
FLOWER = [(245, 150, 170), (250, 220, 90), (255, 255, 250), (150, 170, 240), (200, 150, 230)]

TITLE_DUR = 4.0
MAIN_DUR = 12.0
END_DUR = 5.0
ROW_DT = 1.8  # メッセージ1行あたりの秒数
MAX_ROWS = 6

O = {}  # 注文票の中身（setup で埋める）
T = {}  # 場面の開始時刻


# ---------------------------------------------------------------- 注文票

class OrderError(Exception):
    pass


def color(value, presets, what):
    if isinstance(value, list) and len(value) == 3:
        return tuple(int(v) for v in value)
    if value in presets:
        return presets[value]
    raise OrderError(f"{what} は {', '.join(presets)} か [R, G, B] で指定してください（{value!r}）")


def wrap(s, n):
    """1行 n 文字で折り返す。行末近くに空白があればそこで切る。"""
    rows = []
    s = s.strip()
    while len(s) > n:
        cut = s.rfind(" ", n - 5, n + 1)
        cut = n if cut <= 0 else cut
        rows.append(s[:cut].strip())
        s = s[cut:].strip()
    if s:
        rows.append(s)
    return rows


def load_order(path):
    with open(path, encoding="utf-8") as fp:
        o = json.load(fp)
    base = os.path.dirname(os.path.abspath(path))
    if o.get("template") not in TEMPLATES:
        raise OrderError(f"template は {', '.join(TEMPLATES)} のどれかにしてください")
    for key in ("title", "from"):
        if not o.get(key):
            raise OrderError(f"{key} を書いてください")
    msgs = o.get("messages") or []
    if not isinstance(msgs, list) or not msgs:
        raise OrderError("messages に1行以上のメッセージを書いてください")
    for key in ("photo", "music"):
        if o.get(key):
            o[key] = os.path.join(base, o[key])
            if not os.path.exists(o[key]):
                raise OrderError(f"{key} のファイルが見つかりません: {o[key]}")
    o["rows"] = [r for m in msgs for r in wrap(m, 14 if o.get("photo") else 22)]
    if len(o["rows"]) > MAX_ROWS:
        raise OrderError(f"メッセージが長すぎます（折り返して {len(o['rows'])} 行、{MAX_ROWS} 行まで）")
    scene = o.get("scene_captions") or []
    if len(scene) not in (0, 2) or any(len(c) > 20 for c in scene):
        raise OrderError("scene_captions は20文字以内を2つ書いてください")
    o["hair"] = color(o.get("hair", "brown"), HAIR_COLORS, "hair")
    o["dress"] = color(o.get("dress", "red"), DRESS_COLORS, "dress")
    o["accent"] = color(o.get("accent", "red"), DRESS_COLORS, "accent")
    for key in ("cat", "cat2"):
        v = o.get(key, "chatora" if key == "cat" else "shiro")
        if v not in CAT_COLORS:
            raise OrderError(f"{key} は {', '.join(CAT_COLORS)} のどれかにしてください（{v!r}）")
        o[key] = CAT_COLORS[v]
    o["candles"] = max(1, min(9, int(o.get("candles", 3))))
    return o


def setup(o):
    O.clear()
    O.update(o)
    hv.HAIR, hv.DRESS = o["hair"], o["dress"]
    set_cat(o["cat"])
    T["main"] = TITLE_DUR
    T["msg"] = T["main"] + MAIN_DUR
    T["end"] = T["msg"] + 1.6 + ROW_DT * len(o["rows"]) + 2.4
    T["dur"] = T["end"] + END_DUR
    O["photo_img"] = load_photo(o["photo"]) if o.get("photo") else None


def set_cat(colors):
    hv.CAT, hv.CAT_LIGHT, hv.CAT_STRIPE = colors


# ---------------------------------------------------------------- 写真

PHOTO_BOX = (110, 150, 530, 540)


def load_photo(path):
    """写真を色鉛筆で描いたような風合いにする。"""
    x0, y0, x1, y1 = PHOTO_BOX
    w, h = (x1 - x0) * SS, (y1 - y0) * SS
    img = ImageOps.fit(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), (w, h), Image.LANCZOS)
    soft = ImageOps.posterize(ImageEnhance.Color(img).enhance(0.85).filter(ImageFilter.SMOOTH_MORE), 5)
    soft = Image.blend(PAPER.crop((0, 0, w, h)), soft, 0.85)
    edges = ImageOps.grayscale(img).filter(ImageFilter.GaussianBlur(SS)).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges).point(lambda v: 255 - int(v * 0.45))
    return ImageChops.multiply(soft, Image.merge("RGB", (edges,) * 3))


def photo(f, p, seed=40):
    if p <= 0:
        return
    x0, y0, x1, y1 = PHOTO_BOX
    frame = poly((x0, y0), (x1, y0), (x1, y1), (x0, y1), closed=True)
    pad = poly((x0 - 14, y0 - 14), (x1 + 14, y0 - 14), (x1 + 14, y1 + 14), (x0 - 14, y1 + 14), closed=True)
    shape(f, pad, (255, 253, 248), p=p, seed=seed, alpha=0.9)
    m, box = f._mask(frame, seed, blur=0.6, amp=1.2)
    img = O["photo_img"].crop((box[0] - x0 * SS, box[1] - y0 * SS, box[2] - x0 * SS, box[3] - y0 * SS))
    f.img.paste(img, box[:2], m.point(lambda v: int(v * clamp(p * 2 - 1))))
    f.stroke(frame, p=p, seed=seed + 1, width=3)
    for k, (tx, ty) in enumerate(((x0 + 20, y0 - 6), (x1 - 20, y0 - 6))):  # マスキングテープ
        a = -0.5 if k == 0 else 0.5
        tape = [(tx + rot(dx, dy, a)[0], ty + rot(dx, dy, a)[1]) for dx, dy in ((-40, -12), (40, -12), (40, 12), (-40, 12))]
        f.fill(tape + tape[:1], (240, 200, 120), alpha=0.85 * p, seed=seed + 2 + k, offset=(0, 0))


# ---------------------------------------------------------------- 小物

def floor(f, u, rgb, alpha=0.45):
    f.fill(poly((-20, 560), (1300, 560), (1300, 740), (-20, 740), closed=True), rgb,
           alpha=alpha * prog(u, 0, 1.0), seed=10, offset=(0, 4))
    f.stroke(line((-20, 560), (1300, 562)), p=prog(u, 0, 1.0), seed=11, width=3.5)


def confetti(f, t, t0, n=70, seed=300):
    """紙ふぶき。上から少しずつ降りはじめ、そのあとは降りつづける。"""
    if t < t0:
        return
    rnd = random.Random(seed)
    for i in range(n):
        x0, spd = rnd.uniform(0, W), rnd.uniform(110, 190)
        y = -20 - rnd.uniform(0, 600) + (t - t0) * spd
        if y < -20:
            continue
        y = (y + 20) % 620 - 20
        x = x0 + 25 * math.sin(t * 2 + i)
        a = t * 3 + i
        pts = [rot(dx, dy * math.cos(t * 5 + i), a) for dx, dy in ((-7, -4), (7, -4), (7, 4), (-7, 4))]
        f.draw.polygon([((x + dx) * SS, (y + dy) * SS) for dx, dy in pts], fill=CONFETTI[i % 5] + (230,))


def garland(f, p, seed=310):
    sag = lambda x: 30 + 50 * math.sin(math.pi * x / W)
    f.stroke([(x, sag(x)) for x in range(-20, W + 21, 20)], p=p, seed=seed, width=2.5)
    for k in range(12):
        x = 60 + k * 105
        fp = clamp((p * W - x) / 120)
        y = sag(x)
        shape(f, poly((x - 22, y), (x + 22, y), (x, y + 42), closed=True), CONFETTI[k % 5], p=fp, seed=seed + 1 + k)


def flower(f, x, y, r, col, p, t, seed):
    if p <= 0:
        return
    sway = 1.5 * math.sin(t * 2 + seed)
    for k in range(5):
        a = k * 2 * math.pi / 5 + seed
        f.fill(circle(x + sway + math.cos(a) * r * p, y + math.sin(a) * r * p, r * 0.7 * p + 0.5, span=6.3, seed=seed + k),
               col, alpha=0.95, seed=seed + k, offset=(0, 0))
    dot(f, x + sway, y, r * 0.45 * p + 0.5, color=(240, 180, 60), alpha=p, seed=seed + 6)


def meadow(f, t, bloom, seed=900, n=46):
    rnd = random.Random(seed)
    for i in range(n):
        x, y = rnd.uniform(20, 1260), rnd.uniform(580, 705)
        r = rnd.uniform(9, 13) * (0.8 + 0.4 * (y - 570) / 130)
        p = clamp(bloom * 2.2 - abs(x - 640) / 600)
        if p > 0:
            f.stroke(line((x, y + 3 * r * p), (x, y), over=0), p=p, seed=seed + i, color=(90, 150, 80), width=2.5, passes=1)
            flower(f, x, y, r, FLOWER[i % 5], p, t, seed + i * 10)


def rising_hearts(f, t, u, t0, x, y, n=3, spread=40, seed=90):
    for k in range(n):
        st = t0 + k * 0.5
        ph = ((u - st) / 1.8) % 1 if u > st else -1
        if ph >= 0:
            heart(f, x + (k - n // 2) * spread + 10 * math.sin(t * 3 + k), y - 120 * ph, 15 + 3 * k,
                  p=prog(ph, 0, 0.15), alpha=1 - ph, seed=seed + k)


def size_for(s, size, width):
    return int(min(size, width / max(1, len(s)) / 1.04))


def scene_caps(defaults):
    return O.get("scene_captions") or defaults


# ---------------------------------------------------------------- 場面

def scene_title(f, t):
    a = 1 - prog(t, TITLE_DUR - 0.6, 0.6)
    title, sub = O["title"], O.get("subtitle", "")
    f.text(title, 560, 300, size_for(title, 96, 860), p=prog(t, 0.2, 1.2), alpha=a, seed=1, center=True)
    f.stroke(poly(*[(300 + i * 10, 372 + 6 * math.sin(i * 0.55)) for i in range(53)]),
             p=prog(t, 1.3, 0.8), seed=2, color=O["accent"], width=5, alpha=int(210 * a))
    if sub:
        f.text(sub, 560, 450, size_for(sub, 36, 820), p=prog(t, 1.8, 0.9), alpha=a, seed=3, center=True)
    if a > 0.3:
        cat(f, 1040, 470, s=1.1, p=prog(t, 1.2, 1.2), eyes="happy", tail=t * 3, seed=610)
        for k in range(3):
            sparkle(f, 960 + k * 70, 300 - 30 * (k % 2), 10, prog(t, 2.0 + k * 0.2, 0.3) * a * (0.6 + 0.4 * math.sin(t * 6 + k)),
                    seed=620 + k)


def scene_birthday(f, t, u):
    floor(f, u, (200, 150, 100))
    garland(f, prog(u, 0.2, 1.4))
    # テーブル
    tp = prog(u, 0.4, 0.6)
    cloth = poly((430, 470), (850, 470), (870, 560), (410, 560), closed=True)
    shape(f, cloth, (245, 245, 250), p=tp, seed=20)
    f.stroke([(410 + i * 11.5, 560 + 6 * abs(math.sin(i * 0.8))) for i in range(41)], p=tp, seed=21,
             color=O["accent"], width=4)
    # ケーキ
    cx, base = 640, 470
    cp = prog(u, 0.8, 1.4)
    shape(f, poly((cx - 150, base), (cx + 150, base), (cx + 150, base - 90), (cx - 150, base - 90), closed=True),
          (250, 232, 208), p=stage(cp, 0, 3), seed=30)
    f.stroke([(cx - 150 + i * 10, base - 90 + (16 if i % 3 == 1 else 4)) for i in range(31)], p=stage(cp, 0, 3),
             seed=31, color=(240, 150, 170), width=5)
    for k in range(6):
        shape(f, circle(cx - 115 + k * 46, base - 40, 13, span=6.3, seed=32 + k), (230, 64, 74), p=stage(cp, 1, 3),
              seed=32 + k, width=2.5)
    shape(f, poly((cx - 95, base - 90), (cx + 95, base - 90), (cx + 95, base - 160), (cx - 95, base - 160), closed=True),
          (255, 246, 236), p=stage(cp, 1, 3), seed=40)
    f.stroke([(cx - 95 + i * 9.5, base - 160 + (14 if i % 3 == 1 else 3)) for i in range(21)], p=stage(cp, 1, 3),
             seed=41, color=(240, 150, 170), width=5)
    n = O["candles"]
    gap = min(34, 160 / max(1, n - 1))
    blow = 7.0
    lit = 1 - prog(u, blow, 0.15)
    flames = []
    for k in range(n):
        x = cx - gap * (n - 1) / 2 + k * gap
        shape(f, poly((x - 5, base - 160), (x + 5, base - 160), (x + 5, base - 215), (x - 5, base - 215), closed=True),
              CONFETTI[k % 5], p=stage(cp, 2, 3), seed=50 + k, width=2.5)
        if u > 2.2 and lit > 0:
            fl = 1 + 0.15 * math.sin(t * 20 + k * 2)
            pts = [(x + 7 * math.sin(a) * (1 - math.cos(a) * 0.3), base - 228 - 13 * fl * math.cos(a)) for a in
                   [i * math.pi / 15 for i in range(31)]]
            f.fill(pts, (255, 170, 60), alpha=lit * prog(u, 2.2, 0.3), seed=60 + k, offset=(0, 0))
            dot(f, x, base - 226, 3, color=(255, 240, 160), alpha=lit, seed=70 + k)
            flames.append(x)
        if blow < u < blow + 1.8:  # けむり
            ph = (u - blow) / 1.8
            f.stroke([(x + 6 * math.sin(i * 0.7 + t * 4), base - 222 - i * 4 * ph * 3) for i in range(12)], seed=80 + k,
                     color=(150, 150, 160), width=2, alpha=int(160 * (1 - ph)), passes=1)
    # 女の子とねこ
    happy = u > blow + 0.3
    blowing = blow - 0.8 < u < blow + 0.2
    girl(f, 300, 560, p=prog(u, 0.6, 1.2), eyes="happy" if happy else ("closed" if blowing else "open"),
         hand=(360, 380) if happy else None, seed=500)
    if blowing:
        for k in range(3):
            ph = (t * 3 + k / 3) % 1
            f.stroke(line((350 + 60 * ph, 360 + k * 12), (370 + 60 * ph, 360 + k * 12), over=0), seed=90 + k,
                     color=(120, 160, 210), width=3, alpha=int(200 * (1 - ph)), passes=1)
    cat(f, 1000, 560, p=prog(u, 1.0, 1.2), eyes="happy" if happy else "open", tail=t * 3, seed=620)
    confetti(f, t, T["main"] + blow + 0.3)
    caps = scene_caps(["ろうそく ふーっ", "おめでとう！"])
    caption(f, t, caps[0], T["main"] + 2.6, T["main"] + blow, 101, y=150)  # 旗の下に書く
    caption(f, t, caps[1], T["main"] + blow + 0.6, T["msg"] - 0.3, 102, y=150)
    f.tint((255, 246, 230))
    for x in flames:
        f.glow(x, base - 226, 18, (255, 220, 140), 0.5 * lit)


def scene_wedding(f, t, u):
    floor(f, u, (150, 190, 120), alpha=0.4)
    # 花のアーチ
    ap = prog(u, 0.3, 1.6)
    f.stroke(circle(640, 560, 330, a0=math.pi, span=math.pi, wob=0.01), p=ap, seed=20, color=(90, 150, 80), width=7)
    for k in range(17):
        a = math.pi + math.pi * (k + 0.5) / 17
        flower(f, 640 + math.cos(a) * 330, 560 + math.sin(a) * 330, 16, FLOWER[k % 5],
               clamp((ap * 17 - k) / 2) * prog(u, 1.0, 1.0), t, 30 + k * 10)
    # 2ひきのねこ（花嫁と花むこ）
    s = 1.35
    for k, (x, colors) in enumerate(((545, O["cat"]), (735, O["cat2"]))):
        set_cat(colors)
        cp = prog(u, 1.4 + k * 0.4, 1.2)
        cat(f, x, 560, s=s, p=cp, eyes="happy" if u > 5.5 else "open", tail=t * 2.5 + k, head=(10 - 20 * k, 0),
            seed=620 + k * 40)
        P = lambda dx, dy: (x + (10 - 20 * k + dx) * s, 560 + (-88 + dy) * s)  # 頭の中心からの位置
        if k == 0:  # ベール
            vp = prog(u, 3.0, 0.6)
            veil = [P(34 * math.cos(a), -22 + 26 * math.sin(a)) for a in [math.pi + i * math.pi / 12 for i in range(13)]]
            veil += [P(52, 70), P(20, 82), P(-20, 82), P(-52, 70), veil[0]]
            f.fill(veil, (255, 255, 255), alpha=0.5 * vp, seed=200, offset=(0, 0))
            f.stroke(veil, p=vp, seed=200, color=(190, 190, 210), width=2.5)
            for j in range(4):
                flower(f, *P(-27 + j * 18, -36 + 5 * abs(j - 1.5)), 8, FLOWER[j], vp, t, 210 + j * 10)
        else:  # ちょうネクタイ
            bp = prog(u, 3.4, 0.4)
            for d in (-1, 1):
                shape(f, poly(P(0, 34), P(d * 20, 24), P(d * 20, 46), closed=True), (60, 60, 80), p=bp, seed=230 + d,
                      width=2.5)
            dot(f, *P(0, 35), 6, color=(60, 60, 80), alpha=bp, seed=233)
    set_cat(O["cat"])
    rising_hearts(f, t, u, 5.5, 640, 380, n=3, spread=30)
    for k in range(6):
        sparkle(f, 380 + k * 105, 160 + 50 * (k % 2), 9, prog(u, 6 + k * 0.2, 0.3) * (0.5 + 0.5 * math.sin(t * 5 + k)),
                seed=240 + k)
    caps = scene_caps(["けっこん おめでとう", "すえながく おしあわせに"])
    caption(f, t, caps[0], T["main"] + 2.0, T["main"] + 6.4, 101)
    caption(f, t, caps[1], T["main"] + 6.8, T["msg"] - 0.3, 102)
    f.tint((255, 246, 236))


def scene_pet(f, t, u):
    # 空と丘
    hp = prog(u, 0, 1.0)
    for k, (cx, half, h, col) in enumerate(((300, 520, 120, (150, 200, 120)), (1000, 560, 90, (130, 185, 110)))):
        pts = hill(cx, 560, half, h)
        f.fill(pts + [(cx + half, 560), (cx - half, 560)], col, alpha=0.5 * hp, seed=12 + k, offset=(0, 4))
        f.stroke(pts, p=hp, seed=12 + k, width=3)
    floor(f, u, (140, 190, 110), alpha=0.5)
    sp = prog(u, 0.4, 0.8)
    shape(f, circle(1100, 130, 48, span=6.3, seed=20), (255, 210, 90), p=sp, seed=20)
    for k in range(10):
        a = k * math.pi / 5 + t * 0.4
        f.stroke(line((1100 + math.cos(a) * 66, 130 + math.sin(a) * 66), (1100 + math.cos(a) * 88, 130 + math.sin(a) * 88),
                      over=0), p=sp, seed=21 + k, color=(240, 180, 60), width=3, passes=1)
    meadow(f, t, prog(u, 1.0, 2.5))
    # 飼い主となでられるねこ
    owner = O.get("owner", True)
    cx = 640 if owner else 600
    pet = prog(u, 4.5, 0.4)
    stroke_ = 12 * math.sin(t * 7) * pet
    if owner:
        girl(f, 530, 585, pose="sit", p=prog(u, 0.8, 1.2), eyes="happy" if u > 5 else "open",
             hand=(lerp(590, cx - 10 + stroke_, pet), lerp(520, 460, pet)), seed=500)
    cat(f, cx, 585, s=1.05, p=prog(u, 1.2, 1.2), eyes="happy" if u > 5 else "open", tail=t * 2.5, seed=620)
    rising_hearts(f, t, u, 5.5, cx - 30, 420)
    # ちょうちょ
    bx, by = 300 + 120 * math.sin(t * 0.6), 300 + 40 * math.sin(t * 1.3)
    flap = abs(math.sin(t * 9))
    for d in (-1, 1):
        f.fill(circle(bx + d * 14 * flap, by, 14 * flap + 2, ry=16, span=6.3, seed=250 + d), (250, 200, 90),
               alpha=prog(u, 2.0, 0.5), seed=250 + d, offset=(0, 0))
    caps = scene_caps(["いつも そばに いてくれて", "ありがとう"])
    caption(f, t, caps[0], T["main"] + 2.0, T["main"] + 6.4, 101)
    caption(f, t, caps[1], T["main"] + 6.8, T["msg"] - 0.3, 102)
    f.tint((255, 248, 234))


def scene_shop(f, t, u):
    floor(f, u, (180, 180, 185))
    bp = lambda d: prog(u, 0.2 + d, 0.7)
    acc = O["accent"]
    shape(f, poly((300, 150), (980, 150), (980, 560), (300, 560), closed=True), (245, 232, 205), p=bp(0), seed=20)
    # 看板
    shape(f, poly((420, 168), (860, 168), (860, 238), (420, 238), closed=True), (255, 252, 244), p=bp(0.3), seed=21)
    name = O["title"]
    f.text(name, 640, 212, size_for(name, 44, 400), p=prog(u, 1.0, 0.8), seed=22, center=True)
    # ひさし
    ap = bp(0.5)
    for k in range(10):
        x = 290 + k * 70
        stripe = poly((x, 262), (x + 70, 262), (x + 70, 312), (x, 312), closed=True)
        f.fill(stripe, acc if k % 2 == 0 else (255, 255, 250), alpha=0.85 * ap, seed=30 + k, offset=(0, 0))
    edge = [(290 + i * 5, 312 + 12 * abs(math.sin(math.pi * i / 14))) for i in range(141)]
    f.fill(edge + [(990, 312), (290, 312)], acc, alpha=0.6 * ap, seed=41, offset=(0, 0))
    f.stroke(poly((290, 262), (990, 262), (990, 312)) + edge[::-1] + [(290, 262)], p=ap, seed=40, width=3)
    # 窓とドア
    for k, x0 in enumerate((340, 760)):
        win = poly((x0, 360), (x0 + 180, 360), (x0 + 180, 490), (x0, 490), closed=True)
        f.fill(win, (190, 215, 240), alpha=0.6 * bp(0.7), seed=50 + k, offset=(0, 0))
        f.stroke(win, p=bp(0.7), seed=50 + k, width=3.5)
        for j in range(3):
            flower(f, x0 + 40 + j * 50, 500, 11, FLOWER[(j + k) % 5], bp(1.0), t, 60 + k * 30 + j * 10)
    shape(f, poly((570, 370), (710, 370), (710, 560), (570, 560), closed=True), (170, 120, 80), p=bp(0.8), seed=55)
    dot(f, 690, 470, 5, color=(240, 200, 90), alpha=bp(0.8), seed=56)
    # OPEN の札（ゆれる）
    op = prog(u, 4.0, 0.3)
    if op > 0:
        a = 0.35 * math.sin((u - 4.0) * 6) * math.exp(-(u - 4.0) * 0.8)
        R = lambda x, y: (640 + rot(x, y, a)[0], 392 + rot(x, y, a)[1])
        f.stroke([R(-20, 26), (640, 392), R(20, 26)], p=op, seed=57, width=2)
        shape(f, [R(-44, 26), R(44, 26), R(44, 62), R(-44, 62), R(-44, 26)], (255, 252, 244), p=op, seed=58)
        f.text("OPEN", R(0, 52)[0], R(0, 52)[1], 24, p=op, seed=59, center=True, color=acc)
    # 女の子がやってくる
    wk = prog(u, 1.6, 2.6)
    gx = lerp(-80, 470, wk)
    walking = 0 < wk < 1
    girl(f, gx, 560 - (3 * abs(math.sin(gx * 0.05)) if walking else 0), walk=gx * 0.05 if walking else None,
         eyes="happy" if u > 5 else "open", hand=(gx + 50, 380) if u > 5 else None, seed=500)
    cat(f, 1090, 560, p=prog(u, 1.0, 1.2), eyes="happy" if u > 5 else "open", tail=t * 3, seed=620)
    for k in range(5):
        sparkle(f, 440 + k * 100, 140 + 20 * (k % 2), 9, prog(u, 4.6 + k * 0.15, 0.3) * (0.5 + 0.5 * math.sin(t * 5 + k)),
                seed=70 + k)
    confetti(f, t, T["main"] + 5.0, n=40)
    caps = scene_caps(["あたらしい おみせ", "オープン しました！"])
    caption(f, t, caps[0], T["main"] + 1.6, T["main"] + 4.6, 101, y=60)
    caption(f, t, caps[1], T["main"] + 5.0, T["msg"] - 0.3, 102, y=60)
    f.tint((255, 248, 236))


def scene_message(f, t, u):
    rows = O["rows"]
    has_photo = O["photo_img"] is not None
    x0 = 590 if has_photo else 100
    x1 = 1200 if has_photo else 1180
    top = 300 - 40 * (len(rows) - 1) + (40 if has_photo else 0)
    top = max(top, 160)
    for i in range(len(rows) + 1):  # 便せんの罫線
        y = top + 16 + i * 80
        f.stroke(line((x0 - 10, y), (x1, y + 2)), p=prog(u, 0.2 + i * 0.08, 0.8), seed=20 + i,
                 color=(150, 180, 220), width=2, alpha=150, passes=1)
    if has_photo:
        photo(f, prog(u, 0.3, 1.0))
    for i, r in enumerate(rows):
        st = 1.2 + i * ROW_DT
        cx = (x0 + x1) / 2 if not has_photo else x0
        f.text(r, cx, top + i * 80, 42, p=prog(u, st, 0.25 + 0.07 * len(r)), seed=30 + i, center=not has_photo)
    done = 1.2 + ROW_DT * len(rows)
    heart(f, x1 - 20, top + len(rows) * 80 + 40, 18, p=prog(u, done, 0.4), seed=60)
    f.tint((255, 248, 236))


def scene_end(f, t, u):
    frm = O["from"] + " より"
    f.text(frm, 640, 330, size_for(frm, 64, 1000), p=prog(u, 0.2, 1.0), seed=1, center=True)
    if O.get("date"):
        f.text(O["date"], 640, 410, 30, p=prog(u, 1.0, 0.5), seed=2, center=True)
    for k, x in enumerate((170, 300, 980, 1110)):  # 文字をよけて左右にハート
        st = 1.0 + k * 0.4
        ph = ((u - st) / 2.4) % 1 if u > st else -1
        if ph >= 0:
            heart(f, x + 12 * math.sin(t * 3 + k), 520 - 300 * ph, 16 + 3 * (k % 2), p=prog(ph, 0, 0.1),
                  alpha=1 - ph, seed=90 + k)
    cat(f, 1110, 620, s=0.7, p=prog(u, 0.6, 1.0), eyes="happy", tail=t * 3, seed=620)
    f.tint((255, 248, 236))


MAIN = {"birthday": scene_birthday, "wedding": scene_wedding, "pet": scene_pet, "shop": scene_shop}


def render(t, boil, watermark=False):
    f = Frame(t, boil)
    if t < T["main"]:
        scene_title(f, t)
    elif t < T["msg"]:
        MAIN[O["template"]](f, t, t - T["main"])
    elif t < T["end"]:
        scene_message(f, t, t - T["msg"])
    else:
        scene_end(f, t, t - T["end"])
    img = f.result()
    paper = PAPER.resize((W, H))
    for end in (T["main"], T["msg"], T["end"]):  # 場面の切りかえ: 紙にフェード
        k = prog(t, end - 0.5, 0.5) if t < end else 0
        if k > 0:
            img = Image.blend(img, paper, k)
    fo = prog(t, T["dur"] - 0.8, 0.8)
    if fo > 0:
        img = Image.blend(img, paper, fo)
    if watermark:
        mark = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mark).text((W // 2, H // 2), "SAMPLE", font=ImageFont.truetype(FONT_PATH, 200), fill=70,
                                  anchor="mm")
        img.paste((255, 255, 255), (0, 0), mark.rotate(18))
    return img


# ---------------------------------------------------------------- 書き出し

def _render_to(job):
    i, path, watermark = job
    render(i / FPS, i, watermark).save(path)


def preview(path, watermark):
    times = [2.6, T["main"] + 4.0, T["main"] + 10.0, T["end"] - 1.0, T["end"] + 3.6]
    sheet = Image.new("RGB", (1280, 360 * 3), (40, 40, 48))
    for k, s in enumerate(times):
        sheet.paste(render(s, int(s * FPS), watermark).resize((640, 360)), ((k % 2) * 640, (k // 2) * 360))
    sheet.save(path)


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if not exe:
            sys.exit("ffmpeg が見つかりません（pip install imageio-ffmpeg）")
        return exe


def write_video(out, watermark, jobs):
    ffmpeg = find_ffmpeg()
    n = int(T["dur"] * FPS)
    with tempfile.TemporaryDirectory() as tmp:
        work = [(i, os.path.join(tmp, f"{i:05d}.png"), watermark) for i in range(n)]
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            for k, _ in enumerate(pool.imap(_render_to, work, chunksize=4)):
                if k % (FPS * 5) == 0:
                    print(f"{k / FPS:5.1f}s / {T['dur']:.1f}s", flush=True)
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%05d.png")]
        if O.get("music"):
            cmd += ["-i", O["music"], "-af", f"afade=t=out:st={T['dur'] - 2:.2f}:d=2", "-c:a", "aac", "-b:a", "192k",
                    "-shortest"]
        cmd += ["-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-crf", "20", "-preset", "slow",
                "-movflags", "+faststart", out]
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="注文票から手描き風メッセージ動画をつくる")
    ap.add_argument("order", help="注文票（JSON）")
    ap.add_argument("--out", help="書き出し先（省略すると out/<注文票の名前>.mp4）")
    ap.add_argument("--preview", action="store_true", help="動画のかわりに確認用の静止画を書き出す")
    ap.add_argument("--sample", action="store_true", help="「SAMPLE」の透かしを入れる")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="同時に描くコマ数")
    args = ap.parse_args()
    try:
        order = load_order(args.order)
    except (OrderError, json.JSONDecodeError) as e:
        sys.exit(f"注文票のエラー: {e}")
    setup(order)
    stem = os.path.splitext(os.path.basename(args.order))[0] + ("_sample" if args.sample else "")
    out = args.out or os.path.join("out", stem + ("_preview.png" if args.preview else ".mp4"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if args.preview:
        preview(out, args.sample)
    else:
        write_video(out, args.sample, args.jobs)
    print("wrote", out, f"({T['dur']:.1f}s)")


if __name__ == "__main__":
    main()
