"""Tiny diagram engine: ONE layout definition -> draw.io XML page + PNG preview + layout checks.

Why: hand-drawn diagrams drift out of sync and accumulate overlaps. Here every box, arrow and label
is defined once in code; the same geometry is written to draw.io (diagrams.net) and rendered to PNG,
and automatic checks fail the build if text overflows a box, boxes overlap, an arrow passes through
a box, two arrows run on top of each other, or a label collides with something.
"""
from __future__ import annotations

import html
import math
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from PIL import ImageFont  # noqa: E402

INK, INK2, EDGE, GRID = "#0b0b0b", "#52514e", "#52514e", "#b8b7b1"
REG = font_manager.findfont("DejaVu Sans")
BOLD = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
_fonts: dict = {}

# kind -> (fill, stroke, shape)
KINDS = {
    "actor": ("#f1effc", "#6250d6", "round"),
    "env": ("#fff6e0", "#b07800", "round"),
    "code": ("#e8f1fc", "#2a78d6", "round"),
    "store": ("#e6f6ef", "#148f63", "cylinder"),
    "external": ("#fdeee6", "#c9541d", "round"),
    "output": ("#e6f6ef", "#148f63", "round"),
    "decision": ("#fff6e0", "#b07800", "rhombus"),
    "risk": ("#fde9ea", "#c93a39", "round"),
    "control": ("#e6f6ef", "#148f63", "round"),
    "where": ("#e8f1fc", "#2a78d6", "round"),
    "note": ("#f7f7f5", "#b8b7b1", "rect"),
    "frozen": ("#e8f1fc", "#2a78d6", "round"),
    "trainable": ("#fdeee6", "#c9541d", "round"),
    "newhead": ("#e6f6ef", "#148f63", "round"),
    "terminal": ("#f1f1ef", "#52514e", "pill"),
    "header": ("none", "none", "rect"),
    "time": ("#2a78d6", "#2a78d6", "round"),
}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(BOLD if bold else REG, size)
    return _fonts[key]


def text_w(s: str, size: int, bold: bool = False) -> float:
    return font(size, bold).getlength(s)


def wrap(text: str, width: float, size: int, bold: bool = False) -> list[str]:
    """Greedy word wrap that respects explicit newlines. Raises if a single word is too wide."""
    out = []
    for para in text.split("\n"):
        words, line = para.split(" "), ""
        for w in words:
            cand = (line + " " + w).strip()
            if text_w(cand, size, bold) <= width:
                line = cand
            else:
                if line:
                    out.append(line)
                if text_w(w, size, bold) > width:
                    raise ValueError(f"word too wide for box: '{w}' ({text_w(w, size, bold):.0f}px > {width:.0f}px)")
                line = w
        out.append(line)
    return out


@dataclass
class Box:
    id: str
    x: float
    y: float
    w: float
    h: float
    title: str
    sub: str = ""
    kind: str = "code"
    size: int = 13
    sub_size: int = 11
    align: str = "center"
    group: bool = False
    fill: str | None = None
    font_color: str | None = None

    @property
    def rect(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def side_point(self, side: str, f: float = 0.5):
        x0, y0, x1, y1 = self.rect
        return {"L": (x0, y0 + f * self.h), "R": (x1, y0 + f * self.h),
                "T": (x0 + f * self.w, y0), "B": (x0 + f * self.w, y1)}[side]


@dataclass
class Edge:
    id: str
    src: str | None
    dst: str | None
    pts: list
    label: str = ""
    dashed: bool = False
    arrow: bool = True
    label_above: bool = False
    exit: tuple | None = None
    entry: tuple | None = None
    color: str = EDGE
    width: float = 1.6
    size: int = 11


@dataclass
class Text:
    x: float
    y: float
    w: float
    h: float
    text: str
    size: int = 12
    bold: bool = False
    align: str = "left"
    color: str = INK


@dataclass
class Diagram:
    slug: str
    name: str
    W: int
    H: int
    boxes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    texts: list = field(default_factory=list)

    def __post_init__(self):
        self._n = 0

    # ---------------------------------------------------------------- building
    def title(self, title: str, subtitle: str = ""):
        self.texts.append(Text(30, 18, self.W - 60, 30, title, 20, True))
        if subtitle:
            self.texts.append(Text(30, 50, self.W - 60, 22, subtitle, 13, False, color=INK2))

    def box(self, id, x, y, w, h, title, sub="", kind="code", **kw) -> Box:
        b = Box(id, x, y, w, h, title, sub, kind, **kw)
        self.boxes.append(b)
        return b

    def group(self, id, x, y, w, h, label) -> Box:
        b = Box(id, x, y, w, h, label, "", "note", size=12, align="left", group=True)
        self.boxes.append(b)
        return b

    def text(self, x, y, w, h, text, size=12, bold=False, align="left", color=INK):
        self.texts.append(Text(x, y, w, h, text, size, bold, align, color))

    def get(self, id) -> Box:
        return next(b for b in self.boxes if b.id == id)

    def edge(self, src, dst, label="", exit=("R", 0.5), entry=("L", 0.5), via=(), dashed=False, arrow=True,
             color=EDGE):
        s, d = self.get(src), self.get(dst)
        p0, p1 = s.side_point(*exit), d.side_point(*entry)
        pts = [p0, *via, p1] if via else [p0, *self._auto(p0, p1, exit[0], entry[0]), p1]
        self._n += 1
        self.edges.append(Edge(f"e{self._n}", src, dst, pts, label, dashed, arrow, exit=exit, entry=entry,
                               color=color))

    def line(self, p0, p1, label="", dashed=False, arrow=True, label_above=False, via=(), color=EDGE, width=1.6):
        self._n += 1
        self.edges.append(Edge(f"e{self._n}", None, None, [p0, *via, p1], label, dashed, arrow, label_above,
                               color=color, width=width))

    @staticmethod
    def _auto(p0, p1, es, ns):
        (x0, y0), (x1, y1) = p0, p1
        if abs(x0 - x1) < 0.5 or abs(y0 - y1) < 0.5:
            return []
        hor_out, hor_in = es in "LR", ns in "LR"
        if hor_out and hor_in:
            mx = (x0 + x1) / 2
            return [(mx, y0), (mx, y1)]
        if not hor_out and not hor_in:
            my = (y0 + y1) / 2
            return [(x0, my), (x1, my)]
        return [(x1, y0)] if hor_out else [(x0, y1)]

    def legend(self, x, y, items, w=None, title="Legend"):
        """items: list of (kind | 'edge' | 'edge-dashed', label)."""
        row_h, pad = 30, 12
        w = w or 220
        h = pad * 2 + 22 + row_h * len(items)
        self.boxes.append(Box(f"legend_{x}_{y}", x, y, w, h, title, "", "note", size=12, align="left", group=True))
        for i, (kind, label) in enumerate(items):
            yy = y + pad + 24 + i * row_h
            if kind.startswith("edge"):
                self.line((x + pad, yy + 9), (x + pad + 40, yy + 9), dashed=kind.endswith("dashed"))
            else:
                self.box(f"lg_{x}_{y}_{i}", x + pad, yy, 40, 18, "", "", kind)
            self.text(x + pad + 52, yy - 1, w - pad * 2 - 52, 20, label, 11)

    # ------------------------------------------------------------------ geometry
    def _label_box(self, e: Edge):
        if not e.label:
            return None
        segs = list(zip(e.pts, e.pts[1:]))
        lengths = [math.dist(a, b) for a, b in segs]
        half, acc = sum(lengths) / 2, 0.0
        for (a, b), L in zip(segs, lengths):
            if acc + L >= half and L > 0:
                t = (half - acc) / L
                mx, my = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
                break
            acc += L
        lines = e.label.split("\n")
        w = max(text_w(s, e.size) for s in lines) + 8
        h = len(lines) * e.size * 1.25 + 4
        if e.label_above:
            return (mx - w / 2, my - h - 2, mx + w / 2, my - 2)
        return (mx - w / 2, my - h / 2, mx + w / 2, my + h / 2)

    # ------------------------------------------------------------------ checks
    def check(self) -> list[str]:
        problems = []
        solid = [b for b in self.boxes if not b.group and b.kind != "header"]
        # 1. text fits
        for b in self.boxes:
            if not (b.title or b.sub):
                continue
            pad_x = 10 if b.kind != "decision" else b.w * 0.22
            avail = b.w - 2 * pad_x
            try:
                lines_t = wrap(b.title, avail, b.size, bold=not b.group or True) if b.title else []
                lines_s = wrap(b.sub, avail, b.sub_size) if b.sub else []
            except ValueError as exc:
                problems.append(f"[{self.slug}] box '{b.id}': {exc}")
                continue
            need = len(lines_t) * b.size * 1.3 + len(lines_s) * b.sub_size * 1.3 + 12
            room = b.h * (0.62 if b.kind == "decision" else 1.0)
            if b.group:
                need = b.size * 1.3 + 6
                room = 26
            if need > room:
                problems.append(f"[{self.slug}] box '{b.id}': text needs {need:.0f}px height, box has {room:.0f}px")
        # 2. bounds + overlaps
        for b in self.boxes:
            x0, y0, x1, y1 = b.rect
            if x0 < 0 or y0 < 0 or x1 > self.W or y1 > self.H:
                problems.append(f"[{self.slug}] box '{b.id}' outside the page")
        for i, a in enumerate(solid):
            for c in solid[i + 1:]:
                if _overlap(_grow(a.rect, 8), c.rect):
                    problems.append(f"[{self.slug}] boxes '{a.id}' and '{c.id}' overlap or touch (<8px)")
        for g in [b for b in self.boxes if b.group]:
            band = (g.x, g.y, g.x + g.w, g.y + 26)
            for b in solid:
                if _overlap(band, b.rect):
                    problems.append(f"[{self.slug}] box '{b.id}' covers the label of group '{g.id}'")
        # 3. edges vs boxes
        for e in self.edges:
            for a, c in zip(e.pts, e.pts[1:]):
                if abs(a[0] - c[0]) > 0.5 and abs(a[1] - c[1]) > 0.5:
                    problems.append(f"[{self.slug}] edge {e.id} has a diagonal segment {a}->{c}")
                for b in solid:
                    if _seg_hits_rect(a, c, _grow(b.rect, -2)):
                        problems.append(f"[{self.slug}] edge {e.id} ({e.src}->{e.dst}) passes through box '{b.id}'")
        # 3b. edges must not run through legends or free text (except the legend's own sample lines)
        legends = [g for g in self.boxes if g.group and g.id.startswith("legend_")]
        def inside(e, r):
            return all(r[0] <= x <= r[2] and r[1] <= y <= r[3] for x, y in e.pts)
        for e in self.edges:
            for g in legends:
                if inside(e, g.rect):
                    continue
                for a, c in zip(e.pts, e.pts[1:]):
                    if _seg_hits_rect(a, c, g.rect):
                        problems.append(f"[{self.slug}] edge {e.id} runs through legend '{g.id}'")
            for t in self.texts:
                r = (t.x, t.y, t.x + t.w, t.y + t.h)
                if any(inside(e, g.rect) for g in legends):
                    continue
                for a, c in zip(e.pts, e.pts[1:]):
                    if _seg_hits_rect(a, c, r):
                        problems.append(f"[{self.slug}] edge {e.id} runs through text '{t.text[:25]}'")
        # 4. edges on top of each other
        segs = [(e.id, a, c) for e in self.edges for a, c in zip(e.pts, e.pts[1:])]
        for i, (ida, a1, a2) in enumerate(segs):
            for idb, b1, b2 in segs[i + 1:]:
                if ida != idb and _collinear_overlap(a1, a2, b1, b2):
                    problems.append(f"[{self.slug}] edges {ida} and {idb} overlap along a segment")
        # 5. labels
        lbls = [(e.id, self._label_box(e)) for e in self.edges if e.label]
        for eid, r in lbls:
            for b in solid:
                if _overlap(r, b.rect):
                    problems.append(f"[{self.slug}] label of {eid} overlaps box '{b.id}'")
        for i, (ea, ra) in enumerate(lbls):
            for eb, rb in lbls[i + 1:]:
                if _overlap(ra, rb):
                    problems.append(f"[{self.slug}] labels of {ea} and {eb} overlap")
        for t in self.texts:
            try:
                lines = wrap(t.text, t.w, t.size, t.bold)
            except ValueError as exc:
                problems.append(f"[{self.slug}] text: {exc}")
                continue
            if len(lines) * t.size * 1.3 > t.h + 2:
                problems.append(f"[{self.slug}] text '{t.text[:30]}' needs more height")
        return problems

    def crossings(self) -> int:
        segs = [(e.id, a, c) for e in self.edges for a, c in zip(e.pts, e.pts[1:])]
        n = 0
        for i, (ia, a1, a2) in enumerate(segs):
            for ib, b1, b2 in segs[i + 1:]:
                if ia != ib and _proper_cross(a1, a2, b1, b2):
                    n += 1
        return n

    # ------------------------------------------------------------------ draw.io
    def to_drawio(self, page_id: str) -> str:
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        n = 0

        def nid():
            nonlocal n
            n += 1
            return f"{page_id}-{n}"

        ids = {}
        for t in self.texts:
            style = (f"text;html=1;whiteSpace=wrap;align={t.align};verticalAlign=top;fontSize={t.size};"
                     f"fontStyle={1 if t.bold else 0};fontColor={t.color};fontFamily=Helvetica;")
            cells.append(_vertex(nid(), html.escape(t.text).replace("\n", "<br>"), style, t.x, t.y, t.w, t.h))
        for b in sorted(self.boxes, key=lambda b: not b.group):  # groups first (drawn underneath)
            ids[b.id] = nid()
            fill, stroke, shape = KINDS[b.kind]
            fill = b.fill or fill
            fc = b.font_color or INK
            common = (f"html=1;whiteSpace=wrap;fontFamily=Helvetica;fontSize={b.size};fontColor={fc};"
                      f"fillColor={fill};strokeColor={stroke};strokeWidth=1.5;align={b.align};"
                      f"spacingLeft=6;spacingRight=6;")
            if b.group:
                style = (f"rounded=1;arcSize=4;dashed=1;dashPattern=6 4;fillColor=none;strokeColor={GRID};"
                         f"verticalAlign=top;align=left;spacingLeft=10;spacingTop=2;fontSize={b.size};fontStyle=1;"
                         f"fontColor={INK2};html=1;whiteSpace=wrap;fontFamily=Helvetica;")
                value = html.escape(b.title)
            else:
                shape_style = {"round": "rounded=1;arcSize=10;", "rect": "rounded=0;", "pill": "rounded=1;arcSize=50;",
                               "rhombus": "rhombus;", "cylinder": "shape=cylinder3;boundedLbl=1;size=8;"}[shape]
                if b.kind == "header":
                    shape_style = "text;"
                style = shape_style + common + "verticalAlign=middle;"
                t = html.escape(b.title).replace("\n", "<br>")
                s = html.escape(b.sub).replace("\n", "<br>")
                value = (f"<b>{t}</b>" if t else "") + (f"<br><span style=\"font-size:{b.sub_size}px\">{s}</span>"
                                                        if s and t else s)
            cells.append(_vertex(ids[b.id], value, style, b.x, b.y, b.w, b.h))
        for e in self.edges:
            style = (f"edgeStyle=none;rounded=0;html=1;fontFamily=Helvetica;fontSize={e.size};fontColor={INK};"
                     f"strokeColor={e.color};strokeWidth={e.width};"
                     f"endArrow={'block' if e.arrow else 'none'};endFill=1;endSize=7;"
                     f"{'dashed=1;dashPattern=6 4;' if e.dashed else ''}")
            style += "labelBackgroundColor=none;verticalAlign=bottom;" if e.label_above else "labelBackgroundColor=#ffffff;"
            pts_xml = "".join(f'<mxPoint x="{x:g}" y="{y:g}"/>' for x, y in e.pts[1:-1])
            arr = f'<Array as="points">{pts_xml}</Array>' if pts_xml else ""
            label = html.escape(e.label).replace("\n", "<br>")
            if e.src:
                (ex, ey), (nx, ny) = _frac(self.get(e.src), e.exit), _frac(self.get(e.dst), e.entry)
                style += f"exitX={ex:g};exitY={ey:g};exitDx=0;exitDy=0;entryX={nx:g};entryY={ny:g};entryDx=0;entryDy=0;"
                geo = f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry>'
                cells.append(f'<mxCell id="{nid()}" value="{_attr(label)}" style="{style}" edge="1" parent="1" '
                             f'source="{ids[e.src]}" target="{ids[e.dst]}">{geo}</mxCell>')
            else:
                (x0, y0), (x1, y1) = e.pts[0], e.pts[-1]
                geo = (f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x0:g}" y="{y0:g}" as="sourcePoint"/>'
                       f'<mxPoint x="{x1:g}" y="{y1:g}" as="targetPoint"/>{arr}</mxGeometry>')
                cells.append(f'<mxCell id="{nid()}" value="{_attr(label)}" style="{style}" edge="1" parent="1">'
                             f'{geo}</mxCell>')
        model = (f'<mxGraphModel dx="{self.W}" dy="{self.H}" grid="1" gridSize="10" guides="1" tooltips="1" '
                 f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.W}" pageHeight="{self.H}" '
                 f'background="#ffffff" math="0" shadow="0"><root>{"".join(cells)}</root></mxGraphModel>')
        return f'<diagram id="{page_id}" name="{html.escape(self.name)}">{model}</diagram>'

    # ------------------------------------------------------------------ PNG
    def to_png(self, path: str, scale: float = 2.0):
        fig = plt.figure(figsize=(self.W / 100, self.H / 100), dpi=100 * scale)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.W)
        ax.set_ylim(self.H, 0)
        ax.axis("off")
        fig.patch.set_facecolor("white")
        pt = 100 / 72  # px per point at dpi 100

        def draw_text(x, y, w, h, lines_spec, align="center", valign="middle"):
            total = sum(sz * 1.3 for _, sz, _ in lines_spec)
            cy = y + (h - total) / 2 if valign == "middle" else y + 2
            for s, sz, bold in lines_spec:
                tx = {"center": x + w / 2, "left": x + 8, "right": x + w - 8}[align]
                ax.text(tx, cy + sz * 0.65, s, fontsize=sz / pt, fontweight="bold" if bold else "normal",
                        ha=align if align != "center" else "center", va="center", color=lines_spec_color[0],
                        family="DejaVu Sans")
                cy += sz * 1.3

        lines_spec_color = [INK]
        for b in sorted(self.boxes, key=lambda b: not b.group):
            fill, stroke, shape = KINDS[b.kind]
            fill = b.fill or fill
            if b.group:
                ax.add_patch(FancyBboxPatch((b.x, b.y), b.w, b.h, boxstyle="round,pad=0,rounding_size=8",
                                            fc="none", ec=GRID, lw=1.2, ls=(0, (6, 4))))
                ax.text(b.x + 10, b.y + 14, b.title, fontsize=b.size / pt, fontweight="bold", color=INK2,
                        va="center", family="DejaVu Sans")
                continue
            if b.kind == "header":
                pass
            elif shape == "rhombus":
                ax.add_patch(Polygon([(b.x + b.w / 2, b.y), (b.x + b.w, b.y + b.h / 2), (b.x + b.w / 2, b.y + b.h),
                                      (b.x, b.y + b.h / 2)], closed=True, fc=fill, ec=stroke, lw=1.5))
            elif shape == "cylinder":
                ax.add_patch(Rectangle((b.x, b.y + 4), b.w, b.h - 8, fc=fill, ec="none"))
                from matplotlib.patches import Arc, Ellipse
                ax.add_patch(Ellipse((b.x + b.w / 2, b.y + b.h - 4), b.w, 8, fc=fill, ec=stroke, lw=1.5))
                ax.add_patch(Rectangle((b.x + 0.75, b.y + 4), b.w - 1.5, b.h - 8, fc=fill, ec="none"))
                ax.plot([b.x, b.x], [b.y + 4, b.y + b.h - 4], color=stroke, lw=1.5)
                ax.plot([b.x + b.w, b.x + b.w], [b.y + 4, b.y + b.h - 4], color=stroke, lw=1.5)
                ax.add_patch(Ellipse((b.x + b.w / 2, b.y + 4), b.w, 8, fc=fill, ec=stroke, lw=1.5))
                ax.add_patch(Arc((b.x + b.w / 2, b.y + 4), b.w, 8, theta1=180, theta2=360, ec=stroke, lw=1.5))
            else:
                r = {"round": 8, "rect": 0.01, "pill": b.h / 2}[shape]
                ax.add_patch(FancyBboxPatch((b.x, b.y), b.w, b.h, boxstyle=f"round,pad=0,rounding_size={r}",
                                            fc=fill, ec=stroke, lw=1.5))
            pad_x = 10 if b.kind != "decision" else b.w * 0.22
            avail = b.w - 2 * pad_x
            spec = [(s, b.size, True) for s in (wrap(b.title, avail, b.size, True) if b.title else [])]
            spec += [(s, b.sub_size, False) for s in (wrap(b.sub, avail, b.sub_size) if b.sub else [])]
            lines_spec_color[0] = b.font_color or INK
            draw_text(b.x, b.y, b.w, b.h, spec, b.align)
        lines_spec_color[0] = INK
        for t in self.texts:
            lines_spec_color[0] = t.color
            draw_text(t.x - 8 if t.align == "left" else t.x, t.y, t.w + (16 if t.align == "left" else 0), t.h,
                      [(s, t.size, t.bold) for s in wrap(t.text, t.w, t.size, t.bold)], t.align, "top")
        lines_spec_color[0] = INK
        for e in self.edges:
            xs, ys = zip(*e.pts)
            ax.plot(xs, ys, color=e.color, lw=e.width * 0.75, ls=(0, (6, 4)) if e.dashed else "-",
                    solid_capstyle="butt")
            if e.arrow:
                (ax_, ay), (bx, by) = e.pts[-2], e.pts[-1]
                ang = math.atan2(by - ay, bx - ax_)
                L, Wd = 9, 4.5
                p1 = (bx - L * math.cos(ang) + Wd * math.sin(ang), by - L * math.sin(ang) - Wd * math.cos(ang))
                p2 = (bx - L * math.cos(ang) - Wd * math.sin(ang), by - L * math.sin(ang) + Wd * math.cos(ang))
                ax.add_patch(Polygon([(bx, by), p1, p2], closed=True, fc=e.color, ec=e.color, lw=0.5))
            r = self._label_box(e)
            if r:
                if not e.label_above:
                    ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fc="white", ec="none"))
                lines = e.label.split("\n")
                yy = r[1] + 2
                for s in lines:
                    ax.text((r[0] + r[2]) / 2, yy + e.size * 0.62, s, fontsize=e.size / pt, ha="center", va="center",
                            color=INK, family="DejaVu Sans")
                    yy += e.size * 1.25
        fig.savefig(path, dpi=100 * scale, facecolor="white")
        plt.close(fig)


# ---------------------------------------------------------------------- helpers
def _vertex(id, value, style, x, y, w, h):
    return (f'<mxCell id="{id}" value="{_attr(value)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" as="geometry"/></mxCell>')


def _attr(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _frac(b: Box, side):
    s, f = side
    return {"L": (0, f), "R": (1, f), "T": (f, 0), "B": (f, 1)}[s]


def _grow(r, d):
    return (r[0] - d, r[1] - d, r[2] + d, r[3] + d)


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _seg_hits_rect(p, q, r):
    x0, y0, x1, y1 = r
    if abs(p[1] - q[1]) < 0.5:  # horizontal
        y = p[1]
        lo, hi = sorted((p[0], q[0]))
        return y0 < y < y1 and lo < x1 and hi > x0
    if abs(p[0] - q[0]) < 0.5:  # vertical
        x = p[0]
        lo, hi = sorted((p[1], q[1]))
        return x0 < x < x1 and lo < y1 and hi > y0
    return False


def _collinear_overlap(a1, a2, b1, b2, tol=4):
    if abs(a1[1] - a2[1]) < 0.5 and abs(b1[1] - b2[1]) < 0.5 and abs(a1[1] - b1[1]) < tol:
        lo = max(min(a1[0], a2[0]), min(b1[0], b2[0]))
        hi = min(max(a1[0], a2[0]), max(b1[0], b2[0]))
        return hi - lo > tol
    if abs(a1[0] - a2[0]) < 0.5 and abs(b1[0] - b2[0]) < 0.5 and abs(a1[0] - b1[0]) < tol:
        lo = max(min(a1[1], a2[1]), min(b1[1], b2[1]))
        hi = min(max(a1[1], a2[1]), max(b1[1], b2[1]))
        return hi - lo > tol
    return False


def _proper_cross(a1, a2, b1, b2):
    ah, bh = abs(a1[1] - a2[1]) < 0.5, abs(b1[1] - b2[1]) < 0.5
    if ah == bh:
        return False
    h1, h2, v1, v2 = (a1, a2, b1, b2) if ah else (b1, b2, a1, a2)
    y, x = h1[1], v1[0]
    return (min(h1[0], h2[0]) + 1 < x < max(h1[0], h2[0]) - 1) and (min(v1[1], v2[1]) + 1 < y < max(v1[1], v2[1]) - 1)


def write_drawio(diagrams, path):
    pages = "".join(d.to_drawio(f"p{i + 1}") for i, d in enumerate(diagrams))
    xml = (f'<mxfile host="app.diagrams.net" agent="dlhandson build_diagrams.py" version="24.0.0">{pages}</mxfile>')
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
