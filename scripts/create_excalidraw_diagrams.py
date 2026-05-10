#!/usr/bin/env python3
"""Generate poster-oriented Excalidraw drafts for the final report."""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("docs/figures/excalidraw_inspiration")
UPDATED = 1_715_000_000_000


class SceneBuilder:
    def __init__(self) -> None:
        self.elements: list[dict] = []
        self._n = 0

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n:04d}"

    def _seed(self) -> int:
        return 1_000_000 + self._n * 7919

    def _base(
        self,
        element_type: str,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        stroke: str = "#1e1e1e",
        bg: str = "transparent",
        fill: str = "solid",
        stroke_width: int = 2,
        roughness: float = 1.0,
        opacity: int = 100,
        roundness: int | None = 3,
    ) -> dict:
        return {
            "id": self._id(element_type),
            "type": element_type,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "angle": 0,
            "strokeColor": stroke,
            "backgroundColor": bg,
            "fillStyle": fill,
            "strokeWidth": stroke_width,
            "strokeStyle": "solid",
            "roughness": roughness,
            "opacity": opacity,
            "groupIds": [],
            "frameId": None,
            "roundness": {"type": roundness} if roundness else None,
            "seed": self._seed(),
            "version": 1,
            "versionNonce": self._seed() + 17,
            "isDeleted": False,
            "boundElements": None,
            "updated": UPDATED,
            "link": None,
            "locked": False,
        }

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        stroke: str = "#1e1e1e",
        bg: str = "transparent",
        fill: str = "solid",
        stroke_width: int = 2,
        roughness: float = 1.0,
        opacity: int = 100,
        roundness: int | None = 3,
    ) -> str:
        el = self._base(
            "rectangle",
            x,
            y,
            w,
            h,
            stroke=stroke,
            bg=bg,
            fill=fill,
            stroke_width=stroke_width,
            roughness=roughness,
            opacity=opacity,
            roundness=roundness,
        )
        self.elements.append(el)
        return el["id"]

    def ellipse(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        stroke: str = "#1e1e1e",
        bg: str = "transparent",
        stroke_width: int = 2,
        opacity: int = 100,
    ) -> str:
        el = self._base(
            "ellipse",
            x,
            y,
            w,
            h,
            stroke=stroke,
            bg=bg,
            stroke_width=stroke_width,
            opacity=opacity,
            roundness=None,
        )
        self.elements.append(el)
        return el["id"]

    def diamond(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        stroke: str = "#1e1e1e",
        bg: str = "transparent",
        stroke_width: int = 2,
    ) -> str:
        el = self._base(
            "diamond",
            x,
            y,
            w,
            h,
            stroke=stroke,
            bg=bg,
            stroke_width=stroke_width,
        )
        self.elements.append(el)
        return el["id"]

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: int = 22,
        w: float = 240,
        h: float | None = None,
        color: str = "#1e1e1e",
        align: str = "left",
        valign: str = "top",
        bold: bool = False,
    ) -> str:
        lines = text.count("\n") + 1
        height = h if h is not None else size * 1.25 * lines
        el = self._base(
            "text",
            x,
            y,
            w,
            height,
            stroke=color,
            bg="transparent",
            stroke_width=1,
            roundness=None,
        )
        el.update(
            {
                "text": text,
                "fontSize": size,
                "fontFamily": 1,
                "textAlign": align,
                "verticalAlign": valign,
                "containerId": None,
                "originalText": text,
                "autoResize": False,
                "lineHeight": 1.25,
            }
        )
        self.elements.append(el)
        return el["id"]

    def arrow(
        self,
        pts: list[tuple[float, float]],
        *,
        color: str = "#1e1e1e",
        width: int = 3,
        end: str | None = "arrow",
        start: str | None = None,
        opacity: int = 100,
        roughness: float = 1.0,
    ) -> str:
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        el = self._base(
            "arrow",
            x0,
            y0,
            x1 - x0,
            y1 - y0,
            stroke=color,
            bg="transparent",
            stroke_width=width,
            roughness=roughness,
            opacity=opacity,
            roundness=2,
        )
        el.update(
            {
                "points": [[x - x0, y - y0] for x, y in pts],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": start,
                "endArrowhead": end,
            }
        )
        self.elements.append(el)
        return el["id"]

    def line(
        self,
        pts: list[tuple[float, float]],
        *,
        color: str = "#1e1e1e",
        width: int = 2,
        opacity: int = 100,
    ) -> str:
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        el = self._base(
            "line",
            x0,
            y0,
            x1 - x0,
            y1 - y0,
            stroke=color,
            bg="transparent",
            stroke_width=width,
            opacity=opacity,
            roundness=2,
        )
        el.update(
            {
                "points": [[x - x0, y - y0] for x, y in pts],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": None,
            }
        )
        self.elements.append(el)
        return el["id"]

    def grid(
        self,
        x: float,
        y: float,
        cols: int,
        rows: int,
        cell: float,
        *,
        pilot_cols: set[int] | None = None,
        pilot_rows: set[int] | None = None,
        bg: str = "#eef6ff",
    ) -> None:
        pilot_cols = pilot_cols or set()
        pilot_rows = pilot_rows or set()
        for r in range(rows):
            for c in range(cols):
                is_pilot = c in pilot_cols and r in pilot_rows
                shade = "#f8fbff"
                if (r + c) % 5 == 0:
                    shade = bg
                if is_pilot:
                    shade = "#1971c2"
                self.rect(
                    x + c * cell,
                    y + r * cell,
                    cell - 1,
                    cell - 1,
                    stroke="#b6c2cf",
                    bg=shade,
                    stroke_width=1,
                    roughness=0.4,
                    roundness=None,
                )

    def feature_stack(self, x: float, y: float, w: float, h: float, colors: list[str]) -> None:
        for i, color in enumerate(colors):
            self.rect(
                x + i * 16,
                y - i * 12,
                w,
                h,
                stroke="#495057",
                bg=color,
                stroke_width=2,
                opacity=85,
                roughness=0.7,
            )

    def save(self, filename: str) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        scene = {
            "type": "excalidraw",
            "version": 2,
            "source": "codex-generated-poster-draft",
            "elements": self.elements,
            "appState": {
                "theme": "light",
                "viewBackgroundColor": "#fffdf7",
                "gridSize": None,
                "exportBackground": True,
                "exportScale": 2,
                "currentItemFontFamily": 1,
                "currentItemStrokeWidth": 2,
                "currentItemRoughness": 1,
            },
            "files": {},
        }
        (OUT_DIR / filename).write_text(json.dumps(scene, indent=2), encoding="utf-8")


def build_problem_io() -> None:
    s = SceneBuilder()
    s.rect(40, 40, 1320, 760, stroke="#adb5bd", bg="#fffdf7", stroke_width=1, roughness=0.4)
    s.text(70, 70, "One OFDM slot becomes one decoding decision", size=34, w=760, bold=True)
    s.text(
        72,
        122,
        "The neural receiver maps a noisy 5G resource grid to soft bit estimates. "
        "BLER is measured directly on thresholded bit logits.",
        size=20,
        w=760,
        color="#495057",
    )

    # Transmitter side.
    s.rect(80, 230, 165, 118, stroke="#343a40", bg="#e7f5ff", stroke_width=2)
    s.text(102, 252, "coded bits", size=22, w=125, align="center")
    for i in range(24):
        x = 105 + (i % 8) * 14
        y = 300 + (i // 8) * 12
        s.text(x, y, "1" if i % 3 else "0", size=10, w=10, color="#1971c2")

    s.arrow([(245, 290), (320, 290)], color="#1971c2", width=3)
    s.rect(320, 208, 230, 172, stroke="#343a40", bg="#f8f9fa", stroke_width=2)
    s.text(344, 224, "16-QAM OFDM grid", size=21, w=185, align="center")
    s.grid(352, 270, 11, 7, 14, pilot_cols={0, 2, 4, 6, 8, 10}, pilot_rows={1, 5})
    s.text(352, 374, "128 subcarriers x 14 symbols", size=14, w=180, color="#495057", align="center")

    # Channel in the middle.
    s.arrow([(550, 290), (620, 290)], color="#495057", width=3)
    s.ellipse(615, 205, 235, 170, stroke="#c92a2a", bg="#fff5f5", stroke_width=3, opacity=90)
    s.text(650, 226, "wireless channel", size=23, w=165, align="center", bold=True)
    s.text(652, 268, "UMa / TDL-C\nnoise + fading\n4 Rx antennas", size=17, w=160, align="center", color="#862e2e")
    for offset, color in [(0, "#fa5252"), (18, "#f59f00"), (36, "#15aabf")]:
        s.line(
            [(642, 340 + offset / 3), (690, 326 + offset / 2), (735, 350 + offset / 4), (802, 329 + offset / 2)],
            color=color,
            width=2,
            opacity=70,
        )

    s.arrow([(850, 290), (930, 290)], color="#495057", width=3)

    # Receiver input tensor.
    s.rect(930, 190, 335, 210, stroke="#343a40", bg="#edf2ff", stroke_width=2)
    s.text(958, 210, "receiver input tensor", size=23, w=280, align="center", bold=True)
    s.feature_stack(980, 285, 86, 72, ["#d0ebff", "#a5d8ff", "#74c0fc"])
    s.feature_stack(1096, 285, 86, 72, ["#e5dbff", "#d0bfff", "#b197fc"])
    s.rect(1210, 282, 24, 78, stroke="#2b8a3e", bg="#d3f9d8", stroke_width=2)
    s.rect(1238, 282, 24, 78, stroke="#2b8a3e", bg="#b2f2bb", stroke_width=2)
    s.text(968, 365, "8 received\nchannels", size=14, w=105, align="center", color="#1864ab")
    s.text(1088, 365, "8 LS channel\nestimate channels", size=14, w=120, align="center", color="#5f3dc4")
    s.text(1200, 365, "2 pilot-distance\nmaps", size=14, w=86, align="center", color="#2b8a3e")

    # Decoder and metric below.
    s.arrow([(1098, 400), (1098, 475)], color="#343a40", width=3)
    s.rect(925, 475, 340, 95, stroke="#343a40", bg="#fff3bf", stroke_width=2)
    s.text(960, 496, "neural receiver", size=24, w=270, align="center", bold=True)
    s.text(966, 532, "outputs 7168 soft bit LLRs", size=17, w=260, align="center", color="#5f3dc4")
    s.arrow([(1098, 570), (1098, 645)], color="#343a40", width=3)
    s.rect(940, 645, 305, 88, stroke="#343a40", bg="#e6fcf5", stroke_width=2)
    s.text(966, 666, "block-error check", size=21, w=250, align="center", bold=True)
    s.text(985, 704, "any bit wrong means error", size=16, w=220, align="center", color="#087f5b")

    # Callout with exact dimensions.
    s.rect(84, 520, 535, 165, stroke="#868e96", bg="#ffffff", stroke_width=1, roughness=0.6)
    s.text(112, 545, "Dimensions used in the experiments", size=23, w=460, bold=True)
    s.text(
        114,
        590,
        "128 subcarriers\n14 OFDM symbols\n16-QAM gives 4 bits per resource element\n4 receive antennas",
        size=18,
        w=420,
        color="#343a40",
    )
    s.rect(650, 520, 210, 165, stroke="#1971c2", bg="#e7f5ff", stroke_width=2)
    s.text(676, 548, "primary metric", size=21, w=160, align="center", bold=True, color="#0b7285")
    s.text(680, 592, "BLER\nvs\naverage FLOPs", size=23, w=155, align="center", color="#0b7285")

    s.save("problem_io_slot_to_bler.excalidraw")


def build_architecture() -> None:
    s = SceneBuilder()
    s.rect(30, 35, 1810, 900, stroke="#adb5bd", bg="#fffdf7", stroke_width=1, roughness=0.4)
    s.text(65, 65, "Compute-aware MoE neural receiver", size=38, w=780, bold=True)
    s.text(
        68,
        122,
        "One shared stem is always evaluated. The router then selects exactly one expert for the slot.",
        size=21,
        w=1100,
        color="#495057",
    )

    # Input block.
    s.rect(70, 260, 240, 250, stroke="#343a40", bg="#f8f9fa", stroke_width=2)
    s.text(100, 284, "OFDM slot", size=25, w=180, align="center", bold=True)
    s.grid(112, 342, 12, 8, 13, pilot_cols={0, 2, 4, 6, 8, 10}, pilot_rows={1, 6})
    s.text(92, 462, "16 channels from data\n+ 2 pilot-distance maps", size=17, w=200, align="center", color="#495057")

    # Stem as visual stack.
    s.arrow([(310, 385), (390, 385)], color="#343a40", width=3)
    s.rect(390, 230, 270, 320, stroke="#495057", bg="#edf2ff", stroke_width=2)
    s.text(420, 255, "shared CNN stem", size=26, w=210, align="center", bold=True)
    s.feature_stack(435, 345, 110, 88, ["#d0ebff", "#a5d8ff", "#74c0fc"])
    s.text(418, 455, "18 -> 64 -> 64 -> 56", size=19, w=220, align="center", color="#1864ab")
    s.rect(438, 504, 170, 34, stroke="#1864ab", bg="#e7f5ff", stroke_width=1)
    s.text(454, 511, "always paid 285M", size=15, w=140, align="center", color="#1864ab")

    # Pooling neck.
    s.arrow([(660, 385), (735, 385)], color="#343a40", width=3)
    s.rect(735, 260, 185, 250, stroke="#495057", bg="#fff9db", stroke_width=2)
    s.text(760, 284, "pool features", size=24, w=140, align="center", bold=True)
    s.rect(780, 345, 100, 30, stroke="#f08c00", bg="#ffe8cc", stroke_width=2)
    s.rect(780, 392, 100, 30, stroke="#f08c00", bg="#ffd8a8", stroke_width=2)
    s.text(784, 350, "mean", size=16, w=90, align="center")
    s.text(788, 397, "max", size=16, w=86, align="center")
    s.text(760, 456, "112-D router input", size=17, w=140, align="center", color="#e67700")

    # Router switch.
    s.arrow([(920, 385), (1010, 385)], color="#343a40", width=3)
    s.diamond(1010, 270, 210, 230, stroke="#e67700", bg="#fff3bf", stroke_width=3)
    s.text(1058, 320, "router", size=27, w=110, align="center", bold=True)
    s.text(1040, 365, "MLP\n112 -> 64 -> 3", size=18, w=150, align="center", color="#7c4a00")
    s.ellipse(1080, 447, 70, 32, stroke="#7c4a00", bg="#ffe8cc", stroke_width=2)
    s.text(1086, 453, "top-1", size=15, w=58, align="center", color="#7c4a00")
    s.rect(1000, 542, 230, 54, stroke="#c92a2a", bg="#fff5f5", stroke_width=2)
    s.text(1018, 558, "no oracle SNR access", size=18, w=195, align="center", color="#c92a2a")

    # Expert lanes.
    lane_x = 1320
    lanes = [
        ("nano", 260, "#2b8a3e", "#d3f9d8", "4 blocks\n320M total"),
        ("small", 410, "#1971c2", "#d0ebff", "8 blocks\n695M total"),
        ("large", 595, "#c92a2a", "#ffe3e3", "8 blocks\n1604M total"),
    ]
    for name, y, color, bg, detail in lanes:
        s.rect(lane_x, y, 155, 84, stroke=color, bg=bg, stroke_width=3)
        s.text(lane_x + 30, y + 12, name, size=22, w=95, align="center", bold=True, color=color)
        s.text(lane_x + 26, y + 45, detail, size=14, w=105, align="center", color="#343a40")
    s.text(1280, 205, "heterogeneous experts", size=24, w=240, align="center", bold=True)

    # Colored routes.
    s.arrow([(1220, 384), (1265, 384), (1265, 302), (1320, 302)], color="#2b8a3e", width=4, opacity=85)
    s.arrow([(1220, 384), (1320, 452)], color="#1971c2", width=5, opacity=85)
    s.arrow([(1220, 384), (1265, 384), (1265, 637), (1320, 637)], color="#c92a2a", width=6, opacity=85)

    # Join and output.
    s.arrow([(1475, 302), (1530, 302), (1530, 480), (1605, 480)], color="#868e96", width=2, opacity=50)
    s.arrow([(1475, 452), (1605, 480)], color="#868e96", width=2, opacity=50)
    s.arrow([(1475, 637), (1530, 637), (1530, 480), (1605, 480)], color="#868e96", width=2, opacity=50)
    s.rect(1605, 410, 225, 140, stroke="#343a40", bg="#e6fcf5", stroke_width=2)
    s.text(1636, 438, "selected path", size=23, w=160, align="center", bold=True)
    s.text(1640, 482, "bit LLRs\n+ channel estimate", size=18, w=150, align="center", color="#087f5b")

    # Lower explanatory strip.
    s.rect(80, 725, 1710, 126, stroke="#868e96", bg="#ffffff", stroke_width=1)
    s.text(115, 746, "What the diagram should communicate on a poster", size=24, w=500, bold=True)
    s.text(
        116,
        790,
        "Dense receiver spends the large-network cost on every slot. "
        "This MoE keeps the shared stem fixed, then adapts the remaining compute "
        "per slot through hard top-1 routing.",
        size=18,
        w=1290,
        color="#343a40",
    )
    s.rect(1560, 755, 190, 65, stroke="#0b7285", bg="#e3fafc", stroke_width=2)
    s.text(1582, 768, "objective", size=17, w=145, align="center", bold=True, color="#0b7285")
    s.text(1578, 795, "BLER vs FLOPs", size=16, w=150, align="center", color="#0b7285")

    s.save("compute_aware_moe_architecture.excalidraw")


def build_mode_b() -> None:
    s = SceneBuilder()
    s.rect(40, 40, 1220, 680, stroke="#adb5bd", bg="#fffdf7", stroke_width=1, roughness=0.4)
    s.text(72, 72, "Post-hoc deployment view", size=36, w=650, bold=True)
    s.text(
        74,
        126,
        "The middle expert helps the optimization trajectory, but its expert head "
        "can be removed at inference for the selected operating point.",
        size=20,
        w=790,
        color="#495057",
    )

    # Before.
    s.text(112, 210, "trained MoE", size=26, w=230, bold=True)
    s.rect(85, 260, 280, 300, stroke="#343a40", bg="#f8f9fa", stroke_width=2)
    s.text(125, 290, "router", size=24, w=190, align="center", bold=True)
    for i, (name, y, color, bg) in enumerate(
        [
            ("nano", 360, "#2b8a3e", "#d3f9d8"),
            ("small", 425, "#1971c2", "#d0ebff"),
            ("large", 490, "#c92a2a", "#ffe3e3"),
        ]
    ):
        s.rect(175, y, 110, 42, stroke=color, bg=bg, stroke_width=2)
        s.text(190, y + 9, name, size=17, w=80, align="center", color=color)
        s.arrow([(150, 313), (150, y + 21), (175, y + 21)], color=color, width=2 + i)
    s.text(112, 588, "all three routes active", size=17, w=225, align="center", color="#495057")

    s.arrow([(385, 410), (500, 410)], color="#343a40", width=3)
    s.text(406, 370, "analysis", size=18, w=75, align="center", color="#495057")

    # Evidence card.
    s.rect(500, 230, 310, 355, stroke="#868e96", bg="#ffffff", stroke_width=2)
    s.text(528, 260, "per-expert success check", size=23, w=250, align="center", bold=True)
    s.rect(548, 330, 90, 105, stroke="#2b8a3e", bg="#d3f9d8", stroke_width=2)
    s.rect(646, 330, 90, 105, stroke="#1971c2", bg="#d0ebff", stroke_width=2)
    s.rect(744, 330, 90, 105, stroke="#c92a2a", bg="#ffe3e3", stroke_width=2)
    s.text(565, 348, "nano", size=16, w=58, align="center", color="#2b8a3e")
    s.text(660, 348, "small", size=16, w=62, align="center", color="#1971c2")
    s.text(762, 348, "large", size=16, w=58, align="center", color="#c92a2a")
    s.text(563, 386, "0%", size=27, w=60, align="center", bold=True)
    s.text(660, 386, "0%", size=27, w=60, align="center", bold=True)
    s.text(755, 382, "decodes\nsome", size=18, w=70, align="center", bold=True)
    s.text(
        536,
        470,
        "Only the largest expert decodes routed blocks successfully in the evaluated setting.",
        size=17,
        w=250,
        align="center",
        color="#343a40",
    )

    s.arrow([(830, 410), (945, 410)], color="#343a40", width=3)
    s.text(846, 370, "Mode B", size=18, w=75, align="center", color="#495057")

    # After deployment.
    s.text(960, 210, "deployed inference", size=26, w=255, bold=True)
    s.rect(925, 260, 280, 300, stroke="#343a40", bg="#f8f9fa", stroke_width=2)
    s.text(965, 290, "same router", size=24, w=190, align="center", bold=True)
    s.rect(1015, 360, 110, 42, stroke="#2b8a3e", bg="#d3f9d8", stroke_width=2)
    s.text(1030, 369, "nano", size=17, w=80, align="center", color="#2b8a3e")
    s.rect(1015, 425, 110, 42, stroke="#495057", bg="#f1f3f5", stroke_width=2)
    s.line([(1025, 456), (1115, 434)], color="#495057", width=3)
    s.text(1032, 433, "sink", size=17, w=76, align="center", color="#343a40")
    s.rect(1015, 490, 110, 42, stroke="#c92a2a", bg="#ffe3e3", stroke_width=2)
    s.text(1030, 499, "large", size=17, w=80, align="center", color="#c92a2a")
    s.arrow([(990, 313), (990, 381), (1015, 381)], color="#2b8a3e", width=2)
    s.arrow([(990, 313), (990, 446), (1015, 446)], color="#495057", width=4)
    s.arrow([(990, 313), (990, 511), (1015, 511)], color="#c92a2a", width=4)
    s.text(955, 588, "small expert compute removed", size=17, w=230, align="center", color="#495057")

    # Result badge.
    s.rect(390, 625, 520, 70, stroke="#0b7285", bg="#e3fafc", stroke_width=3)
    s.text(420, 643, "MoE + Mode B", size=24, w=185, align="center", bold=True, color="#0b7285")
    s.text(625, 641, "BLER 0.9021   at   47.3% dense-large FLOPs", size=22, w=260, align="center", color="#0b7285")

    s.save("mode_b_deployment_takeaway.excalidraw")


def build_poster_main() -> None:
    s = SceneBuilder()
    s.rect(35, 35, 1585, 905, stroke="#adb5bd", bg="#fffdf7", stroke_width=1, roughness=0.35)
    s.text(72, 68, "Adaptive compute for neural 5G reception", size=38, w=880)
    s.text(
        76,
        124,
        "Each slot carries different channel difficulty. The receiver keeps a shared front-end, "
        "then spends expert compute only where the router assigns it.",
        size=20,
        w=1040,
        color="#495057",
    )

    # Left signal panel.
    s.rect(75, 225, 345, 455, stroke="#495057", bg="#f8f9fa", stroke_width=2)
    s.text(110, 250, "one received OFDM slot", size=25, w=270, align="center")
    s.grid(128, 326, 13, 9, 15, pilot_cols={0, 2, 4, 6, 8, 10, 12}, pilot_rows={1, 7}, bg="#e7f5ff")
    s.text(132, 482, "128 x 14 resource grid", size=18, w=205, align="center", color="#1864ab")
    s.rect(118, 535, 250, 82, stroke="#868e96", bg="#ffffff", stroke_width=1)
    s.text(142, 552, "signal + LS estimate", size=20, w=200, align="center")
    s.text(130, 586, "16 input channels", size=17, w=225, align="center", color="#495057")
    s.rect(130, 636, 225, 26, stroke="#2b8a3e", bg="#d3f9d8", stroke_width=1)
    s.text(148, 641, "+ pilot-distance maps", size=15, w=190, align="center", color="#2b8a3e")

    # Shared front-end.
    s.arrow([(420, 452), (505, 452)], color="#343a40", width=3)
    s.rect(505, 250, 320, 405, stroke="#364fc7", bg="#edf2ff", stroke_width=3)
    s.text(552, 278, "shared front-end", size=26, w=225, align="center")
    for i, (x, y, w, h, label, color) in enumerate(
        [
            (560, 352, 160, 42, "18", "#d0ebff"),
            (548, 407, 184, 50, "64", "#a5d8ff"),
            (536, 470, 208, 58, "64", "#74c0fc"),
            (524, 542, 232, 66, "56", "#4dabf7"),
        ]
    ):
        s.rect(x, y, w, h, stroke="#1864ab", bg=color, stroke_width=2, opacity=90)
        s.text(x + w / 2 - 25, y + h / 2 - 12, label, size=20, w=50, align="center", color="#0b4778")
        if i < 3:
            s.arrow([(640, y + h + 4), (640, y + h + 22)], color="#1864ab", width=2)
    s.rect(548, 623, 185, 34, stroke="#1864ab", bg="#e7f5ff", stroke_width=1)
    s.text(568, 630, "285M FLOPs always", size=15, w=145, align="center", color="#1864ab")

    # Router as a compute switch.
    s.arrow([(825, 452), (910, 452)], color="#343a40", width=3)
    s.ellipse(910, 306, 250, 250, stroke="#e67700", bg="#fff3bf", stroke_width=3)
    s.text(960, 350, "router", size=31, w=150, align="center")
    s.text(946, 404, "mean + max pooled\nstem features", size=17, w=175, align="center", color="#7c4a00")
    s.rect(956, 486, 155, 36, stroke="#e67700", bg="#ffe8cc", stroke_width=2)
    s.text(988, 493, "top-1", size=18, w=90, align="center", color="#7c4a00")
    s.text(900, 586, "no true SNR is given", size=18, w=260, align="center", color="#c92a2a")

    # Expert bank with compute-scale bars.
    s.text(1220, 230, "expert bank", size=27, w=260, align="center")
    experts = [
        ("nano", 305, 145, "#2b8a3e", "#d3f9d8", "320M"),
        ("small", 425, 210, "#1971c2", "#d0ebff", "695M"),
        ("large", 565, 320, "#c92a2a", "#ffe3e3", "1604M"),
    ]
    for name, y, bar_w, stroke, bg, flops in experts:
        s.rect(1220, y, 340, 82, stroke=stroke, bg=bg, stroke_width=3)
        s.text(1240, y + 16, name, size=24, w=90, color=stroke)
        s.rect(1350, y + 25, bar_w, 28, stroke=stroke, bg="#ffffff", stroke_width=2)
        s.text(1360, y + 30, flops, size=15, w=90, color="#343a40")
    s.text(1230, 673, "only one expert runs for a slot", size=19, w=310, align="center", color="#495057")

    # Colored route fan-out.
    s.arrow([(1145, 432), (1188, 432), (1188, 346), (1220, 346)], color="#2b8a3e", width=4, opacity=85)
    s.arrow([(1145, 432), (1220, 466)], color="#1971c2", width=5, opacity=85)
    s.arrow([(1145, 432), (1188, 432), (1188, 606), (1220, 606)], color="#c92a2a", width=6, opacity=85)

    # Output rail.
    s.arrow([(1390, 690), (1390, 760), (915, 760)], color="#343a40", width=3)
    s.rect(720, 720, 195, 82, stroke="#087f5b", bg="#e6fcf5", stroke_width=2)
    s.text(748, 738, "soft bit LLRs", size=23, w=140, align="center", color="#087f5b")
    s.text(750, 775, "7168 values", size=16, w=140, align="center", color="#087f5b")
    s.arrow([(720, 760), (615, 760)], color="#343a40", width=3)
    s.rect(455, 720, 160, 82, stroke="#087f5b", bg="#e6fcf5", stroke_width=2)
    s.text(484, 738, "BLER", size=23, w=105, align="center", color="#087f5b")
    s.text(468, 775, "any bit wrong", size=16, w=132, align="center", color="#087f5b")

    # Bottom comparison strip.
    s.line([(95, 865), (1530, 865)], color="#ced4da", width=2, opacity=90)
    s.text(112, 827, "Dense receiver", size=19, w=170, color="#868e96")
    s.arrow([(300, 840), (650, 840)], color="#adb5bd", width=3, opacity=90)
    s.rect(660, 812, 190, 56, stroke="#868e96", bg="#f1f3f5", stroke_width=2)
    s.text(684, 826, "large every time", size=18, w=140, align="center", color="#868e96")
    s.arrow([(850, 840), (1040, 840)], color="#adb5bd", width=3, opacity=90)
    s.text(1060, 827, "100% dense-large FLOPs", size=19, w=260, color="#868e96")

    s.text(112, 890, "MoE deployment", size=19, w=170, color="#0b7285")
    s.arrow([(300, 904), (650, 904)], color="#0b7285", width=4, opacity=90)
    s.rect(660, 876, 190, 56, stroke="#0b7285", bg="#e3fafc", stroke_width=2)
    s.text(686, 890, "route per slot", size=18, w=135, align="center", color="#0b7285")
    s.arrow([(850, 904), (1040, 904)], color="#0b7285", width=4, opacity=90)
    s.text(1060, 890, "MoE + Mode B: 0.9021 BLER at 47.3% FLOPs", size=19, w=430, color="#0b7285")

    s.save("poster_main_polished.excalidraw")


def main() -> None:
    build_problem_io()
    build_architecture()
    build_mode_b()
    build_poster_main()


if __name__ == "__main__":
    main()
