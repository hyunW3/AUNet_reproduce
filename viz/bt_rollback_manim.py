"""Animated before_root:bt rollback trace, in Manim (manim.community).

Replays the REAL captured trace of prompt="hello" generating " world" through
apps/aunet/generate_bt.online_bt_loop with the llama3 BPE parser (1 rollback):
'hello's patch boundary is held speculative, the model samples '·wor', then
feeding 'l' commits 'hello|' (a change before the frontier) -> the '·wor' bytes
were off-policy, so the loop drops '·worl' and re-samples under the corrected mask.

Render (needs manim + ffmpeg; manim could NOT be installed on the training node,
so run this on your laptop or any box with manim):

    pip install manim                 # needs system cairo/pango/ffmpeg
    manim -qm -o bt_rollback.mp4 bt_rollback_manim.py BTRollback
    # -ql draft / -qh 1080p / --format=gif for a GIF
"""
from manim import (
    Scene, VGroup, Square, Text, Line, Rectangle,
    FadeIn, FadeOut, Create, Indicate, Transform, Wiggle,
    LEFT, RIGHT, UP, DOWN, ORIGIN, WHITE, config,
)

config.background_color = WHITE
GRAY, BLUE, RED, INK = "#9AA0A6", "#1A73E8", "#D93025", "#202124"
SIDE = 0.92
PROMPT = "hello"


class BTRollback(Scene):
    def construct(self):
        self.cells = []          # VGroup(square, label) per byte
        self.bars = {}           # boundary index -> Line
        self.rb = 0

        title = Text("before_root:bt    'hello'  +  ' world'", font="Monospace",
                     weight="BOLD", color=INK).scale(0.5).to_edge(UP).shift(LEFT * 1.2)
        self.rb_label = Text("rollbacks: 0", font="Monospace", color=RED).scale(0.4).to_edge(UP).to_edge(RIGHT)
        self.add(title, self.rb_label, self._legend())
        self.caption = Text("", font="Monospace", color=INK).scale(0.45).to_edge(DOWN)
        self.add(self.caption)

        self._say("prompt 'hello' fed — tail boundary held speculative (commit_margin=2)")
        for ch in PROMPT:
            self._add_cell(ch, GRAY, animate=False)
        self.play(FadeIn(VGroup(*self.cells)))
        self.wait(0.6)

        # --- first pass: sample '·wor' (no boundary committed yet) ---
        for ch in " wor":
            self._feed(ch, BLUE, f"feed '{self._g(ch)}'  ->  ACCEPT")

        # --- feed 'l' : 'hello|' commits, '·worl' becomes off-policy -> ROLLBACK ---
        self._add_cell("l", BLUE)
        self.play(FadeIn(self.cells[-1]), run_time=0.4)
        self._say("feed 'l': 'hello|' boundary commits (change before frontier)")
        bar = self._make_bar(4)                       # boundary after 'hello' (index 4)
        self.bars[4] = bar
        self.play(Create(bar))
        self.play(Indicate(bar, color=RED, scale_factor=1.2))
        drop = VGroup(*self.cells[5:10])              # '·worl'
        self.play(*[c.animate.set_color(RED) for c in drop])
        self.play(Wiggle(drop))
        self.rb += 1
        self._update_rb()
        self._say("ROLLBACK (rb=1): drop '·worl', restore parser, re-sample from 'hello|'")
        self.play(FadeOut(drop, shift=DOWN * 0.6))
        del self.cells[5:10]
        self.wait(0.4)

        # --- second pass: re-sample '·world' under the corrected mask ---
        for ch in " world":
            self._feed(ch, BLUE, f"feed '{self._g(ch)}'  ->  ACCEPT")

        # --- finalize: commit trailing token ---
        bar2 = self._make_bar(10)                     # boundary after 'world'
        self.bars[10] = bar2
        self.play(Create(bar2))
        self._say("finalize(): commit trailing token  ==  offline get_levels_mask  OK")
        self.wait(2.0)

    # ---- helpers -------------------------------------------------------- #
    def _g(self, ch):
        return "·" if ch == " " else ch

    def _add_cell(self, ch, color, animate=True):
        i = len(self.cells)
        sq = Square(side_length=SIDE, color=color, fill_opacity=0.12, stroke_width=3)
        lbl = Text(self._g(ch), font="Monospace", color=color, weight="BOLD").scale(0.6)
        cell = VGroup(sq, lbl)
        cell.move_to(LEFT * 5.3 + RIGHT * i * (SIDE + 0.04) + UP * 0.4)
        lbl.move_to(sq.get_center())
        self.cells.append(cell)
        return cell

    def _feed(self, ch, color, caption):
        cell = self._add_cell(ch, color)
        self._say(caption)
        self.play(FadeIn(cell, shift=DOWN * 0.3), run_time=0.4)

    def _make_bar(self, idx):
        # vertical bar on the right edge of cell `idx`
        sq = self.cells[idx][0]
        x = sq.get_right()[0]
        return Line(start=[x, sq.get_bottom()[1] - 0.12, 0],
                    end=[x, sq.get_top()[1] + 0.12, 0], color=RED, stroke_width=8)

    def _say(self, text):
        new = Text(text, font="Monospace", color=INK).scale(0.45).to_edge(DOWN)
        if "ROLLBACK" in text or "off-policy" in text or "commits" in text:
            new.set_color(RED)
        self.play(Transform(self.caption, new), run_time=0.3)

    def _update_rb(self):
        new = Text(f"rollbacks: {self.rb}", font="Monospace", color=RED).scale(0.4)
        new.to_edge(UP).to_edge(RIGHT)
        self.play(Transform(self.rb_label, new), run_time=0.2)

    def _legend(self):
        items = [("prompt", GRAY), ("generated", BLUE), ("rolled back", RED)]
        g = VGroup()
        for txt, col in items:
            sw = Rectangle(width=0.3, height=0.3, color=col, fill_opacity=0.12, stroke_width=3)
            lb = Text(txt, font="Monospace", color=col).scale(0.35)
            lb.next_to(sw, RIGHT, buff=0.1)
            g.add(VGroup(sw, lb))
        g.arrange(RIGHT, buff=0.6).shift(DOWN * 1.4)
        return g
