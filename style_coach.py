#!/usr/bin/env python3
"""
Stockfish FEN analyser + Style Coach (Defensive / Normal / Aggressive)

What’s new vs prior build:
- Stronger style separation AND real variety:
  • MultiPV de-dup (as before)
  • NEW: Per-move probe — if MultiPV lacks variety, we score a curated set of legal moves
    with Stockfish using searchmoves (shallow) so styles can actually diverge
  • Style cp-bonus + "within-margin diversity" selection
  • Safety caps + mate override still apply

Other features kept:
- Blue arrow = Style coach (your turn only), Orange = Best move preview
- Style-aware opening micro-book (~6 plies)
- Promotion dialog, stable POV eval, stale-result guards
- Fullscreen toggle: button or F11 (Esc exits)

Install:
    python -m pip install python-chess pillow cairosvg
    sudo apt install stockfish
"""

import io, os, math, shutil, threading, tkinter as tk
from typing import Optional, List, Tuple, Dict
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import chess, chess.engine, chess.svg, cairosvg

# ---------- Config ----------
DEPTH = 18          # main analysis depth (MultiPV)
MULTIPV = 12        # ask for more lines so the re-ranker has options

# Per-move probe (used when MultiPV doesn't give enough distinct first moves)
PER_MOVE_DEPTH = 9  # shallow, fast
MAX_PROBE_MOVES = 10  # how many legal moves to evaluate individually

BOARD_SIZE = 320
SQUARE = BOARD_SIZE // 8

# Win% curve: ~+300cp ≈ 75%
K_LOG = 0.004

# Style safety caps: how far we may deviate from engine #1 (caps on Δcp and Δwin%)
ADHERENCE = {
    "Aggressive": (140, 9.0),  # allow bolder divergence
    "Normal":     (30,  3.0),
    "Defensive":  (20,  2.0),
}

# Extra bias (in cp) per "style point" when re-ranking candidates (bigger = bolder)
STYLE_CP_BONUS = {"Aggressive": 60, "Defensive": 40}

# If an alternative move is within this many cp of engine #1,
# prefer the one that fits the style better.
DIVERSITY_MARGIN_CP = 60


def win_prob(cp_signed: int) -> float:
    return 100.0 / (1.0 + math.exp(-K_LOG * cp_signed))


def fmt_cp(cp: int) -> str:
    return f"{cp/100:.2f}" if cp < 0 else f"+{cp/100:.2f}"


def score_cp_signed(info_score: chess.engine.PovScore, pov_is_white: bool) -> int:
    pov = chess.WHITE if pov_is_white else chess.BLACK
    sc = info_score.pov(pov)
    if sc.is_mate():
        m = sc.mate()
        return 100000 if (m and m > 0) else -100000
    cp = sc.score(mate_score=100000)
    return int(cp if cp is not None else 0)


def mate_in(info_score: chess.engine.PovScore, pov_is_white: bool) -> Optional[int]:
    pov = chess.WHITE if pov_is_white else chess.BLACK
    sc = info_score.pov(pov)
    if sc.is_mate():
        m = sc.mate()
        return int(m) if (m and m > 0) else None
    return None


def find_stockfish_path() -> Optional[str]:
    envp = os.environ.get("STOCKFISH_PATH")
    if envp and os.path.isfile(envp) and os.access(envp, os.X_OK):
        return envp
    which = shutil.which("stockfish")
    if which:
        return which
    cand = "/usr/games/stockfish"
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stockfish FEN analyser + Style Coach")
        self.geometry("1000x760")
        self.resizable(False, False)

        # --- Fullscreen bindings ---
        self.is_fullscreen = False
        self.prev_geometry = None
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda e: self.exit_fullscreen())

        # FEN input
        ttk.Label(self, text="Paste a FEN string:").pack(anchor="w", padx=10, pady=(10, 2))
        self.fen_var = tk.StringVar()
        fen_ent = ttk.Entry(self, textvariable=self.fen_var, width=140)
        fen_ent.pack(padx=10, fill="x")
        fen_ent.bind("<Return>", lambda _ : self.load_fen())

        # Controls row
        ctl = ttk.Frame(self); ctl.pack(anchor="w", padx=10, pady=(6, 0))
        self.coach_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="Coach mode", variable=self.coach_on,
                        command=self.redraw).pack(side="left", padx=(0, 10))

        ttk.Label(ctl, text="Style:").pack(side="left")
        self.style_var = tk.StringVar(value="Normal")
        style_cb = ttk.Combobox(
            ctl, textvariable=self.style_var,
            values=["Off", "Defensive", "Normal", "Aggressive"],
            state="readonly", width=12
        )
        style_cb.pack(side="left", padx=(4, 12))
        style_cb.bind("<<ComboboxSelected>>", lambda _e: self.update_style_preview())

        # I'm playing as (sets stable POV + default orientation)
        play_fr = ttk.Frame(self); play_fr.pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Label(play_fr, text="I'm playing as:").pack(side="left")
        self.play_color = tk.StringVar(value="white")
        ttk.Radiobutton(play_fr, text="White", value="white",
                        variable=self.play_color, command=self.on_pick_color).pack(side="left")
        ttk.Radiobutton(play_fr, text="Black", value="black",
                        variable=self.play_color, command=self.on_pick_color).pack(side="left")

        # Orientation options
        opt_fr = ttk.Frame(self); opt_fr.pack(anchor="w", padx=10, pady=(0, 4))
        self.follow_turn = tk.BooleanVar(value=False)  # fixed by default
        ttk.Checkbutton(opt_fr, text="Flip with side to move",
                        variable=self.follow_turn, command=self.redraw).pack(side="left", padx=(0, 10))
        ttk.Button(opt_fr, text="Flip now", command=self.flip_once).pack(side="left")

        # Buttons
        btn_fr = ttk.Frame(self); btn_fr.pack(pady=8)
        ttk.Button(btn_fr, text="Load FEN", command=self.load_fen).grid(row=0, column=0, padx=5)
        ttk.Button(btn_fr, text="Reset",    command=self.reset_pos).grid(row=0, column=1, padx=5)
        ttk.Button(btn_fr, text="Fullscreen", command=self.toggle_fullscreen).grid(row=0, column=2, padx=5)
        self.best_btn = ttk.Button(btn_fr, text="Best move", command=self.preview_best)
        self.best_btn.grid(row=0, column=3, padx=5)
        ttk.Label(btn_fr, text="Your move (SAN):").grid(row=0, column=4, padx=(20, 4))
        self.move_var = tk.StringVar()
        ttk.Entry(btn_fr, width=14, textvariable=self.move_var).grid(row=0, column=5)
        ttk.Button(btn_fr, text="Evaluate", command=self.eval_san).grid(row=0, column=6, padx=5)

        # Eval label
        self.eval_lbl = ttk.Label(self, text="", font=("Courier New", 12))
        self.eval_lbl.pack(pady=6)

        # Board canvas
        self.canvas = tk.Canvas(self, width=BOARD_SIZE, height=BOARD_SIZE, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self._img = None
        self._sel = None

        # Status bar (also shows "why" text for the style suggestion)
        self.status = ttk.Label(self, text="", relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

        # Engine
        path = find_stockfish_path()
        if not path:
            messagebox.showerror("Engine not found",
                                 "Could not find Stockfish.\nTry:  sudo apt install stockfish\n"
                                 "Or set env var STOCKFISH_PATH.")
            self.destroy(); return
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            self.engine.configure({"Threads": max(1, os.cpu_count() // 2), "Hash": 256})
        except Exception as e:
            messagebox.showerror("Engine error", f"Failed to start Stockfish at:\n{path}\n\n{e}")
            self.destroy(); return

        # Internal board
        self.board = chess.Board()
        self.base_fen = self.board.fen()
        self.fixed_bottom_is_white = True  # honored when follow_turn is False

        # Analysis caches / guards
        self.last_infos: Optional[List[chess.engine.InfoDict]] = None
        self.permove_cache: Dict[str, chess.engine.PovScore] = {}  # uci -> score
        self.style_sugg_move: Optional[chess.Move] = None
        self.style_sugg_reason: str = ""
        self.last_eval_fen: Optional[str] = None
        self.eval_gen: int = 0  # increases with each async_eval call

        # Apply initial color selection (White)
        self.on_pick_color()

        self.redraw()
        self.async_eval()

    # ---------- Fullscreen ----------
    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.prev_geometry = self.geometry()
            try:
                self.attributes("-fullscreen", True)
            except Exception:
                self.state("zoomed")
        else:
            self.exit_fullscreen()

    def exit_fullscreen(self):
        try:
            self.attributes("-fullscreen", False)
        except Exception:
            pass
        if self.prev_geometry:
            self.geometry(self.prev_geometry)
        self.is_fullscreen = False

    # ---------- Helpers ----------
    def my_color_is_white(self) -> bool:
        return self.play_color.get() == "white"

    def _invalidate_after_board_change(self):
        """Clear cached analysis/suggestion when the position changes."""
        self.last_infos = None
        self.permove_cache = {}
        self.last_eval_fen = None
        self.style_sugg_move = None
        self.style_sugg_reason = ""
        self.status.config(text="Thinking…")

    # ---------- Orientation helpers ----------
    def bottom_color(self) -> chess.Color:
        return (self.board.turn if self.follow_turn.get()
                else (chess.WHITE if self.fixed_bottom_is_white else chess.BLACK))

    def flip_once(self):
        if self.follow_turn.get():
            self.follow_turn.set(False)
        self.fixed_bottom_is_white = not self.fixed_bottom_is_white
        self.redraw()

    # ---------- Color selection ----------
    def on_pick_color(self):
        self.follow_turn.set(False)
        self.fixed_bottom_is_white = (self.play_color.get() == "white")
        if self.last_eval_fen != self.board.fen():
            self.async_eval()
        else:
            self.refresh_style_from_cache()

    # ---------- Drawing ----------
    def redraw(self, board: Optional[chess.Board] = None, highlight: Optional[chess.Move] = None):
        if board is None:
            board = self.board

        arrows = []

        # Style coach arrow (blue) – only when it's *your* turn
        if self.coach_on.get() and self.style_var.get() != "Off" and self.style_sugg_move:
            my_turn = self.board.turn == (chess.WHITE if self.my_color_is_white() else chess.BLACK)
            if my_turn:
                arrows.append(chess.svg.Arrow(
                    self.style_sugg_move.from_square,
                    self.style_sugg_move.to_square,
                    color="#3b82f6"  # blue
                ))

        # Best-move preview arrow (orange), when preview is showing a temp board
        if highlight:
            arrows.append(chess.svg.Arrow(
                highlight.from_square, highlight.to_square, color="#f59e0b"  # orange
            ))

        svg = chess.svg.board(
            board=board,
            size=BOARD_SIZE,
            orientation=self.bottom_color(),
            arrows=arrows
        )
        png = cairosvg.svg2png(bytestring=svg.encode())
        self._img = ImageTk.PhotoImage(Image.open(io.BytesIO(png)))
        self.canvas.create_image(0, 0, image=self._img, anchor="nw")

    # ---------- Engine eval (MultiPV + per-move probe) ----------
    def async_eval(self):
        self.eval_gen += 1
        gen = self.eval_gen
        brd = self.board.copy()  # snapshot position for this call
        self.status.config(text="Thinking…")
        threading.Thread(target=self._worker_eval, args=(gen, brd), daemon=True).start()

    def _worker_eval(self, gen: int, brd: chess.Board):
        # 1) MultiPV batch
        info_list = self.engine.analyse(
            brd,
            chess.engine.Limit(depth=DEPTH),
            multipv=MULTIPV
        )

        # Distinct first moves from MultiPV
        unique_moves = []
        seen = set()
        for inf in info_list:
            pv = inf.get("pv")
            if not pv:
                continue
            mv = pv[0]
            if mv not in seen:
                seen.add(mv)
                unique_moves.append(mv)

        # 2) Per-move probe if variety is low OR legal moves are few
        permove_scores: Dict[str, chess.engine.PovScore] = {}
        try_probe = (len(unique_moves) < 2)
        legals = list(brd.legal_moves)

        if try_probe and legals:
            # Pick up to MAX_PROBE_MOVES legal moves using a quick, style-agnostic heuristic
            # (checks/captures/promo/castle/center/pawn-thrust)
            def quick_priority(b: chess.Board, mv: chess.Move) -> float:
                piece = b.piece_at(mv.from_square)
                to_piece = b.piece_at(mv.to_square)
                is_cap = (to_piece is not None) or b.is_en_passant(mv)
                is_promo = mv.promotion is not None
                gives_chk = b.gives_check(mv)
                castle = b.is_castling(mv)

                def center_w(sq: int) -> float:
                    f = chess.square_file(sq); r = chess.square_rank(sq)
                    return -((f-3.5)**2 + (r-3.5)**2)

                center_gain = 0.0
                if piece:
                    center_gain = center_w(mv.to_square) - center_w(mv.from_square)

                pawn_thrust = 0.0
                if piece and piece.piece_type == chess.PAWN and not is_cap:
                    delta = abs(chess.square_rank(mv.to_square) - chess.square_rank(mv.from_square))
                    pawn_thrust = float(delta)

                dev = 0.0
                if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                    if chess.square_rank(mv.from_square) in (0, 7):
                        dev = 1.0

                return (4.0 * gives_chk
                        + 3.0 * castle
                        + 3.0 * (is_cap or is_promo)
                        + 1.5 * max(0.0, center_gain)
                        + 1.2 * dev
                        + 1.0 * pawn_thrust)

            # Sort and keep top candidates not already in MultiPV set
            legals_sorted = sorted(legals, key=lambda m: quick_priority(brd, m), reverse=True)
            probe_moves: List[chess.Move] = []
            mv_set = set(unique_moves)
            for mv in legals_sorted:
                if mv not in mv_set:
                    probe_moves.append(mv)
                if len(probe_moves) >= MAX_PROBE_MOVES:
                    break

            # Evaluate each probe move with searchmoves at shallow depth
            for mv in probe_moves:
                info = self.engine.analyse(
                    brd,
                    chess.engine.Limit(depth=PER_MOVE_DEPTH),
                    searchmoves=[mv]
                )
                if "score" in info:
                    permove_scores[mv.uci()] = info["score"]

        # Return results to UI thread
        self.after(0, self._finish_eval, gen, brd.fen(), info_list, permove_scores)

    def _finish_eval(self, gen: int, fen: str, info_list: list, permove_scores: Dict[str, chess.engine.PovScore]):
        # Drop late/stale results
        if gen != self.eval_gen or fen != self.board.fen():
            return

        self.last_infos = info_list
        self.permove_cache = permove_scores
        self.last_eval_fen = fen

        # Eval label uses *your* POV (stable)
        pov_is_white = self.my_color_is_white()
        top = info_list[0] if info_list else None
        cp = score_cp_signed(top["score"], pov_is_white) if top else 0
        self.show_eval(cp)

        # Update the style-based suggestion
        self.apply_style_pick()
        self.redraw()

    def show_eval(self, cp_signed: int):
        my_prob = win_prob(cp_signed)
        opp_prob = 100.0 - my_prob
        self.eval_lbl.config(
            text=f"Eval: {fmt_cp(cp_signed)}  |  Win% Me:{my_prob:.0f} – Opp:{opp_prob:.0f}  (d{DEPTH})"
        )

    # ---------- UI actions ----------
    def load_fen(self):
        fen = self.fen_var.get().strip()
        if not fen:
            messagebox.showinfo("No FEN", "Paste a FEN first."); return
        try:
            self.board = chess.Board(fen)
        except ValueError as e:
            messagebox.showerror("Bad FEN", str(e)); return
        self.base_fen = self.board.fen()
        self._sel = None
        self._invalidate_after_board_change()
        self.redraw()
        self.async_eval()

    def reset_pos(self):
        self.board = chess.Board(self.base_fen)
        self._sel = None
        self._invalidate_after_board_change()
        self.redraw()
        self.async_eval()

    # Click-to-move (with promotion support)
    def _make_move_with_optional_promotion(self, from_sq: chess.Square, to_sq: chess.Square) -> chess.Move:
        piece = self.board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if to_rank in (0, 7):  # promotion
                ans = simpledialog.askstring(
                    "Promotion", "Promote to (q, r, b, n)?", parent=self
                )
                promo_map = {
                    "q": chess.QUEEN, "r": chess.ROOK,
                    "b": chess.BISHOP, "n": chess.KNIGHT
                }
                promo = promo_map.get((ans or "q").lower()[0], chess.QUEEN)
                return chess.Move(from_sq, to_sq, promotion=promo)
        return chess.Move(from_sq, to_sq)

    def on_click(self, event):
        bottom = self.bottom_color()
        fx = int(event.x // SQUARE)
        fy = int(event.y // SQUARE)
        if not (0 <= fx < 8 and 0 <= fy < 8):
            return

        if bottom == chess.WHITE:
            file, rank = fx, 7 - fy
        else:
            file, rank = 7 - fx, fy

        sq = chess.square(file, rank)

        if self._sel is None:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.board.turn:
                self._sel = sq
        else:
            mv = self._make_move_with_optional_promotion(self._sel, sq)
            self._sel = None
            if self.board.is_legal(mv):
                self.board.push(mv)
                self._invalidate_after_board_change()
                self.redraw()
                self.async_eval()
            else:
                messagebox.showwarning("Illegal", "That move isn't legal.")

    # Best move (non-destructive preview)
    def preview_best(self):
        self.status.config(text="Thinking…")
        threading.Thread(target=self._worker_best, daemon=True).start()

    def _worker_best(self):
        info = self.engine.analyse(self.board, chess.engine.Limit(depth=DEPTH))
        best = info["pv"][0]
        brd = self.board.copy()
        brd.push(best)
        self.after(0, self._finish_best, brd, best, info)

    def _finish_best(self, brd, best, info):
        self.redraw(board=brd, highlight=best)
        pov_is_white = self.my_color_is_white()
        cp = score_cp_signed(info["score"], pov_is_white)
        my_prob = win_prob(cp); opp_prob = 100 - my_prob
        self.eval_lbl.config(
            text=f"Best: {self.board.san(best):6} | {fmt_cp(cp)} | Win% Me:{my_prob:.0f} – Opp:{opp_prob:.0f}  (d{DEPTH})"
        )
        self.status.config(text="Preview – actual board unchanged (Reset to clear)")

    # SAN entry
    def eval_san(self):
        san = self.move_var.get().strip()
        if not san:
            messagebox.showinfo("No move", "Enter a SAN move first."); return
        try:
            mv = self.board.parse_san(san)  # handles e8=Q etc.
        except ValueError as e:
            messagebox.showerror("Bad move", str(e)); return
        self.board.push(mv)
        self._invalidate_after_board_change()
        self.redraw()
        self.async_eval()

    # ---------- Opening micro-book (style-aware, ~first 6 plies) ----------
    def _book_candidates(self, style: str) -> List[str]:
        """
        Return prioritized SAN moves for *side to move* based on move history and style.
        Book is used only early (up to ~6–8 plies).
        """
        if self.board.fullmove_number > 4:
            return []

        to_move_is_white = self.board.turn == chess.WHITE
        hist = [m.uci() for m in self.board.move_stack]

        # Start position: White to move
        if len(hist) == 0 and to_move_is_white:
            if style == "Aggressive":
                return ["e4", "d4", "c4", "Nf3"]
            if style == "Defensive":
                return ["d4", "Nf3", "c4", "e4"]
            return ["e4", "d4", "c4", "Nf3"]

        # Black's first vs White's first
        if len(hist) == 1 and not to_move_is_white:
            w1 = hist[0]
            if w1 == "e2e4":
                if style == "Aggressive":   return ["c5", "d5", "Nf6"]
                if style == "Defensive":    return ["e5", "c6", "e6"]
                return ["e5", "c5", "c6"]
            if w1 == "d2d4":
                if style == "Aggressive":   return ["Nf6", "c5", "f5"]
                if style == "Defensive":    return ["d5", "Nf6", "e6"]
                return ["Nf6", "d5"]
            if w1 == "c2c4":
                if style == "Aggressive":   return ["e5", "Nf6", "c5"]
                if style == "Defensive":    return ["e6", "Nf6", "c6"]
                return ["Nf6", "e5"]

        # A few key continuations (White to move)
        if len(hist) == 2 and to_move_is_white and hist[0] == "e2e4" and hist[1] == "e7e5":
            if style == "Aggressive":   return ["Nf3", "Bc4", "d4", "f4"]
            if style == "Defensive":    return ["Nf3", "Nc3", "d3"]
            return ["Nf3", "Nc3", "Bc4"]

        if len(hist) == 2 and to_move_is_white and hist[0] == "e2e4" and hist[1] == "c7c5":
            if style == "Aggressive":   return ["Nf3", "d4"]
            if style == "Defensive":    return ["Nf3", "c3"]
            return ["Nf3", "d4"]

        if len(hist) == 2 and to_move_is_white and hist[0] == "e2e4" and hist[1] == "c7c6":
            return ["d4", "Nc3", "Nd2"]

        if len(hist) == 2 and to_move_is_white and hist[0] == "e2e4" and hist[1] == "e7e6":
            if style == "Aggressive":   return ["d4", "Nc3"]
            if style == "Defensive":    return ["d4", "e5"]
            return ["d4", "Nc3", "Nd2"]

        if len(hist) == 2 and to_move_is_white and hist[0] == "d2d4" and hist[1] == "d7d5":
            if style == "Aggressive":   return ["c4", "Nf3"]
            if style == "Defensive":    return ["Nf3", "e3", "c3"]
            return ["c4", "Nf3"]

        if len(hist) == 2 and to_move_is_white and hist[0] == "d2d4" and hist[1] == "g8f6":
            if style == "Aggressive":   return ["c4", "Nc3"]
            if style == "Defensive":    return ["Nf3", "c4", "e3"]
            return ["c4", "Nf3"]

        # Italian shell follow-up
        if len(hist) >= 3 and to_move_is_white and hist[0] == "e2e4" and hist[1] == "e7e5" and hist[2] == "g1f3":
            if style == "Aggressive":   return ["Bc4", "d4"]
            if style == "Defensive":    return ["Bc4", "d3"]
            return ["Bc4", "Nc3"]

        # QGD shell: 1.d4 d5 2.c4 (…e6/…c6) → 3.Nc3/Nf3
        if len(hist) >= 4 and to_move_is_white and hist[:3] == ["d2d4","d7d5","c2c4"] and hist[3] in ("e7e6","c7c6"):
            if style == "Aggressive":   return ["Nc3", "Nf3"]
            if style == "Defensive":    return ["Nf3", "e3"]
            return ["Nc3", "Nf3"]

        return []

    def _try_san_list(self, sans: List[str]) -> Optional[chess.Move]:
        for san in sans:
            try:
                mv = self.board.parse_san(san)
            except Exception:
                continue
            if mv in self.board.legal_moves:
                return mv
        return None

    # ---------- Style coach: public triggers ----------
    def update_style_preview(self):
        """Refresh style suggestion instantly when the style dropdown changes."""
        if self.last_eval_fen != self.board.fen():
            self.async_eval()   # cache is for a different position
        else:
            self.refresh_style_from_cache()

    def refresh_style_from_cache(self):
        """Re-rank cached analysis with new style/color without re-analyzing."""
        self.apply_style_pick()
        self.redraw()

    def apply_style_pick(self):
        """Compute style pick + reason, update status."""
        if not self.coach_on.get() or self.style_var.get() == "Off":
            self.style_sugg_move = None
            self.style_sugg_reason = ""
            self.status.config(text="Ready")
            return

        # 1) Try book first (style-aware). If found, use it.
        book_sans = self._book_candidates(self.style_var.get())
        if book_sans:
            mv = self._try_san_list(book_sans)
            if mv:
                self.style_sugg_move = mv
                self.style_sugg_reason = f"book: {', '.join(book_sans)}"
                my_turn = self.board.turn == (chess.WHITE if self.my_color_is_white() else chess.BLACK)
                if my_turn:
                    try: san = self.board.san(mv)
                    except Exception: san = str(mv)
                    self.status.config(text=f"Style {self.style_var.get()}: {san}  —  {self.style_sugg_reason}")
                else:
                    self.status.config(text="Ready")
                return  # stay in book

        # 2) Otherwise, pick from MultiPV + per-move probe with style re-ranking.
        mv, reason = self.pick_styled_move_with_reason()
        self.style_sugg_move = mv
        self.style_sugg_reason = reason
        my_turn = self.board.turn == (chess.WHITE if self.my_color_is_white() else chess.BLACK)
        if mv and my_turn:
            try: san = self.board.san(mv)
            except Exception: san = str(mv)
            self.status.config(text=f"Style {self.style_var.get()}: {san}  —  {reason}")
        else:
            self.status.config(text="Ready")

    # ---------- Style coach: engine re-ranker ----------
    def _style_features(self, board: chess.Board, move: chess.Move) -> dict:
        """Cheap, explainable features for style ranking."""
        b = board
        mv = move
        moved_piece = b.piece_at(mv.from_square)
        to_piece = b.piece_at(mv.to_square)
        is_capture = to_piece is not None or b.is_en_passant(mv)
        is_promo = mv.promotion is not None
        gives_check = b.gives_check(mv)

        # Center heuristic
        def center_weight(sq: int) -> float:
            file = chess.square_file(sq); rank = chess.square_rank(sq)
            return -((file-3.5)**2 + (rank-3.5)**2)

        center_gain = 0.0
        if moved_piece:
            center_gain = center_weight(mv.to_square) - center_weight(mv.from_square)

        dev_new_piece = 0
        if moved_piece and moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            if chess.square_rank(mv.from_square) in (0, 7):
                dev_new_piece = 1

        pawn_thrust = 0
        if moved_piece and moved_piece.piece_type == chess.PAWN and not is_capture:
            delta = abs(chess.square_rank(mv.to_square) - chess.square_rank(mv.from_square))
            pawn_thrust = delta  # 1 or 2

        castle = b.is_castling(mv)
        quiet_dev = int((not is_capture) and moved_piece and moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP))
        king_shelter_pawn = 0
        if moved_piece and moved_piece.piece_type == chess.PAWN and not is_capture:
            file = chess.square_file(mv.from_square)
            if file in (0, 7):  # a/h pawns
                king_shelter_pawn = 1

        simplify_capture = 0
        if is_capture and moved_piece and to_piece:
            simplify_capture = int(moved_piece.piece_type == to_piece.piece_type)

        return {
            "is_check": int(gives_check),
            "is_capture_or_promo": int(is_capture or is_promo),
            "center_gain": center_gain,
            "dev_new_piece": dev_new_piece,
            "pawn_thrust": pawn_thrust,
            "castle": int(castle),
            "quiet_dev": quiet_dev,
            "king_shelter_pawn": king_shelter_pawn,
            "simplify_capture": simplify_capture,
        }

    def _reason_bits(self, style: str, f: dict) -> List[str]:
        """Human-readable reasons for the chosen style features (ordered by importance)."""
        bits: List[str] = []
        if style == "Aggressive":
            if f["is_check"]: bits.append("check")
            if f["is_capture_or_promo"]: bits.append("capture/promotion")
            if f["center_gain"] > 0.0: bits.append("center gain")
            if f["dev_new_piece"]: bits.append("develops new piece")
            if f["pawn_thrust"] >= 2: bits.append("two-square pawn thrust")
            elif f["pawn_thrust"] == 1: bits.append("pawn thrust")
        elif style == "Defensive":
            if f["castle"]: bits.append("castling")
            if f["quiet_dev"]: bits.append("quiet development")
            if f["king_shelter_pawn"]: bits.append("king-shelter pawn move")
            if f["simplify_capture"]: bits.append("simplifies by trade")
            if f["is_check"] == 0 and f["is_capture_or_promo"] == 0: bits.append("avoids forcing moves")
        else:  # Normal
            bits.append("engine #1")
        return bits

    def _style_metric(self, style: str, f: dict) -> float:
        """Numeric style score (higher = more on-style)."""
        if style == "Aggressive":
            return (
                3.0 * f["is_check"]
              + 2.0 * f["is_capture_or_promo"]
              + 1.2 * max(0.0, f["center_gain"])
              + 1.0 * f["dev_new_piece"]
              + (2.0 if f["pawn_thrust"] >= 2 else (0.8 if f["pawn_thrust"] == 1 else 0.0))
            )
        # Defensive
        return (
            2.5 * f["castle"]
          + 2.0 * f["quiet_dev"]
          + 1.5 * f["king_shelter_pawn"]
          + 1.0 * f["simplify_capture"]
          + (0.8 if (f["is_check"] == 0 and f["is_capture_or_promo"] == 0) else 0.0)
        )

    def pick_styled_move_with_reason(self) -> Tuple[Optional[chess.Move], str]:
        """Choose among candidates (MultiPV + per-move probe) using style; return move & reason."""
        style = self.style_var.get()
        pov_is_white = self.my_color_is_white()

        # Build candidate list from MultiPV (unique first moves)
        cands: List[Tuple[chess.Move, float, dict, List[str], Optional[int], float, str]] = []
        seen = set()
        if self.last_infos:
            for inf in self.last_infos:
                pv = inf.get("pv")
                if not pv:
                    continue
                mv = pv[0]
                if mv in seen:
                    continue
                seen.add(mv)
                cp = score_cp_signed(inf["score"], pov_is_white)
                feats = self._style_features(self.board, mv)
                reasons = self._reason_bits(style, feats)
                m_in = mate_in(inf["score"], pov_is_white)
                metric = self._style_metric(style, feats) if style != "Normal" else 0.0
                cands.append((mv, float(cp), feats, reasons, m_in, metric, "pv"))

        # Augment with per-move probe (already computed async) where needed
        # Include only moves not already in PV set
        for uci, sc in self.permove_cache.items():
            mv = chess.Move.from_uci(uci)
            if mv in seen:
                continue
            cp = score_cp_signed(sc, pov_is_white)
            feats = self._style_features(self.board, mv)
            reasons = self._reason_bits(style, feats)
            m_in = mate_in(sc, pov_is_white)
            metric = self._style_metric(style, feats) if style != "Normal" else 0.0
            cands.append((mv, float(cp), feats, reasons, m_in, metric, "probe"))

        if not cands:
            return None, ""

        # Mate override
        mates = [t for t in cands if t[4] is not None and t[4] > 0]
        if mates:
            best = min(mates, key=lambda t: t[4])
            return best[0], f"mate in {best[4]}"

        # Engine top eval (over *all* candidates we have)
        top_cp = max(t[1] for t in cands)
        top_mv = max(cands, key=lambda t: t[1])[0]

        if style == "Normal":
            return top_mv, "engine #1"

        # 1) Adjust scores with a style bonus in cp
        bonus = STYLE_CP_BONUS[style]
        best = max(cands, key=lambda t: t[1] + bonus * t[5])
        pick_mv, pick_cp, _, pick_reasons, _, pick_metric, source = best

        # 2) Diversity boost: if another move is close to engine #1 but more on-style, take it
        alts = [t for t in cands if t[0] != pick_mv and (top_cp - t[1]) <= DIVERSITY_MARGIN_CP]
        switched_for_diversity = False
        if alts:
            alt = max(alts, key=lambda t: t[5])  # most on-style among safe alts
            if alt[5] >= pick_metric + 1.0:      # require a meaningful style gap
                pick_mv, pick_cp, _, pick_reasons, _, _, source = alt
                switched_for_diversity = True

        # 3) Situation-aware safety caps (don’t allow big blunders)
        cap_cp, cap_wp = ADHERENCE[style]
        if top_cp >= 200:   # clearly winning → tighten
            cap_cp *= 0.5; cap_wp *= 0.5
        elif top_cp <= -50: # worse → loosen to create chances
            cap_cp *= 1.4; cap_wp *= 1.4

        drop_cp = top_cp - pick_cp
        drop_wp = win_prob(top_cp) - win_prob(pick_cp)
        if drop_cp > cap_cp or drop_wp > cap_wp:
            return top_mv, f"engine safety (Δ{int(drop_cp)}cp, Δ{drop_wp:.1f}%)"

        reason = ", ".join(pick_reasons) if pick_reasons else "engine tie-break"
        if switched_for_diversity:
            reason += "; within margin"
        if source == "probe":
            reason += " [style probe]"
        return pick_mv, reason

    # ---------- Cleanup ----------
    def destroy(self):
        try:
            if hasattr(self, "engine"):
                self.engine.close()
        finally:
            super().destroy()


if __name__ == "__main__":
    App().mainloop()
