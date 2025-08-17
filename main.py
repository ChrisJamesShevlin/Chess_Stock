#!/usr/bin/env python3
"""
Chess-in-a-flash — Coach for exactly TWO systems:
  • Colle (White)
  • KID (Black)

Install:
    python -m pip install python-chess pillow cairosvg
    sudo apt install stockfish
"""

import io, os, math, shutil, threading, tkinter as tk
from typing import Optional, List
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import chess, chess.engine, chess.svg, cairosvg

DEPTH = 18
BOARD_SIZE = 320
SQUARE = BOARD_SIZE // 8

# logistic curve constant — 75% win chance ~ +300 cp
K_LOG = 0.004


def win_prob(cp_signed: int) -> float:
    return 100.0 / (1.0 + math.exp(-K_LOG * cp_signed))


def fmt_cp(cp: int) -> str:
    return f"{cp/100:.2f}" if cp < 0 else f"+{cp/100:.2f}"


def score_cp_signed(info_score: chess.engine.PovScore, pov_is_white: bool) -> int:
    pov = chess.WHITE if pov_is_white else chess.BLACK
    sc = info_score.pov(pov)
    if sc.is_mate():
        m = sc.mate()
        return 100000 if (m is not None and m > 0) else -100000
    cp = sc.score(mate_score=100000)
    return int(cp if cp is not None else 0)


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


# -------------------------- Coach logic --------------------------

def _try_san(board: chess.Board, san_list: List[str]) -> Optional[chess.Move]:
    """Return first legal move that matches any SAN in the list."""
    for san in san_list:
        try:
            mv = board.parse_san(san)
        except Exception:
            continue
        if mv in board.legal_moves:
            return mv
    return None


def suggest_colle(board: chess.Board) -> Optional[chess.Move]:
    """
    Colle policy (White):
      d4, Nf3, e3, Bd3, O-O, Nbd2, c3, Re1, then e4; else c4 if e4 is clamped.
    """
    if board.turn != chess.WHITE:
        return None
    order = ["d4", "Nf3", "e3", "Bd3", "Bf4", "O-O", "Nbd2", "c3", "Re1"]
    mv = _try_san(board, order)
    if mv: return mv
    mv = _try_san(board, ["e4"])
    if mv: return mv
    mv = _try_san(board, ["c4"])
    if mv: return mv
    return _try_san(board, ["Qe2", "h3", "a4"])


def suggest_kid_black(board: chess.Board) -> Optional[chess.Move]:
    """
    KID shell (Black) used against anything:
      Nf6, g6, Bg7, d6, O-O, then e5/c5/f5 depending on position.
    """
    if board.turn != chess.BLACK:
        return None
    base = ["Nf6", "g6", "Bg7", "d6", "O-O"]
    mv = _try_san(board, base)
    if mv: return mv
    follow = ["e5", "c5", "f5", "Nc6", "Nbd7", "a5", "Na6"]
    return _try_san(board, follow)


# -------------------------- GUI App ------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stockfish FEN analyser + Coach (Colle / KID)")
        self.geometry("900x690")
        self.resizable(False, False)

        # -------- FEN input ---------------------------------------
        ttk.Label(self, text="Paste a FEN string:").pack(anchor="w", padx=10, pady=(10, 2))
        self.fen_var = tk.StringVar()
        fen_ent = ttk.Entry(self, textvariable=self.fen_var, width=120)
        fen_ent.pack(padx=10, fill="x")
        fen_ent.bind("<Return>", lambda _ : self.load_fen())

        # -------- Coach controls ----------------------------------
        ctl = ttk.Frame(self); ctl.pack(anchor="w", padx=10, pady=(6, 0))
        self.coach_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="Coach mode", variable=self.coach_on,
                        command=self.refresh_view).pack(side="left", padx=(0, 10))

        ttk.Label(ctl, text="System:").pack(side="left")
        self.system_var = tk.StringVar(value="KID (Black)")
        systems = ["Colle (White)", "KID (Black)"]  # only two choices
        ttk.Combobox(ctl, textvariable=self.system_var, values=systems, state="readonly",
                     width=18).pack(side="left", padx=(4, 12))

        # -------- side selector (board orientation only) ----------
        self.side_var = tk.StringVar(value="auto")
        side_fr = ttk.Frame(self); side_fr.pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Label(side_fr, text="I am: ").pack(side="left")
        for txt, val in (("Auto (use FEN)", "auto"), ("White", "white"), ("Black", "black")):
            ttk.Radiobutton(side_fr, text=txt, value=val, variable=self.side_var,
                            command=self.refresh_view).pack(side="left")

        # -------- buttons -----------------------------------------
        btn_fr = ttk.Frame(self); btn_fr.pack(pady=8)
        ttk.Button(btn_fr, text="Load FEN", command=self.load_fen).grid(row=0, column=0, padx=5)
        ttk.Button(btn_fr, text="Reset",    command=self.reset_pos).grid(row=0, column=1, padx=5)
        self.best_btn = ttk.Button(btn_fr, text="Best move", command=self.preview_best)
        self.best_btn.grid(row=0, column=2, padx=5)
        ttk.Label(btn_fr, text="Your move (SAN):").grid(row=0, column=3, padx=(20, 4))
        self.move_var = tk.StringVar()
        ttk.Entry(btn_fr, width=12, textvariable=self.move_var).grid(row=0, column=4)
        ttk.Button(btn_fr, text="Evaluate", command=self.eval_san).grid(row=0, column=5, padx=5)

        # -------- eval label --------------------------------------
        self.eval_lbl = ttk.Label(self, text="", font=("Courier New", 12))
        self.eval_lbl.pack(pady=6)

        # -------- board canvas ------------------------------------
        self.canvas = tk.Canvas(self, width=BOARD_SIZE, height=BOARD_SIZE, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self._img = None
        self._sel = None

        # -------- status bar --------------------------------------
        self.status = ttk.Label(self, text="", relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

        # -------- engine ------------------------------------------
        path = find_stockfish_path()
        if not path:
            messagebox.showerror("Engine not found",
                                 "Could not find Stockfish.\nTry:  sudo apt install stockfish\n"
                                 "Or set env var STOCKFISH_PATH.")
            self.destroy(); return
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            self.engine.configure({
                "Threads": max(1, os.cpu_count() // 2),
                "Hash": 256
            })
        except Exception as e:
            messagebox.showerror("Engine error", f"Failed to start Stockfish at:\n{path}\n\n{e}")
            self.destroy(); return

        # internal board
        self.board = chess.Board()
        self.base_fen = self.board.fen()
        self.redraw()
        self.async_eval()

    # ===== basics =================================================
    def my_side(self, turn):
        sel = self.side_var.get()
        if sel == "white": return chess.WHITE
        if sel == "black": return chess.BLACK
        return turn  # auto: orient to side to move

    def show_eval(self, cp_signed: int):
        my_prob = win_prob(cp_signed)
        opp_prob = 100.0 - my_prob
        self.eval_lbl.config(
            text=f"Eval: {fmt_cp(cp_signed)}  |  Win% Me:{my_prob:.0f} – Opp:{opp_prob:.0f}  (d{DEPTH})"
        )

    # ===== coach suggestion (only on YOUR turn, only for chosen system) ==
    def coach_move(self) -> Optional[chess.Move]:
        if not self.coach_on.get():
            return None

        sys_name = self.system_var.get()
        if self.board.turn == chess.WHITE and sys_name == "Colle (White)":
            return suggest_colle(self.board)
        if self.board.turn == chess.BLACK and sys_name == "KID (Black)":
            return suggest_kid_black(self.board)
        return None  # wrong colour for selected system → no arrow

    # ===== drawing ===============================================
    def redraw(self, board=None, highlight=None):
        if board is None:
            board = self.board

        arrows = []
        sugg = self.coach_move()
        if sugg:
            arrows.append(chess.svg.Arrow(sugg.from_square, sugg.to_square, color="#34d399"))  # green
        if highlight:
            arrows.append(chess.svg.Arrow(highlight.from_square, highlight.to_square, color="#f59e0b"))  # orange

        svg = chess.svg.board(
            board=board,
            size=BOARD_SIZE,
            orientation=self.my_side(board.turn),
            arrows=arrows
        )
        png = cairosvg.svg2png(bytestring=svg.encode())
        self._img = ImageTk.PhotoImage(Image.open(io.BytesIO(png)))
        self.canvas.create_image(0, 0, image=self._img, anchor="nw")

    # ===== engine evaluation pipeline ============================
    def async_eval(self):
        self.status.config(text="Thinking…")
        threading.Thread(target=self._worker_eval, daemon=True).start()

    def _worker_eval(self):
        info = self.engine.analyse(self.board, chess.engine.Limit(depth=DEPTH))
        self.after(0, self._finish_eval, info)

    def _finish_eval(self, info):
        pov_is_white = (self.my_side(self.board.turn ^ 1) == chess.WHITE)
        cp = score_cp_signed(info["score"], pov_is_white)
        self.show_eval(cp)
        self.status.config(text="Ready")
        self.redraw()  # update suggestion for the new side to move

    # ===== UI actions ============================================
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
        self.redraw()
        self.async_eval()

    def reset_pos(self):
        self.board = chess.Board(self.base_fen)
        self._sel = None
        self.redraw()
        self.async_eval()

    def refresh_view(self):
        self.redraw()
        self.async_eval()

    # ===== click-to-move (respects orientation) ==================
    def on_click(self, event):
        # Map mouse to square based on current board orientation (who is at bottom now)
        orient = self.my_side(self.board.turn)
        fx = int(event.x // SQUARE)
        fy = int(event.y // SQUARE)
        if not (0 <= fx < 8 and 0 <= fy < 8):
            return

        if orient == chess.WHITE:
            file, rank = fx, 7 - fy
        else:
            file, rank = 7 - fx, fy

        sq = chess.square(file, rank)

        if self._sel is None:
            if self.board.piece_at(sq):
                self._sel = sq
        else:
            mv = chess.Move(self._sel, sq)
            self._sel = None
            if mv in self.board.legal_moves:
                self.board.push(mv)
                self.redraw()
                self.async_eval()
            else:
                messagebox.showwarning("Illegal", "That move isn't legal.")

    # ===== Best-move preview (non-destructive) ===================
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
        pov_is_white = (self.my_side(self.board.turn) == chess.WHITE)  # preview from current POV
        cp = score_cp_signed(info["score"], pov_is_white)
        my_prob = win_prob(cp); opp_prob = 100 - my_prob
        self.eval_lbl.config(
            text=f"Best: {self.board.san(best):6} | {fmt_cp(cp)} | Win% Me:{my_prob:.0f} – Opp:{opp_prob:.0f} (d{DEPTH})"
        )
        self.status.config(text="Preview – actual board unchanged (Reset to clear)")

    # ===== Evaluate SAN ==========================================
    def eval_san(self):
        san = self.move_var.get().strip()
        if not san:
            messagebox.showinfo("No move", "Enter a SAN move first."); return
        try:
            mv = self.board.parse_san(san)
        except ValueError as e:
            messagebox.showerror("Bad move", str(e)); return
        self.board.push(mv)
        self.redraw()
        self.async_eval()

    # ===== cleanup ===============================================
    def destroy(self):
        try:
            if hasattr(self, "engine"):
                self.engine.close()
        finally:
            super().destroy()


if __name__ == "__main__":
    App().mainloop()
