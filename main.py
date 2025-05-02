#!/usr/bin/env python3
"""
Minimal Stockfish GUI for quick position evaluation.
Paste a FEN, choose which side you are, and the app will

* show Stockfish's **best move** and its evaluation from **your** side
* let you type **your own move** (SAN) and see the evaluation for that line

Requires
--------
    pip install python-chess
    sudo apt install stockfish  # or adjust STOCKFISH_PATH below
"""

import tkinter as tk
from tkinter import ttk, messagebox
import chess
import chess.engine
import threading
import os
import sys

# ------------------------------------------------------------
STOCKFISH_PATH = "/usr/games/stockfish"  # change if necessary
DEPTH = 18                                # or use Limit(time=2.0)
# ------------------------------------------------------------

class FenApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stockfish FEN analyser")
        self.geometry("650x300")
        self.resizable(False, False)

        # ---------- FEN input --------------------------------------------
        ttk.Label(self, text="Paste a FEN string:").pack(anchor="w", padx=10, pady=(10, 2))
        self.fen_var = tk.StringVar()
        self.fen_entry = ttk.Entry(self, width=95, textvariable=self.fen_var)
        self.fen_entry.pack(padx=10, fill="x")
        self.fen_entry.focus()

        # ---------- side selection ---------------------------------------
        self.side_var = tk.StringVar(value="auto")
        side_frame = ttk.Frame(self)
        side_frame.pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Label(side_frame, text="I am: ").pack(side="left")
        ttk.Radiobutton(side_frame, text="Auto (use FEN)", variable=self.side_var, value="auto").pack(side="left")
        ttk.Radiobutton(side_frame, text="White", variable=self.side_var, value="white").pack(side="left")
        ttk.Radiobutton(side_frame, text="Black", variable=self.side_var, value="black").pack(side="left")

        # ---------- control buttons --------------------------------------
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=8)
        self.best_btn = ttk.Button(btn_frame, text="Best move", command=self.launch_bestmove)
        self.best_btn.grid(row=0, column=0, padx=5)

        ttk.Label(btn_frame, text="Your move (SAN):").grid(row=0, column=1, padx=(20, 4))
        self.move_var = tk.StringVar()
        ttk.Entry(btn_frame, width=10, textvariable=self.move_var).grid(row=0, column=2)
        ttk.Button(btn_frame, text="Evaluate", command=self.launch_evalmove).grid(row=0, column=3, padx=5)

        # ---------- output ------------------------------------------------
        self.result_lbl = ttk.Label(self, text="", font=("Courier New", 12))
        self.result_lbl.pack(pady=6)

        self.status = ttk.Label(self, text="", relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

        # ---------- engine ----------------------------------------------
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        except FileNotFoundError:
            messagebox.showerror("Engine not found", f"Could not run Stockfish at {STOCKFISH_PATH}")
            self.destroy()

    # ================= helper methods ===================================
    def _get_board(self):
        fen = self.fen_var.get().strip()
        if not fen:
            messagebox.showinfo("No FEN", "Paste a FEN position first.")
            return None
        try:
            return chess.Board(fen)
        except ValueError as e:
            messagebox.showerror("Bad FEN", str(e))
            return None

    def _effective_side(self, side_to_move: bool) -> bool:
        choice = self.side_var.get()
        if choice == "white":
            return chess.WHITE
        if choice == "black":
            return chess.BLACK
        return side_to_move  # auto -> take the side to move from FEN

    def _fmt(self, score: chess.engine.PovScore) -> str:
        if score.is_mate():
            return f"M{score.mate()}"
        cp = score.score()
        sign = "+" if cp >= 0 else ""
        return f"{sign}{cp/100:.2f}"

    def _start_thread(self, target, *args):
        self.status.config(text="Thinking…")
        for w in (self.best_btn,):
            w.config(state="disabled")
        threading.Thread(target=target, args=args, daemon=True).start()

    def _show_result(self, txt):
        self.result_lbl.config(text=txt)
        self.status.config(text="Ready")
        for w in (self.best_btn,):
            w.config(state="normal")

    # ================= BEST MOVE path ====================================
    def launch_bestmove(self):
        board = self._get_board()
        if board:
            self._start_thread(self._analyse_bestmove, board)

    def _analyse_bestmove(self, board):
        info = self.engine.analyse(board, chess.engine.Limit(depth=DEPTH))
        best_mv = info["pv"][0]
        side = self._effective_side(board.turn)
        score = info["score"].white() if side == chess.WHITE else info["score"].black()
        txt = (f"Best: {board.san(best_mv):6} | Eval ({'White' if side==chess.WHITE else 'Black'}): "
               f"{self._fmt(score)} (depth {DEPTH})")
        self.after(0, self._show_result, txt)

    # ================= YOUR MOVE path ====================================
    def launch_evalmove(self):
        board = self._get_board()
        if not board:
            return
        mv_san = self.move_var.get().strip()
        if not mv_san:
            messagebox.showinfo("No move", "Enter a move in SAN first (e.g. Nf3).")
            return
        try:
            mv = board.parse_san(mv_san)
        except ValueError as e:
            messagebox.showerror("Illegal / invalid move", str(e))
            return
        board.push(mv)
        self._start_thread(self._analyse_evalmove, board, mv_san)

    def _analyse_evalmove(self, board, mv_san):
        info = self.engine.analyse(board, chess.engine.Limit(depth=DEPTH))
        side = self._effective_side(not board.turn)  # after push, board.turn is opponent
        score = info["score"].white() if side == chess.WHITE else info["score"].black()
        txt = (f"After {mv_san:6} | Eval ({'White' if side==chess.WHITE else 'Black'}): "
               f"{self._fmt(score)} (depth {DEPTH})")
        self.after(0, self._show_result, txt)

    # ================= CLEANUP ===========================================
    def destroy(self):
        try:
            if hasattr(self, "engine"):
                self.engine.close()
        finally:
            super().destroy()


if __name__ == "__main__":
    FenApp().mainloop()
