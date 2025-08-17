#!/usr/bin/env python3
"""
Chess‑in‑a‑flash 2.1 — win probability shown **for you**
------------------------------------------------------
Whatever colour you select under **“I am:”**:

* **Positive score**  → good for *you*.
* **Win% Me:XX – Opp:YY**  → probability you win vs. the opponent.

All other features (reset, best‑move preview, click‑to‑move, board flip)
are unchanged.

Install once:
    python -m pip install python-chess pillow cairosvg
    sudo apt install stockfish
"""

import io, math, threading, tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import chess, chess.engine, chess.svg, cairosvg

STOCKFISH_PATH = "/usr/games/stockfish"  # adjust if needed
DEPTH = 18                                 # engine depth
BOARD_SIZE = 320; SQUARE = BOARD_SIZE // 8

# logistic curve constant — 75 % win chance ≈ +300 cp
K_LOG = 0.004

def win_prob(cp_signed: int) -> float:
    """Return probability (0‑100) that the *side whose score it is* wins."""
    return 100.0 / (1 + math.exp(-K_LOG * cp_signed))

class App(tk.Tk):
    # --------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.title("Stockfish FEN analyser")
        self.geometry("860x650"); self.resizable(False, False)

        # -------- FEN input ---------------------------------------
        ttk.Label(self,text="Paste a FEN string:").pack(anchor="w",padx=10,pady=(10,2))
        self.fen_var=tk.StringVar()
        fen_ent=ttk.Entry(self,textvariable=self.fen_var,width=122)
        fen_ent.pack(padx=10,fill="x"); fen_ent.bind("<Return>",lambda _ : self.load_fen())

        # -------- side selector -----------------------------------
        self.side_var=tk.StringVar(value="auto")
        side_fr=ttk.Frame(self); side_fr.pack(anchor="w",padx=10,pady=(6,0))
        ttk.Label(side_fr,text="I am: ").pack(side="left")
        for txt,val in (("Auto (use FEN)","auto"),("White","white"),("Black","black")):
            ttk.Radiobutton(side_fr,text=txt,value=val,variable=self.side_var,command=self.refresh_view).pack(side="left")

        # -------- buttons -----------------------------------------
        btn_fr=ttk.Frame(self); btn_fr.pack(pady=8)
        ttk.Button(btn_fr,text="Load FEN",command=self.load_fen).grid(row=0,column=0,padx=5)
        ttk.Button(btn_fr,text="Reset",command=self.reset_pos).grid(row=0,column=1,padx=5)
        self.best_btn=ttk.Button(btn_fr,text="Best move",command=self.preview_best)
        self.best_btn.grid(row=0,column=2,padx=5)
        ttk.Label(btn_fr,text="Your move (SAN):").grid(row=0,column=3,padx=(20,4))
        self.move_var=tk.StringVar(); ttk.Entry(btn_fr,width=12,textvariable=self.move_var).grid(row=0,column=4)
        ttk.Button(btn_fr,text="Evaluate",command=self.eval_san).grid(row=0,column=5,padx=5)

        # -------- eval label --------------------------------------
        self.eval_lbl=ttk.Label(self,text="",font=("Courier New",12))
        self.eval_lbl.pack(pady=6)

        # -------- board canvas ------------------------------------
        self.canvas=tk.Canvas(self,width=BOARD_SIZE,height=BOARD_SIZE,highlightthickness=0)
        self.canvas.pack(); self.canvas.bind("<Button-1>",self.on_click)
        self._img=None; self._sel=None

        # -------- status bar --------------------------------------
        self.status=ttk.Label(self,text="",relief="sunken",anchor="w")
        self.status.pack(side="bottom",fill="x")

        # -------- engine -----------------------------------------
        try:
            self.engine=chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        except FileNotFoundError:
            messagebox.showerror("Engine not found",STOCKFISH_PATH); self.destroy(); return

        # internal board
        self.board=chess.Board(); self.base_fen=self.board.fen()
        self.redraw(); self.async_eval()

    # ===== basics =================================================
    def my_side(self,turn):
        sel=self.side_var.get()
        if sel=="white": return chess.WHITE
        if sel=="black": return chess.BLACK
        return turn

    @staticmethod
    def fmt_cp(cp):
        return f"{'' if cp<0 else '+'}{cp/100:.2f}"

    def show_eval(self, cp_signed):
        """Update label with eval and win% from *my* perspective."""
        my_prob = win_prob(cp_signed)
        opp_prob = 100.0 - my_prob
        self.eval_lbl.config(text=f"Eval: {self.fmt_cp(cp_signed)}  |  Win% Me:{my_prob:.0f} – Opp:{opp_prob:.0f}  (d{DEPTH})")

    # ===== drawing ===============================================
    def redraw(self,board=None,highlight=None):
        if board is None: board=self.board
        svg=chess.svg.board(board=board,size=BOARD_SIZE,orientation=self.my_side(board.turn),lastmove=highlight)
        png=cairosvg.svg2png(bytestring=svg.encode())
        self._img=ImageTk.PhotoImage(Image.open(io.BytesIO(png)))
        self.canvas.create_image(0,0,image=self._img,anchor="nw")

    # ===== engine evaluation pipeline ============================
    def async_eval(self):
        self.status.config(text="Thinking…")
        threading.Thread(target=self._worker_eval,daemon=True).start()
    def _worker_eval(self):
        info=self.engine.analyse(self.board,chess.engine.Limit(depth=DEPTH))
        self.after(0,self._finish_eval,info)
    def _finish_eval(self,info):
        side=self.my_side(self.board.turn^1)  # side that just moved
        cp = info["score"].white().score() if side==chess.WHITE else info["score"].black().score()
        self.show_eval(cp); self.status.config(text="Ready")

    # ===== UI actions ============================================
    def load_fen(self):
        fen=self.fen_var.get().strip()
        if not fen: messagebox.showinfo("No FEN","Paste a FEN first."); return
        try: self.board=chess.Board(fen)
        except ValueError as e: messagebox.showerror("Bad FEN",str(e)); return
        self.base_fen=self.board.fen(); self._sel=None; self.redraw(); self.async_eval()
    def reset_pos(self):
        self.board=chess.Board(self.base_fen); self._sel=None; self.redraw(); self.async_eval()
    def refresh_view(self):
        self.redraw(); self.async_eval()

    # ===== click‑to‑move =========================================
    def on_click(self,event):
        file,rank = event.x//SQUARE, 7-(event.y//SQUARE); sq=chess.square(file,rank)
        if self._sel is None:
            if self.board.piece_at(sq): self._sel=sq
        else:
            mv=chess.Move(self._sel,sq); self._sel=None
            if mv in self.board.legal_moves:
                self.board.push(mv); self.redraw(); self.async_eval()
            else:
                messagebox.showwarning("Illegal","That move isn't legal.")

    # ===== Best‑move preview (non‑destructive) ===================
    def preview_best(self):
        self.status.config(text="Thinking…")
        threading.Thread(target=self._worker_best,daemon=True).start()
    def _worker_best(self):
        info=self.engine.analyse(self.board,chess.engine.Limit(depth=DEPTH))
        best=info["pv"][0]; brd=self.board.copy(); brd.push(best)
        self.after(0,self._finish_best,brd,best,info)
    def _finish_best(self,brd,best,info):
        self.redraw(board=brd,highlight=best)
        side=self.my_side(self.board.turn)  # actual board unchanged
        cp = info["score"].white().score() if side==chess.WHITE else info["score"].black().score()
        my_prob=win_prob(cp); opp_prob=100-my_prob
        self.eval_lbl.config(text=f"Best: {self.board.san(best):6} | {self.fmt_cp(cp)} | Win% Me:{my_prob:.0f} – Opp:{opp_prob:.0f} (d{DEPTH})")
        self.status.config(text="Preview – actual board unchanged (Reset to clear)")

    # ===== Evaluate SAN ==========================================
    def eval_san(self):
        san=self.move_var.get().strip()
        if not san: messagebox.showinfo("No move","Enter a SAN move first."); return
        try: mv=self.board.parse_san(san)
        except ValueError as e: messagebox.showerror("Bad move",str(e)); return
        self.board.push(mv); self.redraw(); self.async_eval()

    # ===== cleanup ===============================================
    def destroy(self):
        try: self.engine.close()
        finally: super().destroy()

if __name__=="__main__":
    App().mainloop()
