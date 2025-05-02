#!/usr/bin/env python3
"""
Chess‑in‑a‑flash: Interactive Stockfish sandbox
----------------------------------------------
Evaluate many candidate moves from the **same position** without
re‑loading the FEN each time.

### What’s new
* **Reset** button – instantly returns the board to the last loaded FEN.
* Engine eval runs after Reset so you see the baseline score.
* `Best move` preview no longer alters the real board – it just
  shows you the line and keeps the position intact (so you can still
  make your own try next).

All prior features remain (board flip, click‑to‑move, SAN entry, etc.).

Install once:
    python -m pip install python-chess pillow cairosvg
    sudo apt install stockfish
"""

import io, threading, tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import chess, chess.engine, chess.svg, cairosvg

STOCKFISH_PATH = "/usr/games/stockfish"
DEPTH = 18
BOARD_SIZE = 320; SQUARE = BOARD_SIZE // 8

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stockfish FEN analyser")
        self.geometry("840x640"); self.resizable(False, False)

        # ------------ FEN input --------------------------------------
        ttk.Label(self, text="Paste a FEN string:").pack(anchor="w", padx=10, pady=(10,2))
        self.fen_var = tk.StringVar()
        fen_entry = ttk.Entry(self, textvariable=self.fen_var, width=120)
        fen_entry.pack(padx=10, fill="x"); fen_entry.bind("<Return>",lambda e:self.load_fen())

        # ------------ side selector ----------------------------------
        self.side_var = tk.StringVar(value="auto")
        side_fr = ttk.Frame(self); side_fr.pack(anchor="w", padx=10, pady=(6,0))
        ttk.Label(side_fr, text="I am: ").pack(side="left")
        for txt,val in (("Auto (use FEN)","auto"),("White","white"),("Black","black")):
            ttk.Radiobutton(side_fr,text=txt,value=val,variable=self.side_var,command=self.refresh_view).pack(side="left")

        # ------------ buttons ---------------------------------------
        btn_fr = ttk.Frame(self); btn_fr.pack(pady=8)
        ttk.Button(btn_fr,text="Load FEN",command=self.load_fen).grid(row=0,column=0,padx=5)
        ttk.Button(btn_fr,text="Reset",command=self.reset_position).grid(row=0,column=1,padx=5)
        self.best_btn = ttk.Button(btn_fr,text="Best move",command=self.preview_bestmove)
        self.best_btn.grid(row=0,column=2,padx=5)
        ttk.Label(btn_fr,text="Your move (SAN):").grid(row=0,column=3,padx=(20,4))
        self.move_var = tk.StringVar()
        ttk.Entry(btn_fr,width=12,textvariable=self.move_var).grid(row=0,column=4)
        ttk.Button(btn_fr,text="Evaluate",command=self.eval_san).grid(row=0,column=5,padx=5)

        # ------------ evaluation label -------------------------------
        self.eval_lbl = ttk.Label(self, text="", font=("Courier New", 12))
        self.eval_lbl.pack(pady=6)

        # ------------ board -----------------------------------------
        self.canvas = tk.Canvas(self,width=BOARD_SIZE,height=BOARD_SIZE,highlightthickness=0)
        self.canvas.pack(); self.canvas.bind("<Button-1>",self.on_click)
        self._img_ref=None; self._sel=None

        # ------------ status bar ------------------------------------
        self.status = ttk.Label(self,text="",relief="sunken",anchor="w")
        self.status.pack(side="bottom",fill="x")

        # ------------ engine ---------------------------------------
        try: self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        except FileNotFoundError:
            messagebox.showerror("Engine not found",STOCKFISH_PATH); self.destroy(); return

        # internal boards
        self.board = chess.Board(); self.base_fen = self.board.fen()
        self.redraw(); self.async_eval()

    # --------------------------------------------------------------
    def my_side(self,turn):
        sel=self.side_var.get();
        if sel=="white": return chess.WHITE
        if sel=="black": return chess.BLACK
        return turn

    @staticmethod
    def fmt(score):
        return f"M{score.mate()}" if score.is_mate() else f"{'' if score.score()<0 else '+'}{score.score()/100:.2f}"

    # ---------- drawing -------------------------------------------
    def redraw(self,board=None,highlight_move=None):
        if board is None: board=self.board
        orientation=self.my_side(board.turn)
        svg=chess.svg.board(board=board,size=BOARD_SIZE,orientation=orientation,lastmove=highlight_move,colors={"square dark":"#b58863","square light":"#f0d9b5"})
        png=cairosvg.svg2png(bytestring=svg.encode())
        self._img_ref=ImageTk.PhotoImage(image=Image.open(io.BytesIO(png)))
        self.canvas.create_image(0,0,image=self._img_ref,anchor="nw")

    # ---------- engine helpers ------------------------------------
    def async_eval(self):
        self.status.config(text="Thinking…"); threading.Thread(target=self._worker_eval,daemon=True).start()
    def _worker_eval(self):
        info=self.engine.analyse(self.board,chess.engine.Limit(depth=DEPTH))
        self.after(0,self._finish_eval,info)
    def _finish_eval(self,info):
        side=self.my_side(self.board.turn^1)
        score=info["score"].white() if side==chess.WHITE else info["score"].black()
        who="White" if side==chess.WHITE else "Black"
        self.eval_lbl.config(text=f"Eval ({who}): {self.fmt(score)} (d{DEPTH})"); self.status.config(text="Ready")

    # ---------- UI callbacks --------------------------------------
    def load_fen(self):
        fen=self.fen_var.get().strip()
        if not fen: messagebox.showinfo("No FEN","Paste a FEN first."); return
        try: self.board=chess.Board(fen)
        except ValueError as e: messagebox.showerror("Bad FEN",str(e)); return
        self.base_fen=self.board.fen(); self._sel=None; self.redraw(); self.async_eval()
    def reset_position(self):
        self.board=chess.Board(self.base_fen); self._sel=None; self.redraw(); self.async_eval()
    def refresh_view(self):
        self.redraw(); self.async_eval()

    # ---------- click‑to‑move -------------------------------------
    def on_click(self,event):
        file,rank=event.x//SQUARE,7-(event.y//SQUARE); sq=chess.square(file,rank)
        if self._sel is None:
            if self.board.piece_at(sq): self._sel=sq
        else:
            mv=chess.Move(self._sel,sq); self._sel=None
            if mv in self.board.legal_moves:
                self.board.push(mv); self.redraw(); self.async_eval()
            else: messagebox.showwarning("Illegal","That move isn't legal.")

    # ---------- Best move preview (non‑destructive) ----------------
    def preview_bestmove(self):
        self.status.config(text="Thinking…"); threading.Thread(target=self._worker_best,daemon=True).start()
    def _worker_best(self):
        info=self.engine.analyse(self.board,chess.engine.Limit(depth=DEPTH))
        best=info["pv"][0]; brd=self.board.copy(); brd.push(best)
        self.after(0,self._finish_best,brd,best,info)
    def _finish_best(self,brd,best,info):
        self.redraw(board=brd,highlight_move=best)
        side=self.my_side(self.board.turn)  # board not changed, so side=turn
        score=info["score"].white() if side==chess.WHITE else info["score"].black()
        who="White" if side==chess.WHITE else "Black"
        self.eval_lbl.config(text=f"Best: {self.board.san(best):6} | Eval ({who}): {self.fmt(score)} (d{DEPTH})")
        self.status.config(text="Preview – position *not* changed (press Reset to clear)")

    # ---------- Evaluate SAN (destructive) ------------------------
    def eval_san(self):
        san=self.move_var.get().strip()
        if not san: messagebox.showinfo("No move","Enter a SAN move first."); return
        try: mv=self.board.parse_san(san)
        except ValueError as e: messagebox.showerror("Bad move",str(e)); return
        self.board.push(mv); self.redraw(); self.async_eval()

    # ---------- cleanup -------------------------------------------
    def destroy(self):
        try: self.engine.close()
        finally: super().destroy()

if __name__=="__main__":
    App().mainloop()
