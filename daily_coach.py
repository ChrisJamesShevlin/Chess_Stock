#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stockfish FEN analyser + Lichess Explorer Assist — No Arrows + Dual Eval Buckets

Changes from your previous version:
- ❌ Removed ALL arrows (no blue pick arrow, no orange preview arrow).
- 🔄 Still updates after every move exactly as before.
- 🔍 Displays **bucketed evaluations** instead of Win%: 
    Winning / Slightly Better / Equal / Slightly Worse / Losing
  for **you** and for **your opponent** simultaneously.
- ✅ Keeps engine strength, depth, MultiPV, Explorer/Cloud toggles, etc.
- 📝 Labels now avoid percent; cp values remain in some status lines (per your request).

Install (unchanged):
    python -m pip install python-chess pillow cairosvg requests
    sudo apt install stockfish
"""

import io, os, math, shutil, threading
from typing import Optional, List, Tuple, Dict
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import chess, chess.engine, chess.svg, cairosvg
import requests

# ---------- Base Config ----------
DEPTH = 18            # Max-depth for main think (tweak if you like)
MULTIPV = 8           # Pull a few candidates
PER_MOVE_DEPTH = 7    # Quick probe for extra candidates
MAX_PROBE_MOVES = 6
BOARD_SIZE = 360
SQUARE = BOARD_SIZE // 8
K_LOG = 0.004  # used for internal win% curve (retained, but not displayed)

# ---------- Lichess API ----------
LICHESS_EXPLORER = "https://explorer.lichess.ovh/lichess"
LICHESS_CLOUD_EVAL = "https://lichess.org/api/cloud-eval"

_http = requests.Session()
_http.headers.update({"User-Agent": "ICCF-Assistant/NoArrowsBuckets/1.0 (+contact@example.com)"})
HTTP_TIMEOUT = 7.0

_explorer_cache: Dict[str, dict] = {}
_cloudeval_cache: Dict[str, dict] = {}

# ---------- ASCII status helpers ----------
ASCII_STATUS = True
_ASCII_MAP = str.maketrans({"—": "-", "–": "-", "Δ": "d", "≤": "<=", "≥": ">=", "≈": "~"})
def _ascii(s: str) -> str: return s.translate(_ASCII_MAP)
def set_status(widget, s: str): widget.config(text=_ascii(s) if ASCII_STATUS else s)

# ---------- Math & bucket helpers ----------
def win_prob(cp_signed: int) -> float:
    # kept for internal use if needed; not shown in UI now
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

# Bucket mapping
#   Winning: ≥ +300
#   Slightly Better: +100..+299
#   Equal: -99..+99
#   Slightly Worse: -299..-100
#   Losing: ≤ -300

def eval_bucket(cp_signed: int) -> str:
    if cp_signed >= 300: return "Winning"
    if 100 <= cp_signed <= 299: return "Slightly Better"
    if -99 <= cp_signed <= 99: return "Equal"
    if -299 <= cp_signed <= -100: return "Slightly Worse"
    return "Losing"  # cp <= -300

# ---------- Engine path ----------
def find_stockfish_path() -> Optional[str]:
    envp = os.environ.get("STOCKFISH_PATH")
    if envp and os.path.isfile(envp) and os.access(envp, os.X_OK):
        return envp
    which = shutil.which("stockfish")
    if which: return which
    cand = "/usr/games/stockfish"
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    return None

# ---------- Lichess helpers ----------
def lichess_explorer(fen: str, top_moves: int = 50) -> Optional[dict]:
    cache_key = f"{fen}|{top_moves}"
    if cache_key in _explorer_cache:
        return _explorer_cache[cache_key]
    params = {"fen": fen, "variant": "standard", "moves": max(1, int(top_moves)), "topGames": 0}
    try:
        r = _http.get(LICHESS_EXPLORER, params=params, timeout=HTTP_TIMEOUT)
        if r.ok:
            data = r.json()
            _explorer_cache[cache_key] = data
            return data
    except Exception:
        pass
    return None

def lichess_cloud_eval(fen: str, multi_pv: int = 5) -> Optional[dict]:
    key = f"{fen}|{multi_pv}"
    if key in _cloudeval_cache:
        return _cloudeval_cache[key]
    params = {"fen": fen, "multiPv": max(1, multi_pv)}
    try:
        r = _http.get(LICHESS_CLOUD_EVAL, params=params, timeout=HTTP_TIMEOUT)
        if r.ok:
            data = r.json()
            _cloudeval_cache[key] = data
            return data
    except Exception:
        pass
    return None

# ---------- Explorer EV / counts ----------
def explorer_bonus_for_move(move_uci: str, ex_data: dict, min_sample: int):
    """
    Returns (bonus_cp, info_dict)
    - bonus_cp is 0 if under min_sample (but info still contains true games/drawP).
    - info_dict: {"games","winP","drawP","avgElo","filtered","ev_cp"}.
    """
    info = {"games": 0, "winP": 0.0, "drawP": 0.0, "avgElo": 0, "filtered": False, "ev_cp": 0.0}
    if not ex_data or "moves" not in ex_data:
        return 0.0, info

    found = next((m for m in ex_data["moves"] if m.get("uci") == move_uci), None)
    if not found:
        return 0.0, info

    w = float(found.get("white", 0))
    d = float(found.get("draws", 0))
    b = float(found.get("black", 0))
    tot = w + d + b
    games = int(found.get("gameCount") or tot)
    avg = int(found.get("averageRating") or ex_data.get("averageRating", 0))

    denom = max(1.0, tot)
    winP  = 100.0 * w / denom
    drawP = 100.0 * d / denom

    # cp-like EV (prefer wins, penalize drawish)
    ev_cp = (winP - 50.0) - 0.35 * (drawP - 30.0)
    ev_cp = max(-60.0, min(60.0, ev_cp))

    info.update({"games": games, "winP": winP, "drawP": drawP, "avgElo": avg, "ev_cp": float(ev_cp)})
    if games < min_sample:
        info["filtered"] = True
        return 0.0, info
    return float(ev_cp), info

# ---------- App ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stockfish (Max) + Lichess Explorer — Buckets, No Arrows")
        self.geometry("1120x880"); self.resizable(True, True)

        # Fullscreen bindings
        self.is_fullscreen = False; self.prev_geometry = None
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda e: self.exit_fullscreen())

        # FEN input
        ttk.Label(self, text="Paste a FEN string:").pack(anchor="w", padx=10, pady=(10, 2))
        self.fen_var = tk.StringVar()
        fen_ent = ttk.Entry(self, textvariable=self.fen_var, width=160)
        fen_ent.pack(padx=10, fill="x")
        fen_ent.bind("<Return>", lambda _ : self.load_fen())

        # Color / orientation
        play_fr = ttk.Frame(self); play_fr.pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Label(play_fr, text="I'm playing as:").pack(side="left")
        self.play_color = tk.StringVar(value="white")
        ttk.Radiobutton(play_fr, text="White", value="white",
                        variable=self.play_color, command=self.on_pick_color).pack(side="left")
        ttk.Radiobutton(play_fr, text="Black", value="black",
                        variable=self.play_color, command=self.on_pick_color).pack(side="left")
        opt_fr = ttk.Frame(self); opt_fr.pack(anchor="w", padx=10, pady=(0, 4))
        self.follow_turn = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_fr, text="Flip with side to move",
                        variable=self.follow_turn, command=self.redraw).pack(side="left", padx=(0, 10))
        ttk.Button(opt_fr, text="Flip now", command=self.flip_once).pack(side="left", padx=(0, 16))

        # Lichess Assist controls (Explorer/Cloud kept)
        assist_fr = ttk.LabelFrame(self, text="Lichess Assist"); assist_fr.pack(anchor="w", padx=10, pady=8, fill="x")
        self.use_explorer = tk.BooleanVar(value=True)
        self.use_cloud = tk.BooleanVar(value=False)
        ttk.Checkbutton(assist_fr, text="Use Explorer", variable=self.use_explorer,
                        command=lambda: self.async_eval()).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(assist_fr, text="Use Cloud Eval", variable=self.use_cloud,
                        command=lambda: self.async_eval()).pack(side="left", padx=(0, 18))
        ttk.Label(assist_fr, text="Explorer weight (cp):").pack(side="left")
        self.explorer_weight = tk.IntVar(value=50)
        ttk.Scale(assist_fr, from_=0, to=120, orient="horizontal",
                  variable=self.explorer_weight, command=lambda _:_).pack(side="left", padx=(6,16), fill="x", expand=True)
        ttk.Label(assist_fr, text="Min games:").pack(side="left")
        self.min_sample = tk.IntVar(value=10)
        ttk.Spinbox(assist_fr, from_=0, to=10000, width=6, textvariable=self.min_sample,
                    command=lambda: self.async_eval()).pack(side="left", padx=(6,8))
        ttk.Label(assist_fr, text="Top moves:").pack(side="left", padx=(12,4))
        self.top_moves = tk.IntVar(value=50)
        ttk.Spinbox(assist_fr, from_=5, to=100, width=5, textvariable=self.top_moves,
                    command=lambda: self.async_eval()).pack(side="left", padx=(0,8))
        self.explorer_status = tk.StringVar(value="Explorer: —")
        ttk.Label(assist_fr, textvariable=self.explorer_status).pack(side="left", padx=(12,0))

        # Buttons
        btn_fr = ttk.Frame(self); btn_fr.pack(pady=8)
        ttk.Button(btn_fr, text="Load FEN", command=self.load_fen).grid(row=0, column=0, padx=5)
        ttk.Button(btn_fr, text="Reset",    command=self.reset_pos).grid(row=0, column=1, padx=5)
        ttk.Button(btn_fr, text="Fullscreen", command=self.toggle_fullscreen).grid(row=0, column=2, padx=5)
        self.best_btn = ttk.Button(btn_fr, text="Best move (preview)", command=self.preview_best)
        self.best_btn.grid(row=0, column=3, padx=5)

        # Eval label + board
        self.eval_lbl = ttk.Label(self, text="", font=("Courier New", 12)); self.eval_lbl.pack(pady=6)
        self.canvas = tk.Canvas(self, width=BOARD_SIZE, height=BOARD_SIZE, highlightthickness=0)
        self.canvas.pack(); self.canvas.bind("<Button-1>", self.on_click)
        self._img = None; self._sel = None

        # Status bar
        self.status = ttk.Label(self, text="", relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

        # Engine (MAX strength — no Elo limiting)
        path = find_stockfish_path()
        if not path:
            messagebox.showerror("Engine not found",
                                 "Could not find Stockfish.\nTry: sudo apt install stockfish\nor set STOCKFISH_PATH.")
            self.destroy(); return
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            # Full strength; set threads/hash as desired
            self.engine.configure({"Threads": 1, "Hash": 128})
            self.engine_lock = threading.Lock()
        except Exception as e:
            messagebox.showerror("Engine error", f"Failed to start Stockfish at:\n{path}\n\n{e}")
            self.destroy(); return

        # State
        self.board = chess.Board(); self.base_fen = self.board.fen()
        self.fixed_bottom_is_white = True
        self.last_infos: Optional[List[chess.engine.InfoDict]] = None
        self.permove_cache: Dict[str, chess.engine.PovScore] = {}
        self.pick_move: Optional[chess.Move] = None
        self.last_eval_fen: Optional[str] = None
        self.eval_running = False; self.latest_request = 0
        self._last_explorer: Optional[dict] = None
        self._last_cloud: Optional[dict] = None

        # Kick off
        self.on_pick_color(); self.redraw(); self.async_eval()

    # ---------- Engine wrappers ----------
    def _restart_engine(self):
        try:
            if hasattr(self, "engine") and self.engine: self.engine.close()
        except Exception: pass
        path = find_stockfish_path()
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": 1, "Hash": 128})

    def _safe_analyse(self, board: chess.Board, limit: chess.engine.Limit, **kwargs):
        try:
            with self.engine_lock:
                return self.engine.analyse(board, limit, **kwargs)
        except chess.engine.EngineTerminatedError:
            self._restart_engine()
            with self.engine_lock:
                return self.engine.analyse(board, limit, **kwargs)

    # ---------- Fullscreen ----------
    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.prev_geometry = self.geometry()
            try: self.attributes("-fullscreen", True)
            except Exception: self.state("zoomed")
        else:
            self.exit_fullscreen()

    def exit_fullscreen(self):
        try: self.attributes("-fullscreen", False)
        except Exception: pass
        if self.prev_geometry: self.geometry(self.prev_geometry)
        self.is_fullscreen = False

    # ---------- Orientation ----------
    def my_color_is_white(self) -> bool:
        return self.play_color.get() == "white"

    def on_pick_color(self):
        self.follow_turn.set(False)
        self.fixed_bottom_is_white = (self.play_color.get() == "white")
        if self.last_eval_fen != self.board.fen(): self.async_eval()

    def bottom_color(self) -> chess.Color:
        return (self.board.turn if self.follow_turn.get()
                else (chess.WHITE if self.fixed_bottom_is_white else chess.BLACK))

    def flip_once(self):
        if self.follow_turn.get(): self.follow_turn.set(False)
        self.fixed_bottom_is_white = not self.fixed_bottom_is_white; self.redraw()

    # ---------- Drawing (NO ARROWS) ----------
    def redraw(self, board: Optional[chess.Board] = None, highlight: Optional[chess.Move] = None):
        if board is None: board = self.board
        # Intentionally no arrows; ignore highlight parameter
        svg = chess.svg.board(board=board, size=BOARD_SIZE, orientation=self.bottom_color())
        png = cairosvg.svg2png(bytestring=svg.encode())
        self._img = ImageTk.PhotoImage(Image.open(io.BytesIO(png)))
        self.canvas.create_image(0, 0, image=self._img, anchor="nw")

    # ---------- Engine eval + Lichess data ----------
    def async_eval(self):
        self.latest_request += 1
        if self.eval_running: return
        self.eval_running = True
        gen = self.latest_request; brd = self.board.copy()
        set_status(self.status, "Thinking...")
        threading.Thread(target=self._worker_eval, args=(gen, brd), daemon=True).start()

    def _worker_eval(self, gen: int, brd: chess.Board):
        try:
            pick_depth, mpv, probe_depth = DEPTH, min(MULTIPV, 10), PER_MOVE_DEPTH
            info_list = self._safe_analyse(brd, chess.engine.Limit(depth=pick_depth), multipv=mpv)

            # gather unique PV heads
            unique_moves, seen = [], set()
            for inf in info_list:
                pv = inf.get("pv")
                if not pv: continue
                mv = pv[0]
                if mv not in seen: seen.add(mv); unique_moves.append(mv)

            # quick probe a few high-priority moves not in MultiPV
            permove_scores: Dict[str, chess.engine.PovScore] = {}
            if len(unique_moves) < 2:
                legals = list(brd.legal_moves)
                def quick_priority(b: chess.Board, mv: chess.Move) -> float:
                    piece = b.piece_at(mv.from_square); to_piece = b.piece_at(mv.to_square)
                    is_cap = (to_piece is not None) or b.is_en_passant(mv)
                    is_promo = mv.promotion is not None; gives_chk = b.gives_check(mv); castle = b.is_castling(mv)
                    def cw(sq): f=chess.square_file(sq); r=chess.square_rank(sq); return -((f-3.5)**2+(r-3.5)**2)
                    center_gain = (cw(mv.to_square)-cw(mv.from_square)) if piece else 0.0
                    pawn_thrust = abs(chess.square_rank(mv.to_square)-chess.square_rank(mv.from_square)) if (piece and piece.piece_type==chess.PAWN and not is_cap) else 0.0
                    dev = 1.0 if (piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(mv.from_square) in (0,7)) else 0.0
                    return 4.0*gives_chk + 3.0*castle + 3.0*(is_cap or is_promo) + 1.5*max(0.0,center_gain) + 1.2*dev + 1.0*pawn_thrust
                legals_sorted = sorted(legals, key=lambda m: quick_priority(brd, m), reverse=True)
                probe_moves = [m for m in legals_sorted if m not in set(unique_moves)][:MAX_PROBE_MOVES]
                for mv in probe_moves:
                    info = self._safe_analyse(brd, chess.engine.Limit(depth=min(probe_depth, PER_MOVE_DEPTH)),
                                              searchmoves=[mv])
                    if "score" in info: permove_scores[mv.uci()] = info["score"]

            # Lichess data
            fen = brd.fen()
            ex_data = lichess_explorer(fen, top_moves=int(getattr(self, "top_moves", tk.IntVar(value=50)).get())) \
                      if self.use_explorer.get() else None
            cloud = lichess_cloud_eval(fen, multi_pv=mpv) if self.use_cloud.get() else None

            self.after(0, self._finish_eval, gen, brd.fen(), info_list, permove_scores, ex_data, cloud)

        except Exception as e:
            self.after(0, lambda: set_status(self.status, f"Engine error: {e}"))
            self.eval_running = False

    def _finish_eval(self, gen: int, fen: str, info_list: list,
                     permove_scores: Dict[str, chess.engine.PovScore],
                     ex_data: Optional[dict], cloud: Optional[dict]):
        if gen != self.latest_request or fen != self.board.fen():
            self.eval_running = False
            if self.latest_request > gen: self.async_eval()
            return
        self.last_infos = info_list; self.permove_cache = permove_scores
        self.last_eval_fen = fen; self._last_explorer = ex_data; self._last_cloud = cloud

        # Explorer status
        if self.use_explorer.get():
            if ex_data and isinstance(ex_data, dict):
                n = len(ex_data.get("moves", []))
                self.explorer_status.set(f"Explorer: {n} moves")
            else:
                self.explorer_status.set("Explorer: no data / error")
        else:
            self.explorer_status.set("Explorer: off")

        # Headline eval (now buckets for you & opponent)
        pov_is_white = self.my_color_is_white()
        cp = score_cp_signed(info_list[0]["score"], pov_is_white) if info_list else 0
        self.show_eval_buckets(cp)

        # Choose pick (engine cp + Explorer bias) — retained behavior
        self.apply_best_pick()
        self.redraw()

        self.eval_running = False
        if self.latest_request > gen: self.async_eval()

    def show_eval_buckets(self, cp_signed: int):
        # Me bucket and opponent bucket (opponent sees negated cp)
        me_b = eval_bucket(cp_signed)
        opp_b = eval_bucket(-cp_signed)
        self.eval_lbl.config(text=f"Eval — Me: {me_b} | Opp: {opp_b}")

    # ---------- UI actions ----------
    def load_fen(self):
        fen = self.fen_var.get().strip()
        if not fen: messagebox.showinfo("No FEN", "Paste a FEN first."); return
        try: self.board = chess.Board(fen)
        except ValueError as e: messagebox.showerror("Bad FEN", str(e)); return
        self.base_fen = self.board.fen(); self._sel = None
        self.pick_move = None
        self.redraw(); self.async_eval()

    def reset_pos(self):
        self.board = chess.Board(self.base_fen); self._sel = None
        self.pick_move = None
        self.redraw(); self.async_eval()

    def _make_move_with_optional_promotion(self, from_sq: chess.Square, to_sq: chess.Square) -> chess.Move:
        piece = self.board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if to_rank in (0, 7):
                ans = simpledialog.askstring("Promotion", "Promote to (q, r, b, n)?", parent=self)
                promo_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
                promo = promo_map.get((ans or "q").lower()[0], chess.QUEEN)
                return chess.Move(from_sq, to_sq, promotion=promo)
        return chess.Move(from_sq, to_sq)

    def on_click(self, event):
        bottom = self.bottom_color()
        fx, fy = int(event.x // SQUARE), int(event.y // SQUARE)
        if not (0 <= fx < 8 and 0 <= fy < 8): return
        file, rank = (fx, 7 - fy) if bottom == chess.WHITE else (7 - fx, fy)
        sq = chess.square(file, rank)
        if self._sel is None:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.board.turn: self._sel = sq
        else:
            mv = self._make_move_with_optional_promotion(self._sel, sq); self._sel = None
            if self.board.is_legal(mv):
                self.board.push(mv); self.pick_move = None
                self.redraw(); self.async_eval()
            else:
                messagebox.showwarning("Illegal", "That move isn't legal.")

    # ---------- Preview (no arrow rendering) ----------
    def preview_best(self):
        set_status(self.status, "Thinking...")
        threading.Thread(target=self._worker_best, daemon=True).start()

    def _worker_best(self):
        try:
            info = self._safe_analyse(self.board, chess.engine.Limit(depth=DEPTH))
            best = info["pv"][0]; brd = self.board.copy(); brd.push(best)
            self.after(0, self._finish_best, brd, best, info)
        except Exception as e:
            self.after(0, lambda: set_status(self.status, f"Engine error: {e}"))

    def _finish_best(self, brd, best, info):
        # No arrow; just repaint the board state after best (preview), but keep actual board unchanged
        # We'll still show cp (not %), removing the previous Win% display
        self.redraw(board=brd)
        cp = score_cp_signed(info["score"], self.my_color_is_white())
        set_status(self.status, f"Best preview: {self.board.san(best):6} | Eval: {fmt_cp(cp)} (board unchanged — Reset to clear preview)")

    # ---------- Pick move: engine cp + Explorer bonus ----------
    def apply_best_pick(self):
        if not self.last_infos:
            set_status(self.status, "No engine data.")
            self.pick_move = None
            return

        pov_is_white = self.my_color_is_white()

        # Gather candidates (engine MultiPV + quick probes)
        pv_moves: List[chess.Move] = []
        per_mv_cp: Dict[str, int] = {}
        for inf in self.last_infos:
            pv = inf.get("pv")
            if not pv: continue
            mv = pv[0]
            u = mv.uci()
            if u in per_mv_cp: continue
            per_mv_cp[u] = score_cp_signed(inf["score"], pov_is_white)
            pv_moves.append(mv)
        for u, sc in self.permove_cache.items():
            per_mv_cp.setdefault(u, score_cp_signed(sc, pov_is_white))
            try:
                mv = chess.Move.from_uci(u)
                if mv not in pv_moves: pv_moves.append(mv)
            except Exception:
                pass

        if not pv_moves:
            set_status(self.status, "No legal moves?")
            self.pick_move = None
            return

        explorer_w = float(self.explorer_weight.get())
        min_games = int(self.min_sample.get())
        ex_data = self._last_explorer if self.use_explorer.get() else None

        # Score = engine cp + explorer_weight * book_ev_cp  (book_ev_cp=0 if filtered)
        best_list: List[Tuple[float, chess.Move, dict]] = []
        for mv in pv_moves:
            u = mv.uci()
            base_cp = per_mv_cp.get(u, -999999)
            ex_bonus_cp, ex_info = explorer_bonus_for_move(u, ex_data, min_games) if ex_data else (0.0, None)
            if ex_info: ex_info = dict(ex_info); ex_info["u"] = u
            total = float(base_cp) + explorer_w * float(ex_bonus_cp)
            src_txt = "pv" if u in [m.get("pv")[0].uci() for m in self.last_infos if m.get("pv")] else ("probe" if u in self.permove_cache else "pv")
            best_list.append((total, mv, {"base_cp": base_cp, "ex": ex_info, "src": src_txt}))

        best_list.sort(key=lambda x: x[0], reverse=True)
        total, pick, meta = best_list[0]
        self.pick_move = pick  # retained, but NOT visualized by arrows

        # Status line (no arrows; keep cp textual)
        base_cp = int(meta["base_cp"])
        mate = None
        for inf in self.last_infos:
            pv = inf.get("pv")
            if pv and pv[0] == pick:
                mate = mate_in(inf["score"], pov_is_white)
                break

        ex_txt = ""
        if self.use_explorer.get():
            exp = meta.get("ex")
            if exp:
                filt = " (filtered)" if exp.get("filtered") else ""
                ex_txt = f" | BookEV:{int(exp.get('ev_cp',0))}cp G:{exp.get('games',0)} D:{exp.get('drawP',0):.0f}%{filt}"
            else:
                ex_txt = f" | Book:— (no data)"

        mtxt = self.board.san(pick)
        mate_txt = f" | M{mate}" if mate is not None else ""
        set_status(self.status, f"Move: {mtxt} | Src:{meta['src']} | Eval:{fmt_cp(base_cp)}{mate_txt}{ex_txt}")

# ---------- Main ----------
if __name__ == "__main__":
    App().mainloop()

