#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stockfish FEN analyser + Rating-Aware Style Coach + Lichess Explorer Assist
- Explorer 'Pool' selector (All / Classical+Rapid / Blitz+Bullet / Correspondence only)
- Robust Explorer requests (repeat 'speeds' params; retry without speeds on 400)
- Status bar shows BookEV / Games / Draw% or 'no data' with pool name
- Explorer status label shows how many moves were returned

Install:
    python -m pip install python-chess pillow cairosvg requests
    sudo apt install stockfish
"""

import io, os, math, shutil, random, threading
from typing import Optional, List, Tuple, Dict
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import chess, chess.engine, chess.svg, cairosvg
import requests

# ---------- Base Config ----------
DEPTH = 16
MULTIPV = 8
PER_MOVE_DEPTH = 7
MAX_PROBE_MOVES = 6
BOARD_SIZE = 360
SQUARE = BOARD_SIZE // 8
K_LOG = 0.004  # win% curve

ADHERENCE = {"Aggressive": (180, 12.0), "Normal": (45, 4.0), "Defensive": (25, 2.5)}  # (Δcp, Δwin%)
STYLE_CP_BONUS = {"Aggressive": 60, "Defensive": 40}
DIVERSITY_MARGIN_CP = 60

# ---------- Rating model ----------
ELO_ANCHORS = [
    (800,  (110.0, 110, 0.22, 5)),
    (1000, (120.0, 110, 0.25, 4)),
    (1200, (80.0,   70, 0.14, 4)),
    (1400, (62.0,   55, 0.11, 3)),
    (1600, (48.0,   45, 0.08, 3)),
    (1800, (38.0,   35, 0.06, 3)),
    (2000, (28.0,   28, 0.045,2)),
    (2200, (20.0,   22, 0.03, 2)),
    (2400, (14.0,   16, 0.02, 2)),
    (2600, (10.0,   10, 0.01, 2)),
]

# ---------- Lichess API ----------
LICHESS_EXPLORER = "https://explorer.lichess.ovh/lichess"
LICHESS_CLOUD_EVAL = "https://lichess.org/api/cloud-eval"

_http = requests.Session()
_http.headers.update({"User-Agent": "ICCF-Assistant/1.2 (+contact@example.com)"})
HTTP_TIMEOUT = 7.0

_explorer_cache: Dict[str, dict] = {}
_cloudeval_cache: Dict[str, dict] = {}

# ---------- ASCII status helpers ----------
ASCII_STATUS = True
_ASCII_MAP = str.maketrans({"—": "-", "–": "-", "Δ": "d", "≤": "<=", "≥": ">=", "≈": "~"})
def _ascii(s: str) -> str: return s.translate(_ASCII_MAP)
def set_status(widget, s: str): widget.config(text=_ascii(s) if ASCII_STATUS else s)

# ---------- Math helpers ----------
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
    if which: return which
    cand = "/usr/games/stockfish"
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    return None

def lerp(a, b, t): return a + (b - a) * t

def rating_profile(target_elo: int) -> Tuple[float, int, float, int]:
    xs = [e for e, _ in ELO_ANCHORS]
    target = max(min(target_elo, xs[-1]), xs[0])
    for i in range(len(xs) - 1):
        e0, p0 = ELO_ANCHORS[i]; e1, p1 = ELO_ANCHORS[i+1]
        if e0 <= target <= e1:
            t = (target - e0) / (e1 - e0) if e1 > e0 else 0.0
            temp = lerp(p0[0], p1[0], t)
            drop = int(round(lerp(p0[1], p1[1], t)))
            misp = lerp(p0[2], p1[2], t)
            topk = int(round(lerp(p0[3], p1[3], t)))
            return float(temp), int(drop), float(misp), max(2, int(topk))
    return 20.0, 20, 0.03, 2

def depth_profile(target_elo: int) -> Tuple[int, int, int]:
    if target_elo < 1000:  return (9, 6, 5)
    if target_elo < 1200:  return (10, 6, 6)
    if target_elo < 1500:  return (12, 8, 7)
    if target_elo < 1800:  return (14, 8, 7)
    return (16, 8, 7)

# ---------- Lichess Explorer helpers ----------
def _pool_values(pool_choice: str):
    if pool_choice == "All":
        return ["classical", "rapid", "blitz", "bullet", "correspondence"]
    if pool_choice == "Classical+Rapid":
        return ["classical", "rapid"]
    if pool_choice == "Blitz+Bullet":
        return ["blitz", "bullet"]
    if pool_choice == "Correspondence only":
        return ["correspondence"]
    return ["classical", "rapid", "blitz", "bullet", "correspondence"]

def lichess_explorer(fen: str, top_moves: int = 12, pool_choice: str = "All") -> Optional[dict]:
    """Explorer request with repeated 'speeds' params; retries without 'speeds' on HTTP 400."""
    cache_key = f"{fen}|{pool_choice}|{top_moves}"
    if cache_key in _explorer_cache:
        return _explorer_cache[cache_key]

    # Build as list of tuples to repeat 'speeds'
    params = [
        ("fen", fen),
        ("variant", "standard"),
        ("moves", str(top_moves)),
        ("topGames", "0"),
    ]
    for sp in _pool_values(pool_choice):
        params.append(("speeds", sp))

    try:
        r = _http.get(LICHESS_EXPLORER, params=params, timeout=HTTP_TIMEOUT)
        print(f"[Explorer] GET {r.url}")
        if r.ok:
            data = r.json()
            _explorer_cache[cache_key] = data
            print(f"[Explorer] ok: {len(data.get('moves', []))} moves")
            return data
        else:
            print(f"[Explorer] HTTP {r.status_code}. Retrying without 'speeds'…")
            r2 = _http.get(LICHESS_EXPLORER,
                           params={"fen": fen, "variant": "standard", "moves": top_moves, "topGames": 0},
                           timeout=HTTP_TIMEOUT)
            print(f"[Explorer] retry GET {r2.url}")
            if r2.ok:
                data = r2.json()
                _explorer_cache[cache_key] = data
                print(f"[Explorer] retry ok: {len(data.get('moves', []))} moves")
                return data
            else:
                print(f"[Explorer] retry HTTP {r2.status_code}: {r2.text[:200]}")
    except Exception as e:
        print(f"[Explorer] request failed: {type(e).__name__}: {e}")
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

def explorer_bonus_for_move(move_uci: str, ex_data: dict, min_sample: int) -> Tuple[float, dict]:
    if not ex_data or "moves" not in ex_data:
        return 0.0, {"games": 0, "winP": 0.0, "drawP": 0.0, "avgElo": 0}
    for m in ex_data["moves"]:
        if m.get("uci") == move_uci:
            games = int(m.get("gameCount", 0)); avg = int(m.get("averageRating", 0))
            if games < min_sample:
                return 0.0, {"games": games, "winP": 0.0, "drawP": 0.0, "avgElo": avg}
            w = float(m.get("white", 0)); d = float(m.get("draws", 0)); b = float(m.get("black", 0))
            total = max(1.0, w + d + b)
            winP = 100.0 * w / total; drawP = 100.0 * d / total
            ev = (winP - 50.0) - 0.35 * (drawP - 30.0)
            bonus = max(-60.0, min(60.0, ev))
            return bonus, {"games": games, "winP": winP, "drawP": drawP, "avgElo": avg}
    return 0.0, {"games": 0, "winP": 0.0, "drawP": 0.0, "avgElo": 0}

# ---------- App ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stockfish + Style Coach + Lichess Explorer Assist")
        self.geometry("1180x900"); self.resizable(True, True)

        # Fullscreen
        self.is_fullscreen = False; self.prev_geometry = None
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda e: self.exit_fullscreen())

        self.target_elo = 1000

        # FEN
        ttk.Label(self, text="Paste a FEN string:").pack(anchor="w", padx=10, pady=(10, 2))
        self.fen_var = tk.StringVar()
        fen_ent = ttk.Entry(self, textvariable=self.fen_var, width=160)
        fen_ent.pack(padx=10, fill="x"); fen_ent.bind("<Return>", lambda _ : self.load_fen())

        # Controls row
        ctl = ttk.Frame(self); ctl.pack(anchor="w", padx=10, pady=(6, 0))
        self.coach_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="Coach mode", variable=self.coach_on,
                        command=self.on_coach_toggle).pack(side="left", padx=(0, 12))
        ttk.Label(ctl, text="Style:").pack(side="left")
        self.style_var = tk.StringVar(value="Normal")
        self.style_cb = ttk.Combobox(ctl, textvariable=self.style_var,
                                     values=["Defensive", "Normal", "Aggressive"],
                                     state="readonly", width=12)
        self.style_cb.pack(side="left", padx=(4, 12))
        self.style_cb.bind("<<ComboboxSelected>>", lambda _e: self.update_style_preview())

        # Rating
        rating_fr = ttk.Frame(self); rating_fr.pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Label(rating_fr, text="My Elo:").pack(side="left")
        self.user_elo_var = tk.StringVar(value="800")
        elo_ent = ttk.Entry(rating_fr, textvariable=self.user_elo_var, width=8)
        elo_ent.pack(side="left", padx=(4, 8))
        elo_ent.bind("<Return>", lambda _e: self.on_elo_change())
        elo_ent.bind("<FocusOut>", lambda _e: self.on_elo_change())
        self.target_elo_lbl = ttk.Label(rating_fr, text="Coach plays at: 1000")
        self.target_elo_lbl.pack(side="left", padx=(8, 12))

        # Side / orientation
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

        # Lichess Assist controls
        assist_fr = ttk.LabelFrame(self, text="Lichess Assist"); assist_fr.pack(anchor="w", padx=10, pady=8, fill="x")
        self.use_explorer = tk.BooleanVar(value=True)
        self.use_cloud = tk.BooleanVar(value=False)
        ttk.Checkbutton(assist_fr, text="Use Explorer", variable=self.use_explorer,
                        command=lambda: self.async_eval()).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(assist_fr, text="Use Cloud Eval", variable=self.use_cloud,
                        command=lambda: self.async_eval()).pack(side="left", padx=(0, 18))
        ttk.Label(assist_fr, text="Explorer weight (cp):").pack(side="left")
        self.explorer_weight = tk.IntVar(value=40)
        ttk.Scale(assist_fr, from_=0, to=120, orient="horizontal",
                  variable=self.explorer_weight, command=lambda _:_).pack(side="left", padx=(6,16), fill="x", expand=True)
        ttk.Label(assist_fr, text="Min games:").pack(side="left")
        self.min_sample = tk.IntVar(value=100)
        ttk.Spinbox(assist_fr, from_=0, to=10000, width=6, textvariable=self.min_sample,
                    command=lambda: self.async_eval()).pack(side="left", padx=(6,8))
        ttk.Label(assist_fr, text="Pool:").pack(side="left", padx=(12, 4))
        self.pool_var = tk.StringVar(value="All")
        pool_cb = ttk.Combobox(assist_fr, textvariable=self.pool_var, state="readonly", width=18,
                               values=["All","Classical+Rapid","Blitz+Bullet","Correspondence only"])
        pool_cb.pack(side="left")
        pool_cb.bind("<<ComboboxSelected>>", lambda _e: self.async_eval())
        # Explorer status label
        self.explorer_status = tk.StringVar(value="Explorer: —")
        ttk.Label(assist_fr, textvariable=self.explorer_status).pack(side="left", padx=(12,0))

        # Buttons
        btn_fr = ttk.Frame(self); btn_fr.pack(pady=8)
        ttk.Button(btn_fr, text="Load FEN", command=self.load_fen).grid(row=0, column=0, padx=5)
        ttk.Button(btn_fr, text="Reset",    command=self.reset_pos).grid(row=0, column=1, padx=5)
        ttk.Button(btn_fr, text="Fullscreen", command=self.toggle_fullscreen).grid(row=0, column=2, padx=5)
        self.best_btn = ttk.Button(btn_fr, text="Best move", command=self.preview_best)
        self.best_btn.grid(row=0, column=3, padx=5)

        # Eval label + board
        self.eval_lbl = ttk.Label(self, text="", font=("Courier New", 12)); self.eval_lbl.pack(pady=6)
        self.canvas = tk.Canvas(self, width=BOARD_SIZE, height=BOARD_SIZE, highlightthickness=0)
        self.canvas.pack(); self.canvas.bind("<Button-1>", self.on_click)
        self._img = None; self._sel = None

        # Status bar
        self.status = ttk.Label(self, text="", relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

        # Engine
        path = find_stockfish_path()
        if not path:
            messagebox.showerror("Engine not found",
                                 "Could not find Stockfish.\nTry: sudo apt install stockfish\nor set STOCKFISH_PATH.")
            self.destroy(); return
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
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
        self.style_sugg_move: Optional[chess.Move] = None
        self.style_sugg_reason: str = ""
        self.last_eval_fen: Optional[str] = None
        self.eval_running = False; self.latest_request = 0
        self._last_explorer: Optional[dict] = None
        self._last_cloud: Optional[dict] = None

        # Kick off
        self.on_pick_color(); self.on_elo_change(); self.redraw(); self.async_eval()

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

    # ---------- Helpers ----------
    def my_color_is_white(self) -> bool:
        return self.play_color.get() == "white"

    def on_pick_color(self):
        self.follow_turn.set(False)
        self.fixed_bottom_is_white = (self.play_color.get() == "white")
        if self.last_eval_fen != self.board.fen(): self.async_eval()
        else: self.refresh_style_from_cache()

    def _invalidate_after_board_change(self):
        self.last_infos = None; self.permove_cache = {}; self.last_eval_fen = None
        self.style_sugg_move = None; self.style_sugg_reason = ""
        set_status(self.status, "Thinking...")

    def on_elo_change(self):
        try: user_elo = max(200, min(3000, int(self.user_elo_var.get())))
        except Exception:
            user_elo = 800; self.user_elo_var.set("800")
        self.target_elo = max(600, min(2600, user_elo + 200))
        self.target_elo_lbl.config(text=f"Coach plays at: {self.target_elo}")
        try:
            with self.engine_lock:
                self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(self.target_elo)})
        except Exception: pass
        if self.last_eval_fen == self.board.fen(): self.refresh_style_from_cache()
        else: self.async_eval()

    # ---------- Coach toggle ----------
    def on_coach_toggle(self):
        if self.coach_on.get():
            try: self.style_cb.config(state="readonly")
            except Exception: pass
            set_status(self.status, "Thinking...")
            if self.last_eval_fen == self.board.fen(): self.refresh_style_from_cache()
            else: self.async_eval()
        else:
            try: self.style_cb.config(state="disabled")
            except Exception: pass
            self.style_sugg_move = None; self.style_sugg_reason = ""
            set_status(self.status, "Coach off"); self.redraw()

    # ---------- Orientation ----------
    def bottom_color(self) -> chess.Color:
        return (self.board.turn if self.follow_turn.get()
                else (chess.WHITE if self.fixed_bottom_is_white else chess.BLACK))

    def flip_once(self):
        if self.follow_turn.get(): self.follow_turn.set(False)
        self.fixed_bottom_is_white = not self.fixed_bottom_is_white; self.redraw()

    # ---------- Drawing ----------
    def redraw(self, board: Optional[chess.Board] = None, highlight: Optional[chess.Move] = None):
        if board is None: board = self.board
        arrows = []
        if self.coach_on.get() and self.style_sugg_move:
            my_turn = self.board.turn == (chess.WHITE if self.my_color_is_white() else chess.BLACK)
            if my_turn:
                arrows.append(chess.svg.Arrow(self.style_sugg_move.from_square,
                                              self.style_sugg_move.to_square, color="#3b82f6"))
        if highlight:
            arrows.append(chess.svg.Arrow(highlight.from_square, highlight.to_square, color="#f59e0b"))
        svg = chess.svg.board(board=board, size=BOARD_SIZE, orientation=self.bottom_color(), arrows=arrows)
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
            pick_depth, mpv, probe_depth = depth_profile(self.target_elo)
            pick_depth = min(pick_depth, DEPTH); mpv = min(mpv, MULTIPV)
            info_list = self._safe_analyse(brd, chess.engine.Limit(depth=pick_depth), multipv=mpv)

            # unique PV heads
            unique_moves, seen = [], set()
            for inf in info_list:
                pv = inf.get("pv")
                if not pv: continue
                mv = pv[0]
                if mv not in seen: seen.add(mv); unique_moves.append(mv)

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

            # Lichess data (worker)
            fen = brd.fen()
            ex_data = lichess_explorer(fen, pool_choice=getattr(self, "pool_var", tk.StringVar(value="All")).get()) \
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

        # Update Explorer status label
        if self.use_explorer.get():
            if ex_data and isinstance(ex_data, dict):
                n = len(ex_data.get("moves", []))
                self.explorer_status.set(f"Explorer: {n} moves")
            else:
                self.explorer_status.set("Explorer: no data / error")
        else:
            self.explorer_status.set("Explorer: off")

        pov_is_white = self.my_color_is_white()
        cp = score_cp_signed(info_list[0]["score"], pov_is_white) if info_list else 0
        self.show_eval(cp)

        self.apply_rating_aware_pick(); self.redraw()

        self.eval_running = False
        if self.latest_request > gen: self.async_eval()

    def show_eval(self, cp_signed: int):
        my_prob = win_prob(cp_signed); opp_prob = 100.0 - my_prob
        self.eval_lbl.config(text=f"Eval: {fmt_cp(cp_signed)}  |  Win% Me:{my_prob:.0f} - Opp:{opp_prob:.0f}")

    # ---------- UI actions ----------
    def load_fen(self):
        fen = self.fen_var.get().strip()
        if not fen: messagebox.showinfo("No FEN", "Paste a FEN first."); return
        try: self.board = chess.Board(fen)
        except ValueError as e: messagebox.showerror("Bad FEN", str(e)); return
        self.base_fen = self.board.fen(); self._sel = None
        self._invalidate_after_board_change(); self.redraw(); self.async_eval()

    def reset_pos(self):
        self.board = chess.Board(self.base_fen); self._sel = None
        self._invalidate_after_board_change(); self.redraw(); self.async_eval()

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
                self.board.push(mv); self._invalidate_after_board_change(); self.redraw(); self.async_eval()
            else:
                messagebox.showwarning("Illegal", "That move isn't legal.")

    # Best move preview
    def preview_best(self):
        set_status(self.status, "Thinking...")
        threading.Thread(target=self._worker_best, daemon=True).start()

    def _worker_best(self):
        try:
            pick_depth, _, _ = depth_profile(self.target_elo)
            info = self._safe_analyse(self.board, chess.engine.Limit(depth=pick_depth))
            best = info["pv"][0]; brd = self.board.copy(); brd.push(best)
            self.after(0, self._finish_best, brd, best, info)
        except Exception as e:
            self.after(0, lambda: set_status(self.status, f"Engine error: {e}"))

    def _finish_best(self, brd, best, info):
        self.redraw(board=brd, highlight=best)
        cp = score_cp_signed(info["score"], self.my_color_is_white())
        my_prob = win_prob(cp); opp_prob = 100 - my_prob
        self.eval_lbl.config(text=f"Best: {self.board.san(best):6} | {fmt_cp(cp)} | Win% Me:{my_prob:.0f} - Opp:{opp_prob:.0f}")
        set_status(self.status, "Preview - actual board unchanged (Reset to clear)")

    # ---------- Style / features ----------
    def _style_features(self, board: chess.Board, move: chess.Move) -> dict:
        b, mv = board, move
        moved_piece = b.piece_at(mv.from_square); to_piece = b.piece_at(mv.to_square)
        is_capture = (to_piece is not None) or b.is_en_passant(mv); is_promo = mv.promotion is not None
        gives_check = b.gives_check(mv)
        def cw(sq): f=chess.square_file(sq); r=chess.square_rank(sq); return -((f-3.5)**2 + (r-3.5)**2)
        center_gain = (cw(mv.to_square)-cw(mv.from_square)) if moved_piece else 0.0
        dev_new_piece = 1 if (moved_piece and moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.square_rank(mv.from_square) in (0,7)) else 0
        pawn_thrust = abs(chess.square_rank(mv.to_square)-chess.square_rank(mv.from_square)) if (moved_piece and moved_piece.piece_type==chess.PAWN and not is_capture) else 0
        castle = b.is_castling(mv); quiet_dev = int((not is_capture) and moved_piece and moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP))
        king_shelter_pawn = 1 if (moved_piece and moved_piece.piece_type==chess.PAWN and not is_capture and chess.square_file(mv.from_square) in (0,7)) else 0
        simplify_capture = int(is_capture and moved_piece and to_piece and moved_piece.piece_type == to_piece.piece_type)
        return {"is_check":int(gives_check),"is_capture_or_promo":int(is_capture or is_promo),"center_gain":center_gain,
                "dev_new_piece":dev_new_piece,"pawn_thrust":pawn_thrust,"castle":int(castle),"quiet_dev":quiet_dev,
                "king_shelter_pawn":king_shelter_pawn,"simplify_capture":simplify_capture}

    def _reason_bits(self, style: str, f: dict) -> List[str]:
        bits: List[str] = []
        if style == "Aggressive":
            if f["is_check"]: bits.append("give check")
            if f["is_capture_or_promo"]: bits.append("win material/promote")
            if f["center_gain"] > 0.0: bits.append("take center space")
            if f["dev_new_piece"]: bits.append("develop new piece")
            if f["pawn_thrust"] >= 2: bits.append("two-square pawn thrust")
            elif f["pawn_thrust"] == 1: bits.append("pawn thrust")
        elif style == "Defensive":
            if f["castle"]: bits.append("castle for king safety")
            if f["quiet_dev"]: bits.append("quiet development")
            if f["king_shelter_pawn"]: bits.append("improve king shelter")
            if f["simplify_capture"]: bits.append("simplify by trade")
            if f["is_check"] == 0 and f["is_capture_or_promo"] == 0: bits.append("avoid forcing moves")
        else:
            bits.append("best engine move")
        return bits

    def _style_metric(self, style: str, f: dict) -> float:
        if style == "Aggressive":
            return (3.0*f["is_check"] + 2.0*f["is_capture_or_promo"] + 1.2*max(0.0,f["center_gain"])
                    + 1.0*f["dev_new_piece"] + (2.0 if f["pawn_thrust"]>=2 else (0.8 if f["pawn_thrust"]==1 else 0.0)))
        if style == "Defensive":
            return (2.5*f["castle"] + 2.0*f["quiet_dev"] + 1.5*f["king_shelter_pawn"]
                    + 1.0*f["simplify_capture"] + (0.8 if (f["is_check"]==0 and f["is_capture_or_promo"]==0) else 0.0))
        return 0.0

    # ---------- Picking + commentary (Explorer/Cloud blend) ----------
    def update_style_preview(self):
        if self.last_eval_fen != self.board.fen(): self.async_eval()
        else: self.refresh_style_from_cache()

    def refresh_style_from_cache(self):
        self.apply_rating_aware_pick(); self.redraw()

    def apply_rating_aware_pick(self):
        if not self.coach_on.get():
            self.style_sugg_move = None; self.style_sugg_reason = ""; set_status(self.status, ""); return
        mv, reason, meta = self.pick_rating_style_move()
        self.style_sugg_move = mv; self.style_sugg_reason = reason

        my_turn = self.board.turn == (chess.WHITE if self.my_color_is_white() else chess.BLACK)
        if mv and my_turn:
            try: san = self.board.san(mv)
            except Exception: san = mv.uci()
            feats = self._style_features(self.board, mv)
            idea = ", ".join(self._reason_bits(self.style_var.get(), feats)[:3]) or "on plan"
            top_cp = meta.get("top_cp", 0.0); pick_cp = meta.get("pick_cp", top_cp)
            dcp = int(top_cp - pick_cp); dwp = win_prob(top_cp) - win_prob(pick_cp); src = meta.get("source", "pv")
            ex_txt = ""
            if self.use_explorer.get():
                exp = meta.get("explorer")
                if exp:
                    ex_txt = f" | BookEV:{int(exp.get('ev_cp',0))}cp G:{exp.get('games',0)} D:{exp.get('drawP',0):.0f}%"
                else:
                    ex_txt = f" | Book:— (no data, pool={self.pool_var.get()})"
            msg = (f"Move: {san} | Idea: {idea} | Style:{self.style_var.get()} | CoachElo:{self.target_elo} "
                   f"| Safety d{dcp}cp d{dwp:.1f}% | Src:{src}{ex_txt}")
            set_status(self.status, msg)
        else:
            set_status(self.status, "Waiting for opponent...")

    def pick_rating_style_move(self) -> Tuple[Optional[chess.Move], str, Dict[str, float]]:
        style = self.style_var.get(); pov_is_white = self.my_color_is_white()

        cands: List[Tuple[chess.Move, float, dict, List[str], Optional[int], float, str]] = []
        seen = set()
        if self.last_infos:
            for inf in self.last_infos:
                pv = inf.get("pv"); 
                if not pv: continue
                mv = pv[0]
                if mv in seen: continue
                seen.add(mv)
                cp = score_cp_signed(inf["score"], pov_is_white)
                feats = self._style_features(self.board, mv)
                reasons = self._reason_bits(style, feats)
                metric = self._style_metric(style, feats)
                m_in = mate_in(inf["score"], pov_is_white)
                cands.append((mv, float(cp), feats, reasons, m_in, metric, "pv"))
        for uci, sc in self.permove_cache.items():
            mv = chess.Move.from_uci(uci)
            if mv in seen: continue
            cp = score_cp_signed(sc, pov_is_white)
            feats = self._style_features(self.board, mv)
            reasons = self._reason_bits(style, feats)
            metric = self._style_metric(style, feats)
            m_in = mate_in(sc, pov_is_white)
            cands.append((mv, float(cp), feats, reasons, m_in, metric, "probe"))
        if not cands: return None, "", {}

        mates = [t for t in cands if t[4] is not None and t[4] > 0]
        if mates:
            best = min(mates, key=lambda t: t[4])
            meta = {"top_cp": best[1], "pick_cp": best[1], "source": "mate"}
            return best[0], f"finish tactic: mate in {best[4]}", meta

        top_cp = max(t[1] for t in cands); top_mv = max(cands, key=lambda t: t[1])[0]
        style_bonus = STYLE_CP_BONUS.get(style, 0)
        temp_cp, max_drop_cp, mistake_prob, top_k_floor = rating_profile(self.target_elo)
        if top_cp >= 200: max_drop_cp = int(max_drop_cp*1.15); temp_cp *= 1.10
        elif top_cp <= -80: max_drop_cp = int(max_drop_cp*1.35); temp_cp *= 1.15
        sorted_cands = sorted(cands, key=lambda t: t[1], reverse=True)
        poolA = [t for t in sorted_cands if (top_cp - t[1]) <= max_drop_cp] or sorted_cands[:max(top_k_floor, len(sorted_cands))]

        # Explorer & Cloud bonuses
        ex_data = self._last_explorer if self.use_explorer.get() else None
        exp_weight = float(self.explorer_weight.get()); min_games = int(self.min_sample.get())
        ex_bonus_map: Dict[str, Tuple[float, dict]] = {}
        if ex_data:
            for mv, *_ in cands:
                b, info = explorer_bonus_for_move(mv.uci(), ex_data, min_games)
                b = max(-exp_weight, min(exp_weight, b))
                ex_bonus_map[mv.uci()] = (b, info)
        cloud = self._last_cloud if self.use_cloud.get() else None
        cloud_bonus_map: Dict[str, float] = {}
        if cloud and "pvs" in cloud:
            for pv in cloud.get("pvs", []):
                if not pv.get("moves"): continue
                first_uci = pv["moves"].split()[0]; cpv = pv.get("cp", 0)
                cloud_bonus_map[first_uci] = max(-20.0, min(20.0, float(cpv)/5.0))

        def weights(pool):
            w = []
            for mv, cp, feats, _reasons, _m_in, metric, _src in pool:
                eff = cp + style_bonus*metric
                if ex_data and mv.uci() in ex_bonus_map: eff += ex_bonus_map[mv.uci()][0]
                if cloud and mv.uci() in cloud_bonus_map: eff += cloud_bonus_map[mv.uci()]
                w.append(math.exp((eff - top_cp) / max(1e-9, temp_cp)))
            s = sum(w) or 1.0
            return [x / s for x in w]

        use_mistake = (random.random() < mistake_prob)
        if use_mistake and len(sorted_cands) > len(poolA):
            widen = int(max_drop_cp * 1.8)
            poolB = [t for t in sorted_cands if (top_cp - t[1]) <= widen] or sorted_cands[:max(top_k_floor+1, 3)]
            old_temp = temp_cp; temp_cp *= 1.4
            probs = weights(poolB); pick = random.choices(poolB, probs, k=1)[0]
            temp_cp = old_temp
        else:
            probs = weights(poolA); pick = random.choices(poolA, probs, k=1)[0]

        pick_mv, pick_cp, _, pick_reasons, _, pick_metric, source = pick

        # Safety clamp
        cap_cp, cap_wp = ADHERENCE.get(style, ADHERENCE["Normal"])
        drop_cp = top_cp - pick_cp; drop_wp = win_prob(top_cp) - win_prob(pick_cp)
        if drop_cp > cap_cp or drop_wp > cap_wp:
            meta = {"top_cp": top_cp, "pick_cp": top_cp, "source": "safe-top"}
            if ex_data and top_mv.uci() in ex_bonus_map:
                b, info = ex_bonus_map[top_mv.uci()]; meta["explorer"] = {"ev_cp": b, **info}
            return top_mv, f"safer choice (saved {int(drop_cp)}cp, {drop_wp:.1f}% win prob)", meta

        # Diversity nudge
        alts = [t for t in cands if t[0] != pick_mv and (top_cp - t[1]) <= DIVERSITY_MARGIN_CP]
        if alts:
            alt = max(alts, key=lambda t: t[5])
            if alt[5] > pick_metric + 1.0:
                pick_mv, pick_reasons, source = alt[0], alt[3], alt[6]
                for m, cpv, *_rest in cands:
                    if m == pick_mv: pick_cp = cpv; break

        friendly = ", ".join(pick_reasons[:3]) if pick_reasons else "on plan"
        meta = {"top_cp": top_cp, "pick_cp": pick_cp, "source": source}
        if ex_data and pick_mv.uci() in ex_bonus_map:
            b, info = ex_bonus_map[pick_mv.uci()]; meta["explorer"] = {"ev_cp": b, **info}
        return pick_mv, f"{friendly} (rating {self.target_elo}, cap {int(max_drop_cp)}cp, {source})", meta

    # ---------- Cleanup ----------
    def destroy(self):
        try:
            if hasattr(self, "engine"): self.engine.close()
        finally:
            super().destroy()

if __name__ == "__main__":
    App().mainloop()
