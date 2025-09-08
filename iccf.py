#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stockfish FEN analyser + Rating-Aware Style Coach + Lichess Explorer Assist (patched)

Patches in this build:
1) Always show real Explorer counts; if count < Min games, bonus=0 but "(filtered)" is shown.
2) Pull 50 top moves from Explorer in the worker, making it very likely the picked move appears.

Extra:
- Explorer status label ("Explorer: N moves").
- Debug print when the picked move isn’t in the Explorer top list.
- ASCII-safe status bar (dashes instead of en-dash, etc.).

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

# Style guardrails and bonuses (Δcp safety, Δwin% safety)
ADHERENCE = {"Aggressive": (180, 12.0), "Normal": (45, 4.0), "Defensive": (25, 2.5)}
STYLE_CP_BONUS = {"Aggressive": 60, "Defensive": 40}
DIVERSITY_MARGIN_CP = 60  # min difference from best to justify variety

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
_http.headers.update({"User-Agent": "ICCF-Assistant/1.4 (+contact@example.com)"})
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

# ---------- Lichess helpers ----------
def lichess_explorer(fen: str, top_moves: int = 50) -> Optional[dict]:
    """Simplest & compatible Explorer request (no 'speeds' filter)."""
    cache_key = f"{fen}|{top_moves}"
    if cache_key in _explorer_cache:
        return _explorer_cache[cache_key]
    params = {"fen": fen, "variant": "standard", "moves": max(1, int(top_moves)), "topGames": 0}
    try:
        r = _http.get(LICHESS_EXPLORER, params=params, timeout=HTTP_TIMEOUT)
        print(f"[Explorer] GET {r.url}")
        if r.ok:
            data = r.json()
            _explorer_cache[cache_key] = data
            print(f"[Explorer] ok: {len(data.get('moves', []))} moves")
            return data
        else:
            print(f"[Explorer] HTTP {r.status_code}: {r.text[:200]}")
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

# ---------- PATCH 1: Always show real counts; filter only affects bonus ----------
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
        offered = [m.get("uci") for m in ex_data["moves"]]
        print(f"[Explorer] picked {move_uci} not in top list; offered(first 15)={offered[:15]}")
        return 0.0, info

    w = float(found.get("white", 0))
    d = float(found.get("draws", 0))
    b = float(found.get("black", 0))
    tot = w + d + b
    # some endpoints omit gameCount; infer from w+d+b when absent/zero
    games = int(found.get("gameCount") or tot)
    # average rating may be per-node not per-move on some responses
    avg = int(found.get("averageRating") or ex_data.get("averageRating", 0))

    # defensively avoid divide-by-zero
    denom = max(1.0, tot)
    winP  = 100.0 * w / denom
    drawP = 100.0 * d / denom

    # cp-like EV (prefer wins, gently penalize drawish lines)
    ev_cp = (winP - 50.0) - 0.35 * (drawP - 30.0)
    ev_cp = max(-60.0, min(60.0, ev_cp))

    info.update({"games": games, "winP": winP, "drawP": drawP, "avgElo": avg, "ev_cp": float(ev_cp)})
    if games < min_sample:
        info["filtered"] = True
        return 0.0, info
    return float(ev_cp), info


    # cp-like EV (softly prefers higher win%, penalizes very drawish)
    ev_cp = (winP - 50.0) - 0.35 * (drawP - 30.0)
    ev_cp = max(-60.0, min(60.0, ev_cp))

    info.update({"games": games, "winP": winP, "drawP": drawP, "avgElo": avg, "ev_cp": float(ev_cp)})
    if games < min_sample:
        info["filtered"] = True
        return 0.0, info          # show counts, but no bias
    return float(ev_cp), info

# ---------- App ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stockfish + Style Coach + Lichess Explorer Assist")
        self.geometry("1180x900"); self.resizable(True, True)

        # Fullscreen bindings
        self.is_fullscreen = False; self.prev_geometry = None
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.bind("<Escape>", lambda e: self.exit_fullscreen())

        self.target_elo = 1000

        # FEN input
        ttk.Label(self, text="Paste a FEN string:").pack(anchor="w", padx=10, pady=(10, 2))
        self.fen_var = tk.StringVar()
        fen_ent = ttk.Entry(self, textvariable=self.fen_var, width=160)
        fen_ent.pack(padx=10, fill="x")
        fen_ent.bind("<Return>", lambda _ : self.load_fen())

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

        # Rating row
        rating_fr = ttk.Frame(self); rating_fr.pack(anchor="w", padx=10, pady=(6, 0))
        ttk.Label(rating_fr, text="My Elo:").pack(side="left")
        self.user_elo_var = tk.StringVar(value="800")
        elo_ent = ttk.Entry(rating_fr, textvariable=self.user_elo_var, width=8)
        elo_ent.pack(side="left", padx=(4, 8))
        elo_ent.bind("<Return>", lambda _e: self.on_elo_change())
        elo_ent.bind("<FocusOut>", lambda _e: self.on_elo_change())
        self.target_elo_lbl = ttk.Label(rating_fr, text="Coach plays at: 1000")
        self.target_elo_lbl.pack(side="left", padx=(8, 12))

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
        self.min_sample = tk.IntVar(value=30)  # set to 5 for testing in UI if you want
        ttk.Spinbox(assist_fr, from_=0, to=10000, width=6, textvariable=self.min_sample,
                    command=lambda: self.async_eval()).pack(side="left", padx=(6,8))
        ttk.Label(assist_fr, text="Top moves:").pack(side="left", padx=(12,4))
        self.top_moves = tk.IntVar(value=50)  # UI control retained; worker uses 50 as per patch 2
        ttk.Spinbox(assist_fr, from_=5, to=100, width=5, textvariable=self.top_moves,
                    command=lambda: self.async_eval()).pack(side="left", padx=(0,8))
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

            # Lichess data (PATCH 2: pull 50 top moves)
            fen = brd.fen()
            ex_data = lichess_explorer(fen, top_moves=50) if self.use_explorer.get() else None
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

    # ---------- Best move preview ----------
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
        is_capture = (to_piece is not None) or b.is_en_passant(mv)
        is_promo = mv.promotion is not None
        gives_check = b.gives_check(mv)
        def cw(sq): f=chess.square_file(sq); r=chess.square_rank(sq); return -((f-3.5)**2 + (r-3.5)**2)
        center_gain = (cw(mv.to_square)-cw(mv.from_square)) if moved_piece else 0.0
        dev_new_piece = 1 if (moved_piece and moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP)
                              and chess.square_rank(mv.from_square) in (0,7)) else 0
        pawn_thrust = abs(chess.square_rank(mv.to_square)-chess.square_rank(mv.from_square)) \
                      if (moved_piece and moved_piece.piece_type==chess.PAWN and not is_capture) else 0
        castle = b.is_castling(mv)
        quiet_dev = int((not is_capture) and moved_piece and moved_piece.piece_type in (chess.KNIGHT, chess.BISHOP))
        king_shelter_pawn = 1 if (moved_piece and moved_piece.piece_type==chess.PAWN and not is_capture
                                  and chess.square_file(mv.from_square) in (0,7)) else 0
        simplify_capture = int(is_capture and moved_piece and to_piece and moved_piece.piece_type == to_piece.piece_type)
        return {
            "is_check": int(gives_check),
            "is_capture_or_promo": int(is_capture or is_promo),
            "center_gain": float(center_gain),
            "pawn_thrust": float(pawn_thrust),
            "castle": int(castle),
            "quiet_dev": int(quiet_dev),
            "king_shelter_pawn": int(king_shelter_pawn),
            "simplify_capture": int(simplify_capture),
        }

    def _style_score(self, feats: dict, style: str) -> float:
        # light, interpretable weights
        if style == "Aggressive":
            return (2.5*feats["is_check"] + 1.7*feats["is_capture_or_promo"] +
                    0.8*feats["center_gain"] + 0.9*feats["pawn_thrust"] + 0.7*feats["quiet_dev"])
        if style == "Defensive":
            return (1.4*feats["quiet_dev"] + 1.6*feats["king_shelter_pawn"] +
                    1.2*feats["simplify_capture"] + 0.6*feats["center_gain"])
        return (1.0*feats["quiet_dev"] + 1.0*feats["center_gain"] +
                1.0*feats["pawn_thrust"] + 1.0*feats["is_capture_or_promo"])

    def refresh_style_from_cache(self):
        # Rebuild suggestion text from cached info
        if not self.last_infos:
            set_status(self.status, "Ready.")
            return
        self.apply_rating_aware_pick()

    def update_style_preview(self):
        if self.coach_on.get():
            self.apply_rating_aware_pick()

    # Core: pick move rating-aware, mix engine + explorer + style
    def apply_rating_aware_pick(self):
        if not self.coach_on.get() or not self.last_infos:
            set_status(self.status, "Coach off" if not self.coach_on.get() else "No engine data.")
            self.style_sugg_move = None
            self.redraw()
            return

        pov_is_white = self.my_color_is_white()
        style = self.style_var.get()
        safety_cp, safety_win = ADHERENCE.get(style, ADHERENCE["Normal"])

        # Base best score from engine
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

        # Include probed moves
        for u, sc in self.permove_cache.items():
            per_mv_cp.setdefault(u, score_cp_signed(sc, pov_is_white))
            try:
                mv = chess.Move.from_uci(u)
                if mv not in pv_moves: pv_moves.append(mv)
            except Exception:
                pass

        if not pv_moves:
            set_status(self.status, "No legal moves?")
            self.style_sugg_move = None
            return

        # Explorer influence (bonus in cp units, then scaled by weight)
        explorer_w = float(self.explorer_weight.get())
        min_games = int(self.min_sample.get())
        ex_data = self._last_explorer if self.use_explorer.get() else None

        # Score each candidate
        best_cp = max(per_mv_cp.values()) if per_mv_cp else 0
        cand: List[Tuple[float, chess.Move, dict]] = []
        for mv in pv_moves:
            u = mv.uci()
            base_cp = per_mv_cp.get(u, -99999)

            # Style nudges
            feats = self._style_features(self.board, mv)
            style_pts = self._style_score(feats, style)
            style_cp_bonus = STYLE_CP_BONUS.get(style, 0) * (style_pts / (style_pts + 5.0) if style_pts > 0 else 0.0)

            # Explorer bonus (PATCH 1 behavior)
            ex_bonus_cp, ex_info = explorer_bonus_for_move(u, ex_data, min_games) if ex_data else (0.0, None)
            if ex_info:
                ex_info = dict(ex_info)  # ensure copy
                ex_info["u"] = u
            # Apply scaled Explorer contribution (ex_bonus_cp can be 0 when filtered)
            total = float(base_cp) + style_cp_bonus + explorer_w * float(ex_bonus_cp)

            # Soft safety: don’t go too far below best
            if best_cp - base_cp > safety_cp:
                total -= (best_cp - base_cp - safety_cp) * 0.75

            cand.append((total, mv, {"style_pts": style_pts, "ex": ex_info, "base_cp": base_cp,
                                     "style_cp_bonus": style_cp_bonus}))

        cand.sort(key=lambda x: x[0], reverse=True)
        pick_score, pick_move, meta = cand[0]
        self.style_sugg_move = pick_move

        # Build status line
        base_cp = int(meta["base_cp"])
        style_cp_bonus = int(round(meta["style_cp_bonus"]))
        mate = None
        for inf in self.last_infos:
            pv = inf.get("pv")
            if pv and pv[0] == pick_move:
                mate = mate_in(inf["score"], pov_is_white)
                break
        src_txt = "pv"
        if pick_move.uci() in self.permove_cache: src_txt = "probe"

        # Explorer text per your spec
        ex_txt = ""
        if self.use_explorer.get():
            exp = meta.get("ex")
            if exp:
                filt = " (filtered)" if exp.get("filtered") else ""
                ex_txt = f" | BookEV:{int(exp.get('ev_cp',0))}cp G:{exp.get('games',0)} D:{exp.get('drawP',0):.0f}%{filt}"
            else:
                ex_txt = f" | Book:— (no data, pool=All)"

        mtxt = self.board.san(pick_move)
        idea = []
        if meta["style_pts"] > 0.5: idea.append(style)
        if meta["style_cp_bonus"] > 0: idea.append(f"+{style_cp_bonus}cp style")
        idea_txt = ", ".join(idea) if idea else "—"
        mate_txt = f" | M{mate}" if mate is not None else ""
        set_status(self.status,
                   f"Move: {mtxt} | Src:{src_txt} | Eval:{fmt_cp(base_cp)}{mate_txt} | Idea:{idea_txt}{ex_txt}")

    # ---------- END style / features ----------

    def toggle_follow(self, *_):
        self.follow_turn.set(not self.follow_turn.get())
        self.redraw()

# ---------- Main ----------
if __name__ == "__main__":
    App().mainloop()
