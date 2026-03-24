"""
Terminal Audio Visualizer + System Monitor — Hacker Edition
3-Panel Layout: [Sound Bars] [CPU/RAM/GPU/Claude] [Network]
Auto-resizes with terminal window.

Usage:
  soundbar              # start visualizer (fullscreen)
  soundbar --stop       # stop background visualizer
  soundbar --gain 40    # audio gain
  soundbar --no-stats   # audio only

Requirements:
  pip install PyAudioWPatch numpy psutil GPUtil
"""

import sys
import os
import time
import signal
import argparse
import threading
import re
import json
import numpy as np
from collections import deque
from pathlib import Path

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("\033[38;2;255;60;60m[!]\033[0m pip install PyAudioWPatch numpy")
    sys.exit(1)

try:
    import psutil
except ImportError:
    psutil = None

try:
    import GPUtil
except ImportError:
    GPUtil = None

# ─── Defaults ────────────────────────────────────────────────────
DEFAULT_GAIN = 40.0
SMOOTHING = 0.55
DECAY = 0.88
BLOCK_SIZE = 2048
FPS = 20
STATS_FPS = 2
PID_FILE = os.path.join(os.environ.get("TEMP", "/tmp"), "soundbar.pid")
CLAUDE_DIR = Path.home() / ".claude"

# ─── ANSI ────────────────────────────────────────────────────────
ESC = "\033"
HIDE_CUR = f"{ESC}[?25l"
SHOW_CUR = f"{ESC}[?25h"
RST = f"{ESC}[0m"
BOLD = f"{ESC}[1m"

C_DIM = f"{ESC}[38;2;0;40;0m"
C_DIM2 = f"{ESC}[38;2;0;60;0m"
C_MID = f"{ESC}[38;2;0;100;0m"
C_GREEN = f"{ESC}[38;2;0;180;30m"
C_BRIGHT = f"{ESC}[38;2;0;255;65m"
C_CYAN = f"{ESC}[38;2;0;200;230m"
C_YELLOW = f"{ESC}[38;2;200;200;0m"
C_RED = f"{ESC}[38;2;255;60;60m"
C_ORANGE = f"{ESC}[38;2;255;140;0m"
C_WHITE = f"{ESC}[38;2;200;240;200m"
C_BDR = f"{ESC}[38;2;0;70;0m"
C_HDR = f"{ESC}[38;2;0;255;65m"
C_LBL = f"{ESC}[38;2;0;140;40m"
C_VAL = f"{ESC}[38;2;0;220;60m"
C_PURPLE = f"{ESC}[38;2;160;80;255m"
C_PURPLE_D = f"{ESC}[38;2;80;40;130m"

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def vlen(s):
    return len(_ANSI_RE.sub('', s))


def vpad(s, w):
    d = w - vlen(s)
    return s + ' ' * d if d > 0 else s


def vtrunc(s, w):
    """Truncate string to visual width w."""
    out = []
    vis = 0
    i = 0
    raw = s
    while i < len(raw) and vis < w:
        if raw[i] == '\033':
            j = raw.find('m', i)
            if j != -1:
                out.append(raw[i:j+1])
                i = j + 1
                continue
        out.append(raw[i])
        vis += 1
        i += 1
    return ''.join(out) + RST


def mv(r, c=1):
    return f"{ESC}[{r};{c}H"


def clr():
    return f"{ESC}[2K"


def term_size():
    try:
        c, r = os.get_terminal_size()
        return r, c
    except Exception:
        return 30, 120


# ─── Gradient ────────────────────────────────────────────────────
def make_grad(h):
    colors = []
    for i in range(h):
        t = i / max(h - 1, 1)
        if t < 0.3:
            s = t / 0.3
            r, g, b = 0, int(20 + 60 * s), 0
        elif t < 0.6:
            s = (t - 0.3) / 0.3
            r, g, b = 0, int(80 + 110 * s), int(5 * s)
        elif t < 0.85:
            s = (t - 0.6) / 0.25
            r, g, b = int(10 * s), int(190 + 55 * s), int(5 + 30 * s)
        else:
            s = (t - 0.85) / 0.15
            r, g, b = int(10 + 90 * s), 255, int(35 + 100 * s)
        colors.append(f"{ESC}[38;2;{r};{g};{b}m")
    return colors


# ─── Audio ───────────────────────────────────────────────────────
audio_buffer = np.zeros(BLOCK_SIZE)
buf_lock = threading.Lock()


def find_loopback(p, idx=None):
    if idx is not None:
        try:
            d = p.get_device_info_by_index(idx)
            if d.get("isLoopbackDevice"):
                return d
        except Exception:
            pass
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        defout = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        base = defout["name"].split("(")[0].strip()
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d.get("isLoopbackDevice") and base in d["name"]:
                return d
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d.get("isLoopbackDevice"):
                return d
    except Exception:
        pass
    return None


def audio_cb(in_data, frame_count, time_info, status):
    global audio_buffer
    data = np.frombuffer(in_data, dtype=np.float32)
    ch = max(1, len(data) // frame_count)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    with buf_lock:
        audio_buffer = data.copy()
    return (in_data, pyaudio.paContinue)


def spectrum(n_bars, max_h, gain, sm, pk):
    with buf_lock:
        data = audio_buffer.copy()
    w = np.hanning(len(data))
    fft = np.abs(np.fft.rfft(data * w))
    fft = fft[1:len(fft) // 2]
    if len(fft) == 0:
        return sm, pk
    n = len(fft)
    bands = np.zeros(n_bars)
    for i in range(n_bars):
        lo = int(n ** (i / n_bars))
        hi = int(n ** ((i + 1) / n_bars))
        lo, hi = max(0, min(lo, n - 1)), max(lo + 1, min(hi, n))
        bands[i] = np.mean(fft[lo:hi]) if hi > lo else 0
    bands = np.log1p(bands * gain) * (max_h / 4)
    bands = np.clip(bands, 0, max_h)
    sm[:] = sm * SMOOTHING + bands * (1 - SMOOTHING)
    pk[:] = np.maximum(sm, pk * DECAY)
    return sm, pk


# ─── Claude Usage ────────────────────────────────────────────────
class ClaudeUsage:
    def __init__(self):
        self.model = ""
        self.s_msgs = 0
        self.s_in = 0
        self.s_out = 0
        self.s_cache_r = 0
        self.s_cache_c = 0
        self.tot_sess = 0
        self.tot_msgs = 0
        self.all_tokens = {}
        self._lock = threading.Lock()
        self._running = True
        self._update()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            time.sleep(3)
            if self._running:
                self._update()

    def _update(self):
        try:
            sf = CLAUDE_DIR / "stats-cache.json"
            if sf.exists():
                d = json.loads(sf.read_text(encoding='utf-8'))
                with self._lock:
                    self.tot_sess = d.get('totalSessions', 0)
                    self.tot_msgs = d.get('totalMessages', 0)
                    self.all_tokens = {}
                    for m, info in d.get('modelUsage', {}).items():
                        self.all_tokens[m] = {
                            'in': info.get('inputTokens', 0),
                            'out': info.get('outputTokens', 0),
                            'cr': info.get('cacheReadInputTokens', 0),
                            'cc': info.get('cacheCreationInputTokens', 0),
                        }
        except Exception:
            pass
        try:
            pd = CLAUDE_DIR / "projects"
            if not pd.exists():
                return
            cands = []
            for p in pd.iterdir():
                if p.is_dir():
                    for f in p.glob("*.jsonl"):
                        if "subagent" not in str(f):
                            cands.append(f)
            if not cands:
                return
            latest = max(cands, key=lambda f: f.stat().st_mtime)
            ti = to = tcr = tcc = cnt = 0
            mdl = ""
            with open(latest, encoding='utf-8') as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        msg = d.get('message', {})
                        if isinstance(msg, dict):
                            u = msg.get('usage', {})
                            if u:
                                ti += u.get('input_tokens', 0)
                                to += u.get('output_tokens', 0)
                                tcr += u.get('cache_read_input_tokens', 0)
                                tcc += u.get('cache_creation_input_tokens', 0)
                                cnt += 1
                                mdl = msg.get('model', mdl) or mdl
                    except Exception:
                        continue
            with self._lock:
                self.s_in, self.s_out = ti, to
                self.s_cache_r, self.s_cache_c = tcr, tcc
                self.s_msgs, self.model = cnt, mdl
        except Exception:
            pass

    def get(self):
        with self._lock:
            return {
                'model': self.model,
                'msgs': self.s_msgs,
                'in': self.s_in, 'out': self.s_out,
                'cache_r': self.s_cache_r, 'cache_c': self.s_cache_c,
                'total': self.s_in + self.s_out + self.s_cache_r + self.s_cache_c,
                'tot_sess': self.tot_sess, 'tot_msgs': self.tot_msgs,
                'all': dict(self.all_tokens),
            }


# ─── System Stats ────────────────────────────────────────────────
class SysStats:
    def __init__(self):
        self.cpu = 0.0
        self.cores = []
        self.ram_pct = 0.0
        self.ram_used = 0.0
        self.ram_tot = 0.0
        self.gpu_name = ""
        self.gpu_load = 0.0
        self.gpu_mem_pct = 0.0
        self.gpu_temp = 0
        self.gpu_mem_u = 0.0
        self.gpu_mem_t = 0.0
        self.net_dl = 0.0
        self.net_ul = 0.0
        self.net_dl_tot = 0.0
        self.net_ul_tot = 0.0
        self._lnet = None
        self._lt = None
        self._h_dl = deque(maxlen=80)
        self._h_ul = deque(maxlen=80)
        self._h_cpu = deque(maxlen=80)
        self._h_gpu = deque(maxlen=80)
        self._lock = threading.Lock()
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._run = False

    def _loop(self):
        while self._run:
            self._upd()
            time.sleep(1.0 / STATS_FPS)

    def _upd(self):
        try:
            if psutil:
                self.cpu = psutil.cpu_percent(interval=0)
                self.cores = psutil.cpu_percent(percpu=True)
                m = psutil.virtual_memory()
                self.ram_pct, self.ram_used, self.ram_tot = m.percent, m.used / (1024**3), m.total / (1024**3)
                net = psutil.net_io_counters()
                now = time.time()
                if self._lnet and self._lt:
                    dt = now - self._lt
                    if dt > 0:
                        self.net_dl = (net.bytes_recv - self._lnet.bytes_recv) / dt
                        self.net_ul = (net.bytes_sent - self._lnet.bytes_sent) / dt
                self.net_dl_tot, self.net_ul_tot = net.bytes_recv, net.bytes_sent
                self._lnet, self._lt = net, now
                with self._lock:
                    self._h_dl.append(self.net_dl)
                    self._h_ul.append(self.net_ul)
                    self._h_cpu.append(self.cpu)
            if GPUtil:
                gpus = GPUtil.getGPUs()
                if gpus:
                    g = gpus[0]
                    self.gpu_name = g.name[:20]
                    self.gpu_load = g.load * 100
                    self.gpu_mem_pct = (g.memoryUtil * 100) if g.memoryUtil else 0
                    self.gpu_temp = int(g.temperature) if g.temperature else 0
                    self.gpu_mem_u = g.memoryUsed / 1024 if g.memoryUsed else 0
                    self.gpu_mem_t = g.memoryTotal / 1024 if g.memoryTotal else 0
                    with self._lock:
                        self._h_gpu.append(self.gpu_load)
        except Exception:
            pass

    def hists(self):
        with self._lock:
            return list(self._h_cpu), list(self._h_dl), list(self._h_ul), list(self._h_gpu)


# ─── Helpers ─────────────────────────────────────────────────────
def fmtb(b):
    if b < 1024: return f"{b:.0f}B"
    if b < 1024**2: return f"{b/1024:.1f}KB"
    if b < 1024**3: return f"{b/1024**2:.1f}MB"
    return f"{b/1024**3:.2f}GB"


def fmts(bps):
    if bps < 1024: return f"{bps:.0f} B/s"
    if bps < 1024**2: return f"{bps/1024:.1f} KB/s"
    if bps < 1024**3: return f"{bps/1024**2:.1f} MB/s"
    return f"{bps/1024**3:.2f} GB/s"


def fmtk(n):
    if n < 1000: return f"{n}"
    if n < 1_000_000: return f"{n/1000:.1f}K"
    return f"{n/1_000_000:.2f}M"


def bar(pct, w, fc):
    f = max(0, min(int(pct / 100 * w), w))
    return f"{fc}{'█' * f}{C_DIM}{'░' * (w - f)}{RST}"


def spark(hist, w, color):
    if not hist:
        return f"{C_DIM}{'─' * w}{RST}"
    data = hist[-w:] if len(hist) >= w else hist
    mx = max(data) if data else 1
    if mx == 0: mx = 1
    blk = " ▁▂▃▄▅▆▇█"
    pad = w - len(data)
    ch = [f"{C_DIM}─" for _ in range(pad)]
    for v in data:
        bi = max(0, min(int(v / mx * 8), 8))
        ch.append(f"{color}{blk[bi]}")
    return "".join(ch) + RST


# ─── Box helpers ─────────────────────────────────────────────────
def bx_line(content, w):
    return f"{C_BDR}║{RST}" + vpad(content, w) + f"{C_BDR}║{RST}"


def bx_top(w):
    return f"{C_BDR}╔{'═' * w}╗{RST}"


def bx_bot(w):
    return f"{C_BDR}╚{'═' * w}╝{RST}"


def bx_sep(w, c='─'):
    return f"{C_BDR}╠{c * w}╣{RST}"


def bx_hdr(title, w):
    lp = (w - len(title) - 4) // 2
    rp = w - len(title) - 4 - lp
    return bx_line(f"{' ' * lp}{C_HDR}{BOLD} ⟨ {title} ⟩ {RST}{' ' * rp}", w)


# ─── Panel: Claude Context ───────────────────────────────────────
def build_pc(cl, w, max_h):
    """Claude usage context panel."""
    L = []
    cd = cl.get() if cl else None

    L.append(bx_hdr("CLAUDE", w))
    L.append(bx_sep(w, '═'))

    if not cd or not cd['model']:
        L.append(bx_line(f" {C_DIM2}Loading...{RST}", w))
        while len(L) < max_h:
            L.append(bx_line("", w))
        return L[:max_h]

    ms = cd['model'].replace('claude-', '').replace('-20251101', '').replace('-20250929', '')
    ao = sum(v.get('out', 0) for v in cd['all'].values())
    ai = sum(v.get('in', 0) for v in cd['all'].values())
    acr = sum(v.get('cr', 0) for v in cd['all'].values())

    # Model
    L.append(bx_line(f" {C_PURPLE}{BOLD}◆{RST} {C_VAL}{ms}{RST}", w))
    L.append(bx_sep(w))

    # Session header
    L.append(bx_line(f" {C_HDR}SESSION{RST}", w))

    # Input / Output
    L.append(bx_line(f" {C_LBL}Input    {RST}{C_CYAN}{fmtk(cd['in']):>10}{RST}", w))
    L.append(bx_line(f" {C_LBL}Output   {RST}{C_BRIGHT}{fmtk(cd['out']):>10}{RST}", w))

    # Cache
    L.append(bx_line(f" {C_LBL}Cache R  {RST}{C_GREEN}{fmtk(cd['cache_r']):>10}{RST}", w))
    L.append(bx_line(f" {C_LBL}Cache C  {RST}{C_YELLOW}{fmtk(cd['cache_c']):>10}{RST}", w))
    L.append(bx_sep(w))
    L.append(bx_line(f" {C_LBL}Total    {RST}{C_WHITE}{BOLD}{fmtk(cd['total']):>10}{RST}", w))
    L.append(bx_line(f" {C_LBL}Messages {RST}{C_VAL}{cd['msgs']:>10}{RST}", w))

    # Context bar (approximate % of 200K window)
    ctx_pct = min(100, cd['total'] / 200_000 * 100) if cd['total'] > 0 else 0
    bw = w - 2
    ctx_c = C_GREEN if ctx_pct < 50 else (C_YELLOW if ctx_pct < 80 else C_RED)
    L.append(bx_sep(w))
    L.append(bx_line(f" {C_LBL}Context Window{RST}", w))
    L.append(bx_line(f" {bar(ctx_pct, bw, ctx_c)}", w))
    L.append(bx_line(f" {ctx_c}{ctx_pct:5.1f}%{RST} {C_DIM2}of ~200K{RST}", w))

    # All-time section
    L.append(bx_sep(w, '═'))
    L.append(bx_line(f" {C_HDR}ALL TIME{RST}", w))
    L.append(bx_line(f" {C_LBL}Sessions {RST}{C_VAL}{cd['tot_sess']:>10}{RST}", w))
    L.append(bx_line(f" {C_LBL}Messages {RST}{C_VAL}{cd['tot_msgs']:>10}{RST}", w))
    L.append(bx_line(f" {C_LBL}Output   {RST}{C_BRIGHT}{fmtk(ao):>10}{RST}", w))
    L.append(bx_line(f" {C_LBL}Cache    {RST}{C_GREEN}{fmtk(acr):>10}{RST}", w))

    # Per-model breakdown (if space)
    if len(L) < max_h - 3:
        L.append(bx_sep(w))
        L.append(bx_line(f" {C_LBL}MODELS{RST}", w))
        for m, v in cd['all'].items():
            if len(L) >= max_h - 1:
                break
            short = m.replace('claude-', '').replace('-20251101', '').replace('-20250929', '')[:14]
            L.append(bx_line(f" {C_DIM2}{short:<14}{RST} {C_VAL}{fmtk(v.get('out',0))}{RST}", w))

    while len(L) < max_h:
        L.append(bx_line("", w))
    return L[:max_h]


# ─── Panel: System (CPU/RAM/GPU) ────────────────────────────────
def build_p2(stats, w, max_h):
    L = []
    bw = w - 12
    if bw < 4: bw = 4
    sw = w - 2

    ch, _, _, gh = stats.hists()

    L.append(bx_hdr("SYSTEM", w))
    L.append(bx_sep(w, '═'))

    # CPU
    cc = C_GREEN if stats.cpu < 60 else (C_YELLOW if stats.cpu < 85 else C_RED)
    L.append(bx_line(f" {C_LBL}CPU {RST}{bar(stats.cpu, bw, cc)} {cc}{stats.cpu:5.1f}%{RST}", w))
    if len(L) < max_h - 6:
        L.append(bx_line(f" {spark(ch, sw, cc)} ", w))
    L.append(bx_sep(w))

    # RAM
    rc = C_CYAN if stats.ram_pct < 70 else (C_YELLOW if stats.ram_pct < 90 else C_RED)
    L.append(bx_line(f" {C_LBL}RAM {RST}{bar(stats.ram_pct, bw, rc)} {rc}{stats.ram_pct:5.1f}%{RST}", w))
    if len(L) < max_h - 5:
        L.append(bx_line(f" {C_DIM2}{stats.ram_used:.1f}G / {stats.ram_tot:.1f}G{RST}", w))
    L.append(bx_sep(w))

    # GPU
    if GPUtil and stats.gpu_name:
        gc = C_PURPLE if stats.gpu_load < 60 else (C_YELLOW if stats.gpu_load < 85 else C_RED)
        L.append(bx_line(f" {C_LBL}GPU {RST}{bar(stats.gpu_load, bw, gc)} {gc}{stats.gpu_load:5.1f}%{RST}", w))
        if len(L) < max_h - 4:
            L.append(bx_line(f" {spark(gh, sw, gc)} ", w))
            ms = f"{stats.gpu_mem_u:.1f}G/{stats.gpu_mem_t:.1f}G" if stats.gpu_mem_t else f"{stats.gpu_mem_pct:.0f}%"
            ts = f" {stats.gpu_temp}°C" if stats.gpu_temp else ""
            L.append(bx_line(f" {C_DIM2}{stats.gpu_name} {ms}{ts}{RST}", w))
    else:
        L.append(bx_line(f" {C_DIM}GPU  N/A{RST}", w))

    # Uptime
    if psutil and len(L) < max_h - 1:
        try:
            up = time.time() - psutil.boot_time()
            L.append(bx_sep(w))
            L.append(bx_line(f" {C_DIM2}UP {C_VAL}{int(up//3600)}h {int(up%3600//60)}m{RST}", w))
        except Exception:
            pass

    while len(L) < max_h:
        L.append(bx_line("", w))
    return L[:max_h]


# ─── Panel 3: Network ───────────────────────────────────────────
def build_p3(stats, w, max_h):
    L = []
    sw = w - 2
    _, hdl, hul, _ = stats.hists()

    L.append(bx_hdr("NETWORK", w))
    L.append(bx_sep(w, '═'))

    # Fixed lines without graphs:
    # header+sep(2) + DL_label+speed(2) + sep(1) + UL_label+speed(2) + sep+peak(2) = 9
    # Each graph needs at least 1 row, so minimum = 9 + 2 = 11
    # If not enough room, use sparkline (1 row) instead of multi-row graph
    fixed = 9
    avail = max(0, max_h - fixed)

    if avail >= 4:
        # Enough room for multi-row graphs
        gh = avail // 2
        gh_dl = gh
        gh_ul = avail - gh_dl
    else:
        # Tight: 1-row sparkline each
        gh_dl = 1
        gh_ul = 1

    # Download
    dl = fmts(stats.net_dl)
    dlt = fmtb(stats.net_dl_tot)
    L.append(bx_line(f" {C_CYAN}{BOLD}▼ DL{RST} {C_VAL}{dl}{RST} {C_DIM2}{dlt}{RST}", w))
    if gh_dl > 1:
        dl_graph = _build_graph(hdl, sw, gh_dl, C_CYAN)
        L.extend([bx_line(f" {row} ", w) for row in dl_graph])
    else:
        L.append(bx_line(f" {spark(hdl, sw, C_CYAN)} ", w))

    L.append(bx_sep(w))

    # Upload
    ul = fmts(stats.net_ul)
    ult = fmtb(stats.net_ul_tot)
    L.append(bx_line(f" {C_ORANGE}{BOLD}▲ UL{RST} {C_VAL}{ul}{RST} {C_DIM2}{ult}{RST}", w))
    if gh_ul > 1:
        ul_graph = _build_graph(hul, sw, gh_ul, C_ORANGE)
        L.extend([bx_line(f" {row} ", w) for row in ul_graph])
    else:
        L.append(bx_line(f" {spark(hul, sw, C_ORANGE)} ", w))

    # Peak info
    pk_dl = max(hdl) if hdl else 0
    pk_ul = max(hul) if hul else 0
    if pk_dl > 0 or pk_ul > 0:
        L.append(bx_sep(w))
        L.append(bx_line(f" {C_DIM2}Peak{RST} {C_CYAN}▼{fmts(pk_dl)}{RST} {C_ORANGE}▲{fmts(pk_ul)}{RST}", w))

    while len(L) < max_h:
        L.append(bx_line("", w))
    return L[:max_h]


def _build_graph(hist, w, h, color):
    """Build a multi-row bar graph from history data."""
    if not hist or h < 1:
        return [f"{C_DIM}{'─' * w}{RST}"]
    data = hist[-w:] if len(hist) >= w else hist
    mx = max(data) if data else 1
    if mx == 0:
        mx = 1

    pad = w - len(data)
    rows = []
    blk_full = "█"
    blk_parts = " ▁▂▃▄▅▆▇█"

    for row_i in range(h, 0, -1):
        chars = [f"{C_DIM}·" for _ in range(pad)]
        threshold_lo = (row_i - 1) / h
        threshold_hi = row_i / h
        for v in data:
            norm = v / mx
            if norm >= threshold_hi:
                chars.append(f"{color}{blk_full}")
            elif norm > threshold_lo:
                frac = (norm - threshold_lo) / (threshold_hi - threshold_lo)
                bi = max(0, min(int(frac * 8), 8))
                chars.append(f"{color}{blk_parts[bi]}")
            else:
                chars.append(f"{C_DIM}·" if row_i == 1 else " ")
        rows.append("".join(chars) + RST)
    return rows


# ─── Render ──────────────────────────────────────────────────────
def render(sm, pk, gain, stats, cl_usage):
    rows, cols = term_size()

    show_panels = stats is not None and cols >= 120
    if show_panels:
        # 4 panels: bars ~40%, claude ~20%, sys ~20%, net ~20%
        pcw_outer = max(24, min(32, int(cols * 0.18)))
        p2w_outer = max(24, min(34, int(cols * 0.20)))
        p3w_outer = max(24, min(34, int(cols * 0.20)))
        bar_w = cols - pcw_outer - p2w_outer - p3w_outer
        pcw = pcw_outer - 2
        p2w = p2w_outer - 2
        p3w = p3w_outer - 2
    else:
        bar_w = cols
        pcw_outer = p2w_outer = p3w_outer = pcw = p2w = p3w = 0

    # Sound bar inner width = bar_w minus 2 borders
    if show_panels:
        bars_inner = bar_w - 2
    else:
        bars_inner = bar_w

    draw_h = max(3, rows - 1)  # use almost full height (border top/bot handled by panel)
    draw_bars = max(10, bars_inner)
    grad = make_grad(draw_h)

    if len(sm) != draw_bars:
        sm = np.zeros(draw_bars)
        pk = np.zeros(draw_bars)
    sm, pk = spectrum(draw_bars, draw_h, gain, sm, pk)

    out = [f"{ESC}[H"]

    if show_panels:
        # All 4 panels rendered via move_to for pixel-perfect placement
        pmax = rows - 2  # inner height (between top/bot borders)
        bar_h = pmax  # sound bars use same height as other panels
        draw_h_actual = max(3, bar_h - 2)  # -2 for header + VU

        # Recompute gradient for actual draw height
        grad = make_grad(draw_h_actual)

        # ── Panel 1: Sound Bars ──
        p1_col = 1
        p1w = bar_w - 2  # inner width
        out.append(mv(1, p1_col) + bx_top(p1w))

        # Header inside box
        hdr = "SOUNDBAR"
        hdr_line = bx_hdr(hdr, p1w)
        out.append(mv(2, p1_col) + hdr_line)

        # Bars
        for ri in range(draw_h_actual):
            row = draw_h_actual - ri
            ch = []
            for col in range(min(draw_bars, p1w)):
                h = sm[col]
                p = pk[col]
                if row <= h * draw_h_actual / max(draw_h, 1):
                    ci = min(int((row - 1) / draw_h_actual * len(grad)), len(grad) - 1)
                    ch.append(f"{grad[ci]}█")
                elif abs(row - pk[col] * draw_h_actual / max(draw_h, 1)) < 0.9 and pk[col] > 1.5:
                    ch.append(f"{C_BRIGHT}▀")
                else:
                    if row % 4 == 0 and col % 6 == 0:
                        ch.append(f"{C_DIM}·")
                    else:
                        ch.append(" ")
            bar_content = "".join(ch) + RST
            tr = 3 + ri
            if tr <= rows - 1:
                out.append(mv(tr, p1_col) + f"{C_BDR}║{RST}" + vpad(bar_content, p1w) + f"{C_BDR}║{RST}")

        # VU meter inside box
        vu_row = 3 + draw_h_actual
        vol = float(np.mean(sm))
        vpct = min(vol / max(draw_h, 1) * 100, 100)
        vuw = max(p1w - 12, 10)
        vul = int(vpct / 100 * vuw)
        vc = C_MID if vpct < 50 else (C_BRIGHT if vpct < 80 else C_WHITE)
        vu_content = f" {C_MID}[VOL]{RST} {vc}{'█' * vul}{C_DIM}{'░' * (vuw - vul)}{RST} {C_BRIGHT}{vpct:3.0f}%{RST}"
        if vu_row <= rows - 1:
            out.append(mv(vu_row, p1_col) + bx_line(vu_content, p1w))

        # Fill remaining rows
        for fr in range(vu_row + 1, rows):
            out.append(mv(fr, p1_col) + bx_line("", p1w))

        # Bottom border
        if rows >= 2:
            out.append(mv(rows, p1_col) + bx_bot(p1w))

        # ── Panel C: Claude Context ──
        pc_col = bar_w + 1
        pc_lines = build_pc(cl_usage, pcw, pmax)
        out.append(mv(1, pc_col) + bx_top(pcw))
        for i, line in enumerate(pc_lines):
            r = 2 + i
            if r <= rows - 1:
                out.append(mv(r, pc_col) + line)
        out.append(mv(rows, pc_col) + bx_bot(pcw))

        # ── Panel 2: System ──
        p2_col = bar_w + pcw_outer + 1
        p2_lines = build_p2(stats, p2w, pmax)
        out.append(mv(1, p2_col) + bx_top(p2w))
        for i, line in enumerate(p2_lines):
            r = 2 + i
            if r <= rows - 1:
                out.append(mv(r, p2_col) + line)
        out.append(mv(rows, p2_col) + bx_bot(p2w))

        # ── Panel 3: Network ──
        p3_col = bar_w + pcw_outer + p2w_outer + 1
        p3_lines = build_p3(stats, p3w, pmax)
        out.append(mv(1, p3_col) + bx_top(p3w))
        for i, line in enumerate(p3_lines):
            r = 2 + i
            if r <= rows - 1:
                out.append(mv(r, p3_col) + line)
        out.append(mv(rows, p3_col) + bx_bot(p3w))

    else:
        # No panels — simple fullscreen bars
        draw_h = max(3, rows - 3)
        grad = make_grad(draw_h)
        if len(sm) != bar_w:
            sm = np.zeros(bar_w)
            pk = np.zeros(bar_w)
        sm, pk = spectrum(bar_w, draw_h, gain, sm, pk)

        hdr = " SOUNDBAR v2 // HACKER EDITION "
        hw = len(hdr)
        lp = max(0, (bar_w - hw) // 2)
        rp = max(0, bar_w - hw - lp)
        out.append(f"{C_DIM}{'░' * lp}{C_HDR}{BOLD}{hdr}{RST}{C_DIM}{'░' * rp}{RST}\n")

        for ri in range(draw_h):
            row = draw_h - ri
            ch = []
            for col in range(bar_w):
                h = sm[col]
                p = pk[col]
                if row <= h:
                    ci = min(int((row - 1) / draw_h * len(grad)), len(grad) - 1)
                    ch.append(f"{grad[ci]}█")
                elif abs(row - p) < 0.9 and p > 1.5:
                    ch.append(f"{C_BRIGHT}▀")
                else:
                    if row % 4 == 0 and col % 6 == 0:
                        ch.append(f"{C_DIM}·")
                    else:
                        ch.append(" ")
            out.append("".join(ch) + RST + '\n')

        vol = float(np.mean(sm))
        vpct = min(vol / max(draw_h, 1) * 100, 100)
        vuw = max(bar_w - 12, 10)
        vul = int(vpct / 100 * vuw)
        vc = C_MID if vpct < 50 else (C_BRIGHT if vpct < 80 else C_WHITE)
        out.append(f" {C_MID}[VOL]{RST} {vc}{'█' * vul}{C_DIM}{'░' * (vuw - vul)}{RST} {C_BRIGHT}{vpct:3.0f}%{RST}")

    sys.stdout.write("".join(out))
    sys.stdout.flush()
    return sm, pk


# ─── PID / Stop ──────────────────────────────────────────────────
def stop_bg():
    if not os.path.exists(PID_FILE):
        print("No running soundbar found.")
        return
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        os.remove(PID_FILE)
        print(f"Soundbar (PID {pid}) stopped.")
    except ProcessLookupError:
        os.remove(PID_FILE)
        print("Stale PID removed.")
    except Exception as e:
        print(f"Error: {e}")


def write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def clean_pid():
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE) as f:
                if int(f.read().strip()) == os.getpid():
                    os.remove(PID_FILE)
    except Exception:
        pass


def list_devs():
    p = pyaudio.PyAudio()
    print(f"\n  {C_HDR}Loopback Devices:{RST}")
    print(f"  {C_BDR}{'─' * 60}{RST}")
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d.get("isLoopbackDevice"):
            print(f"  {C_BRIGHT}[{i}]{RST} {C_VAL}{d['name']}{RST}  {C_DIM2}({int(d['defaultSampleRate'])}Hz){RST}")
    print()
    p.terminate()


# ─── Main ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Soundbar + System Monitor")
    ap.add_argument("--bars", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--top", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--no-claude", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_devs(); return
    if args.stop:
        stop_bg(); return

    if sys.platform == "win32":
        os.system("")

    gain = args.gain

    p = pyaudio.PyAudio()
    dev = find_loopback(p, args.device)
    if not dev:
        print(f"{C_RED}[!]{RST} No WASAPI loopback. Run: soundbar --list")
        p.terminate(); sys.exit(1)

    sr = int(dev["defaultSampleRate"])
    ch = dev["maxInputChannels"]

    r, c = term_size()
    sm = np.zeros(c)
    pk = np.zeros(c)

    stats = SysStats() if (not args.no_stats and psutil) else None
    cl = ClaudeUsage() if not args.no_claude else None

    sys.stdout.write(HIDE_CUR + f"{ESC}[2J{ESC}[H")
    sys.stdout.flush()

    write_pid()
    running = True

    def on_sig(s, f):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    stream = None
    try:
        stream = p.open(
            format=pyaudio.paFloat32, channels=ch, rate=sr,
            input=True, input_device_index=dev["index"],
            frames_per_buffer=BLOCK_SIZE, stream_callback=audio_cb,
        )
        stream.start_stream()
        while running and stream.is_active():
            sm, pk = render(sm, pk, gain, stats, cl)
            time.sleep(1.0 / FPS)
    except Exception as e:
        sys.stdout.write(SHOW_CUR + "\n")
        print(f"{C_RED}[!]{RST} {e}")
    finally:
        if stats: stats.stop()
        if cl: cl.stop()
        if stream:
            try: stream.stop_stream(); stream.close()
            except: pass
        p.terminate()
        clean_pid()
        sys.stdout.write(SHOW_CUR + f"{ESC}[2J{ESC}[H")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
