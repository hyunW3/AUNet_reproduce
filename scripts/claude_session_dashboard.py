#!/usr/bin/env python3
"""
Claude session dashboard.

Scans the local Claude Code state under ~/.claude and serves a single-page web
dashboard that shows the status of *every* session it can find -- old or new --
plus per-session task lists, what each session is currently doing, and live
terminal attach / remote-control.

Data sources (read-only):
  ~/.claude/projects/<encoded-cwd>/<sessionId>.jsonl  -- full transcript of every
        session ever recorded (the canonical "all sessions" record).
  ~/.claude/sessions/<pid>.json                       -- currently-tracked sessions
        with a live status field (active/idle/...).
  ~/.claude/tasks/<sessionId>/<n>.json                -- per-session task lists.

Features:
  * Live/idle/ended status, task progress, token counts (auto-refresh).
  * "Hide ended" toggle.
  * Click a card to see what the session is doing (recent prompt/reply/tool timeline).
  * "Connect terminal": embedded web terminal attached to the live session. If the
    session runs under tmux it does `tmux attach` -> truly synchronized with the
    original terminal; otherwise it runs `claude --resume <id>`.
  * "Remote control": mints a tokenized shareable link to a focused terminal page so
    the session can be driven from another device.

Usage:
  python3 claude_session_dashboard.py [--port 8765] [--host 127.0.0.1]
  python3 claude_session_dashboard.py --remote            # bind 0.0.0.0 + require token
  python3 claude_session_dashboard.py --remote --token MYSECRET

Local use (127.0.0.1) needs no token. Remote use requires the printed token.
Stdlib only -- the browser loads xterm.js from a CDN for the terminal view.
"""
import argparse
import base64
import collections
import fcntl
import json
import os
import pty
import select
import shlex
import socket
import struct
import subprocess
import termios
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HOME = os.path.expanduser("~")
CLAUDE_DIR = os.path.join(HOME, ".claude")
PROJECTS_DIR = os.path.join(CLAUDE_DIR, "projects")
SESSIONS_DIR = os.path.join(CLAUDE_DIR, "sessions")
TASKS_DIR = os.path.join(CLAUDE_DIR, "tasks")
NICK_FILE = os.path.join(CLAUDE_DIR, "dashboard_nicknames.json")

LIVE_IDLE_SECONDS = 90      # tracked + activity within this -> "active"
_nick_lock = threading.Lock()

# runtime config (set in main)
REQUIRE_TOKEN = False
BASE_TOKEN = None
REMOTE_TOKENS = set()
CLAUDE_BIN = "claude"

# ---------------------------------------------------------------------------
# Transcript parsing (cached by mtime+size).
# ---------------------------------------------------------------------------
_transcript_cache = {}


def _to_epoch(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return ts / 1000.0 if ts > 1e12 else float(ts)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _first_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text", "")
            if isinstance(part, str):
                return part
    return ""


def parse_transcript(path):
    summary = {
        "session_id": None, "ai_title": None, "first_prompt": None,
        "last_prompt": None, "cwd": None, "git_branch": None, "version": None,
        "started_at": None, "last_activity": None, "user_msgs": 0,
        "assistant_msgs": 0, "models": set(), "input_tokens": 0,
        "output_tokens": 0, "cache_read_tokens": 0,
    }
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                t = o.get("type")
                if summary["session_id"] is None and o.get("sessionId"):
                    summary["session_id"] = o["sessionId"]
                if t == "ai-title":
                    summary["ai_title"] = o.get("aiTitle") or summary["ai_title"]
                    continue
                if t == "last-prompt":
                    lp = o.get("lastPrompt")
                    if isinstance(lp, str):
                        summary["last_prompt"] = lp
                    continue
                if t not in ("user", "assistant"):
                    continue
                for k_src, k_dst in (("cwd", "cwd"), ("gitBranch", "git_branch"),
                                     ("version", "version")):
                    if o.get(k_src):
                        summary[k_dst] = o[k_src]
                ts = _to_epoch(o.get("timestamp"))
                if ts:
                    if summary["started_at"] is None:
                        summary["started_at"] = ts
                    summary["last_activity"] = ts
                msg = o.get("message") or {}
                if t == "user":
                    if o.get("isSidechain"):
                        continue
                    content = msg.get("content")
                    text = _first_text(content)
                    is_typed = isinstance(content, str) and not o.get("toolUseResult")
                    if is_typed and text and not text.startswith("<"):
                        summary["user_msgs"] += 1
                        if summary["first_prompt"] is None:
                            summary["first_prompt"] = text
                        summary["last_prompt"] = text
                elif t == "assistant":
                    summary["assistant_msgs"] += 1
                    if msg.get("model"):
                        summary["models"].add(msg["model"])
                    usage = msg.get("usage") or {}
                    summary["input_tokens"] += usage.get("input_tokens", 0) or 0
                    summary["output_tokens"] += usage.get("output_tokens", 0) or 0
                    summary["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0) or 0
    except Exception as exc:
        summary["parse_error"] = str(exc)
    summary["models"] = sorted(summary["models"])
    return summary


def get_transcript_summary(path):
    st = os.stat(path)
    key = (st.st_mtime, st.st_size)
    cached = _transcript_cache.get(path)
    if cached and cached[0] == key:
        return cached[1]
    summary = parse_transcript(path)
    summary["file_mtime"] = st.st_mtime
    summary["file_size"] = st.st_size
    summary["path"] = path
    _transcript_cache[path] = (key, summary)
    return summary


# ---------------------------------------------------------------------------
# Recent-activity timeline (tail of a transcript) -> "what is it doing".
# ---------------------------------------------------------------------------
def _tool_brief(inp):
    if not isinstance(inp, dict):
        return ""
    for k in ("command", "file_path", "pattern", "path", "query", "prompt",
              "description", "url", "subject"):
        if inp.get(k):
            return f"{k}: {str(inp[k])[:160]}"
    return ", ".join(list(inp.keys())[:4])


def tail_events(path, limit=50):
    size = os.path.getsize(path)
    span = min(size, 800 * 1024)
    with open(path, "rb") as fh:
        fh.seek(size - span)
        data = fh.read()
    lines = data.decode("utf-8", "replace").split("\n")
    if span < size:
        lines = lines[1:]  # drop the partial first line
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get("type")
        msg = o.get("message") or {}
        ts = _to_epoch(o.get("timestamp"))
        if t == "user" and not o.get("isSidechain"):
            c = msg.get("content")
            if isinstance(c, str) and c and not c.startswith("<") and not o.get("toolUseResult"):
                events.append({"k": "user", "t": ts, "text": c[:4000]})
        elif t == "assistant":
            c = msg.get("content")
            if isinstance(c, list):
                for part in c:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and part.get("text", "").strip():
                        events.append({"k": "assistant", "t": ts,
                                       "text": part["text"][:4000]})
                    elif part.get("type") == "tool_use":
                        events.append({"k": "tool", "t": ts,
                                       "name": part.get("name"),
                                       "brief": _tool_brief(part.get("input"))})
    return events[-limit:]


def session_context(sid):
    """Find the transcript for a session and return recent timeline + meta."""
    path = None
    if os.path.isdir(PROJECTS_DIR):
        for proj in os.listdir(PROJECTS_DIR):
            cand = os.path.join(PROJECTS_DIR, proj, sid + ".jsonl")
            if os.path.isfile(cand):
                path = cand
                break
    if not path:
        return {"error": "no transcript", "events": []}
    summ = get_transcript_summary(path)
    events = tail_events(path)
    doing = None
    for ev in reversed(events):
        if ev["k"] == "tool":
            doing = f"running {ev.get('name')} — {ev.get('brief','')}"
            break
        if ev["k"] == "assistant":
            doing = ev["text"][:200]
            break
    return {
        "session_id": sid,
        "title": summ.get("ai_title") or summ.get("first_prompt"),
        "first_prompt": summ.get("first_prompt"),
        "cwd": summ.get("cwd"),
        "git_branch": summ.get("git_branch"),
        "models": summ.get("models", []),
        "doing": doing,
        "events": events,
    }


# ---------------------------------------------------------------------------
# Live sessions + tasks.
# ---------------------------------------------------------------------------
def load_live_sessions():
    out = {}
    if not os.path.isdir(SESSIONS_DIR):
        return out
    for name in os.listdir(SESSIONS_DIR):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(SESSIONS_DIR, name)) as fh:
                o = json.load(fh)
        except Exception:
            continue
        sid = o.get("sessionId")
        if not sid:
            continue
        pid = o.get("pid")
        alive = bool(pid) and os.path.isdir(f"/proc/{pid}")
        out[sid] = {
            "pid": pid, "status": o.get("status"), "kind": o.get("kind"),
            "updated_at": _to_epoch(o.get("updatedAt")),
            "started_at": _to_epoch(o.get("startedAt")),
            "version": o.get("version"), "cwd": o.get("cwd"),
            "proc_alive": alive,
        }
    return out


def load_tasks():
    out = {}
    if not os.path.isdir(TASKS_DIR):
        return out
    for sid in os.listdir(TASKS_DIR):
        d = os.path.join(TASKS_DIR, sid)
        if not os.path.isdir(d):
            continue
        tasks = []
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, name)) as fh:
                    tasks.append(json.load(fh))
            except Exception:
                continue
        if tasks:
            order = {"in_progress": 0, "pending": 1, "completed": 2}
            tasks.sort(key=lambda t: (order.get(t.get("status"), 3), str(t.get("id"))))
            out[sid] = tasks
    return out


def load_nicknames():
    try:
        with open(NICK_FILE) as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def set_nickname(sid, nick):
    with _nick_lock:
        data = load_nicknames()
        if nick:
            data[sid] = nick[:120]
        else:
            data.pop(sid, None)
        tmp = NICK_FILE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, NICK_FILE)
    return data.get(sid)


def build_state():
    live = load_live_sessions()
    tasks_by_sid = load_tasks()
    nicks = load_nicknames()
    now = time.time()
    sessions, seen = [], set()
    if os.path.isdir(PROJECTS_DIR):
        for proj in sorted(os.listdir(PROJECTS_DIR)):
            pdir = os.path.join(PROJECTS_DIR, proj)
            if not os.path.isdir(pdir):
                continue
            for name in os.listdir(pdir):
                if not name.endswith(".jsonl"):
                    continue
                summ = get_transcript_summary(os.path.join(pdir, name))
                sid = summ.get("session_id") or name[:-6]
                seen.add(sid)
                sessions.append(_assemble(summ, sid, proj, live, tasks_by_sid, now, nicks))
    for sid, lv in live.items():
        if sid in seen:
            continue
        summ = {"session_id": sid, "started_at": lv.get("started_at"),
                "last_activity": lv.get("updated_at"), "models": [],
                "user_msgs": 0, "assistant_msgs": 0, "input_tokens": 0,
                "output_tokens": 0, "cache_read_tokens": 0,
                "version": lv.get("version"), "cwd": lv.get("cwd")}
        sessions.append(_assemble(summ, sid, None, live, tasks_by_sid, now, nicks))
    sessions.sort(key=lambda s: s.get("last_activity") or 0, reverse=True)
    summary = {
        "total": len(sessions),
        "active": sum(1 for s in sessions if s["state"] == "active"),
        "idle": sum(1 for s in sessions if s["state"] == "idle"),
        "ended": sum(1 for s in sessions if s["state"] == "ended"),
        "in_progress_tasks": sum(s["tasks"]["in_progress"] for s in sessions),
        "generated_at": now,
    }
    return {"summary": summary, "sessions": sessions}


def _assemble(summ, sid, proj, live, tasks_by_sid, now, nicks=None):
    lv = live.get(sid)
    tasks = tasks_by_sid.get(sid, [])
    counts = {"pending": 0, "in_progress": 0, "completed": 0, "total": len(tasks)}
    for t in tasks:
        if t.get("status") in counts:
            counts[t["status"]] += 1
    last = summ.get("last_activity")
    if lv and lv.get("updated_at"):
        last = max(last or 0, lv["updated_at"])
    attachable = bool(lv and lv.get("proc_alive"))
    if lv is not None:
        age = now - (lv.get("updated_at") or last or now)
        state = "active" if (lv.get("status") == "active" or age <= LIVE_IDLE_SECONDS) else "idle"
        if not lv.get("proc_alive"):
            state = "ended"
    else:
        state = "ended"
    title = summ.get("ai_title") or summ.get("first_prompt") or "(untitled session)"
    project = None
    if summ.get("cwd"):
        project = os.path.basename(summ["cwd"].rstrip("/"))
    elif proj:
        project = proj.strip("-").split("-")[-1]
    return {
        "session_id": sid, "short_id": sid[:8] if sid else "?",
        "nickname": (nicks or {}).get(sid),
        "title": title.strip()[:200],
        "first_prompt": (summ.get("first_prompt") or "")[:500],
        "last_prompt": (summ.get("last_prompt") or "")[:500],
        "project": project, "cwd": summ.get("cwd"),
        "git_branch": summ.get("git_branch"), "version": summ.get("version"),
        "models": summ.get("models", []),
        "started_at": summ.get("started_at"), "last_activity": last,
        "user_msgs": summ.get("user_msgs", 0),
        "assistant_msgs": summ.get("assistant_msgs", 0),
        "input_tokens": summ.get("input_tokens", 0),
        "output_tokens": summ.get("output_tokens", 0),
        "cache_read_tokens": summ.get("cache_read_tokens", 0),
        "state": state, "live_status": lv.get("status") if lv else None,
        "pid": lv.get("pid") if lv else None, "attachable": attachable,
        "tasks": counts,
        "task_list": [
            {"subject": t.get("subject"), "status": t.get("status"),
             "active_form": t.get("activeForm")} for t in tasks
        ],
    }


# ---------------------------------------------------------------------------
# Terminal subsystem: real PTYs, attach strategy detection.
# ---------------------------------------------------------------------------
def _proc_ppid(pid):
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return int(fh.read().rsplit(")", 1)[1].split()[1])
    except Exception:
        return None


def _ancestry(pid):
    res, cur = set(), pid
    for _ in range(50):
        if cur is None or cur <= 1:
            break
        res.add(cur)
        cur = _proc_ppid(cur)
    return res


def find_tmux_target(pid):
    """Return tmux session name owning `pid`, or None."""
    if not pid:
        return None
    try:
        out = subprocess.check_output(
            ["tmux", "list-panes", "-a", "-F", "#{pane_pid} #{session_name}"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    anc = _ancestry(pid)
    for line in out.splitlines():
        try:
            pp, sess = line.split(" ", 1)
            pp = int(pp)
        except ValueError:
            continue
        if pp in anc:
            return sess
    return None


def choose_attach(sid, live, summ_cwd, mode="auto"):
    """Return (cmd_list, strategy, note, cwd)."""
    lv = live.get(sid)
    pid = lv.get("pid") if lv else None
    cwd = (lv.get("cwd") if lv else None) or summ_cwd or HOME
    if mode in ("auto", "attach") and pid and lv and lv.get("proc_alive"):
        sess = find_tmux_target(pid)
        if sess:
            return (["tmux", "attach", "-t", sess], "tmux",
                    f"Synchronized with the original terminal (tmux session '{sess}'). "
                    f"Both views share one screen.", cwd)
        if mode == "attach":
            return (None, "none",
                    "Session is not running under tmux, so its live terminal cannot be "
                    "mirrored. Use 'resume' to reconnect in a fresh terminal.", cwd)
    # fallback: resume the session in a fresh PTY
    return ([CLAUDE_BIN, "--resume", sid], "resume",
            "Reconnected to this session in a new terminal via `claude --resume`. "
            "Not a mirror of the original screen, but the same conversation state.", cwd)


class Term:
    CAP = 1 << 20  # keep last 1 MiB of output

    def __init__(self, cmd, cwd):
        self.master, slave = pty.openpty()
        env = dict(os.environ)
        env["TERM"] = "xterm-256color"
        self.proc = subprocess.Popen(
            cmd, stdin=slave, stdout=slave, stderr=slave,
            preexec_fn=os.setsid, cwd=cwd if os.path.isdir(cwd or "") else HOME,
            env=env, close_fds=True)
        os.close(slave)
        self.buf = bytearray()
        self.base = 0
        self.lock = threading.Lock()
        self.alive = True
        self.last_used = time.time()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            try:
                r, _, _ = select.select([self.master], [], [], 0.4)
            except (OSError, ValueError):
                break
            if self.master in r:
                try:
                    data = os.read(self.master, 65536)
                except OSError:
                    break
                if not data:
                    break
                with self.lock:
                    self.buf += data
                    if len(self.buf) > self.CAP:
                        drop = len(self.buf) - self.CAP
                        del self.buf[:drop]
                        self.base += drop
            if self.proc.poll() is not None:
                # drain anything left then stop
                try:
                    data = os.read(self.master, 65536)
                    if data:
                        with self.lock:
                            self.buf += data
                except OSError:
                    pass
                break
        self.alive = False

    def read_since(self, since):
        with self.lock:
            self.last_used = time.time()
            total = self.base + len(self.buf)
            since = max(since, self.base)
            return bytes(self.buf[since - self.base:]), total, self.base

    def write(self, data):
        try:
            os.write(self.master, data)
        except OSError:
            self.alive = False

    def resize(self, cols, rows):
        try:
            fcntl.ioctl(self.master, termios.TIOCSWINSZ,
                        struct.pack("HHHH", rows, cols, 0, 0))
        except OSError:
            pass

    def close(self):
        try:
            self.proc.terminate()
        except Exception:
            pass
        try:
            os.close(self.master)
        except OSError:
            pass
        self.alive = False


class TermManager:
    def __init__(self):
        self.terms = {}
        self.counter = 0
        self.lock = threading.Lock()
        threading.Thread(target=self._reaper, daemon=True).start()

    def create(self, cmd, cwd):
        with self.lock:
            self.counter += 1
            tid = f"t{self.counter}"
            self.terms[tid] = Term(cmd, cwd)
            return tid

    def get(self, tid):
        return self.terms.get(tid)

    def _reaper(self):
        while True:
            time.sleep(30)
            now = time.time()
            for tid, term in list(self.terms.items()):
                if not term.alive and now - term.last_used > 60:
                    term.close()
                    self.terms.pop(tid, None)
                elif now - term.last_used > 1800:  # 30 min idle
                    term.close()
                    self.terms.pop(tid, None)


TERMS = TermManager()


def open_terminal(sid, mode="auto"):
    live = load_live_sessions()
    # find cwd from transcript if needed
    cwd = None
    ctx = session_context(sid)
    if isinstance(ctx, dict):
        cwd = ctx.get("cwd")
    cmd, strategy, note, run_cwd = choose_attach(sid, live, cwd, mode)
    if cmd is None:
        return {"error": note, "strategy": strategy}
    tid = TERMS.create(cmd, run_cwd)
    return {"term_id": tid, "strategy": strategy, "note": note,
            "cmd": " ".join(shlex.quote(c) for c in cmd), "cwd": run_cwd}


# ---------------------------------------------------------------------------
# HTTP server.
# ---------------------------------------------------------------------------
def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    # -- helpers -----------------------------------------------------------
    def _send(self, code, body, ctype):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def _authed(self, qs):
        if not REQUIRE_TOKEN:
            return True
        if self.client_address and self.client_address[0] in ("127.0.0.1", "::1"):
            return True
        tok = (qs.get("token", [None])[0]) or self.headers.get("X-Token")
        return tok and (tok == BASE_TOKEN or tok in REMOTE_TOKENS)

    # -- routing -----------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        path = u.path
        if not self._authed(qs):
            return self._send(403, "forbidden: missing/invalid token", "text/plain")
        if path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/remote":
            return self._send(200, REMOTE_PAGE, "text/html; charset=utf-8")
        if path == "/api/sessions":
            try:
                return self._json(build_state())
            except Exception as exc:
                return self._json({"error": str(exc)}, 500)
        if path.startswith("/api/session/") and path.endswith("/context"):
            sid = path[len("/api/session/"):-len("/context")]
            try:
                return self._json(session_context(sid))
            except Exception as exc:
                return self._json({"error": str(exc)}, 500)
        if path.startswith("/api/term/") and path.endswith("/out"):
            tid = path[len("/api/term/"):-len("/out")]
            term = TERMS.get(tid)
            if not term:
                return self._json({"error": "no such terminal", "alive": False}, 404)
            since = int(qs.get("since", ["0"])[0])
            data, total, base = term.read_since(since)
            return self._json({"data": base64.b64encode(data).decode(),
                               "cursor": total, "base": base, "alive": term.alive})
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        path = u.path
        if not self._authed(qs):
            return self._json({"error": "forbidden"}, 403)
        body = self._body()
        if path.startswith("/api/session/") and path.endswith("/nickname"):
            sid = path[len("/api/session/"):-len("/nickname")]
            nick = (body.get("nickname") or "").strip()
            saved = set_nickname(sid, nick)
            return self._json({"ok": True, "nickname": saved})
        if path == "/api/term/new":
            res = open_terminal(body.get("sid"), body.get("mode", "auto"))
            return self._json(res, 200 if "term_id" in res else 400)
        if path.startswith("/api/term/") and path.endswith("/in"):
            tid = path[len("/api/term/"):-len("/in")]
            term = TERMS.get(tid)
            if not term:
                return self._json({"error": "no such terminal"}, 404)
            term.write(base64.b64decode(body.get("data", "")))
            return self._json({"ok": True})
        if path.startswith("/api/term/") and path.endswith("/resize"):
            tid = path[len("/api/term/"):-len("/resize")]
            term = TERMS.get(tid)
            if term:
                term.resize(int(body.get("cols", 80)), int(body.get("rows", 24)))
            return self._json({"ok": True})
        if path.startswith("/api/term/") and path.endswith("/close"):
            tid = path[len("/api/term/"):-len("/close")]
            term = TERMS.get(tid)
            if term:
                term.close()
            return self._json({"ok": True})
        if path == "/api/remote":
            return self._json(self._make_remote(body))
        return self._json({"error": "not found"}, 404)

    def _make_remote(self, body):
        sid = body.get("sid")
        res = open_terminal(sid, body.get("mode", "auto"))
        if "term_id" not in res:
            return res
        tok = base64.urlsafe_b64encode(os.urandom(12)).decode().rstrip("=")
        REMOTE_TOKENS.add(tok)
        host = _lan_ip()
        port = self.server.server_address[1]
        scheme = "http"
        url = f"{scheme}://{host}:{port}/remote?id={res['term_id']}&token={tok}"
        res["remote_url"] = url
        res["token"] = tok
        res["remote_enabled"] = REQUIRE_TOKEN
        if not REQUIRE_TOKEN:
            res["warning"] = ("Server is bound to localhost only — this link works "
                              "only from this machine. Restart with --remote (binds "
                              "0.0.0.0 + token) to reach it from another device.")
        return res


# ---------------------------------------------------------------------------
# Frontend.
# ---------------------------------------------------------------------------
XTERM_HEAD = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@xterm/xterm@5.5.0/css/xterm.min.css">
<script src="https://cdn.jsdelivr.net/npm/@xterm/xterm@5.5.0/lib/xterm.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@xterm/addon-fit@0.10.0/lib/addon-fit.min.js"></script>
"""

# shared client-side terminal driver (xterm + polling)
TERM_JS = r"""
function tokenQS(){ const m=location.search.match(/token=([^&]+)/); return m?('&token='+m[1]):''; }
async function startTerm(elId, termId, statusCb){
  const host = document.getElementById(elId);
  host.innerHTML = "";
  if(!window.Terminal){ host.innerHTML='<div style="padding:20px;color:#f85149">xterm.js failed to load (no internet to CDN). Terminal needs CDN access.</div>'; return null; }
  const term = new Terminal({fontSize:13, cursorBlink:true, theme:{background:'#0b0f14'}, scrollback:5000});
  let fit=null; try{ fit=new FitAddon.FitAddon(); term.loadAddon(fit); }catch(e){}
  term.open(host); if(fit){ try{fit.fit();}catch(e){} }
  let cursor=0, alive=true;
  const q=tokenQS();
  term.onData(d=>{ fetch('/api/term/'+termId+'/in?x=1'+q,{method:'POST',body:JSON.stringify({data:btoa(unescape(encodeURIComponent(d)))})}); });
  function sendSize(){ if(!fit) return; try{fit.fit();}catch(e){} fetch('/api/term/'+termId+'/resize?x=1'+q,{method:'POST',body:JSON.stringify({cols:term.cols,rows:term.rows})}); }
  term.onResize(sendSize); setTimeout(sendSize,150);
  const ro = new ResizeObserver(()=>{ if(fit){try{fit.fit();}catch(e){}} }); ro.observe(host);
  async function poll(){
    if(!alive) return;
    try{
      const r=await fetch('/api/term/'+termId+'/out?since='+cursor+q);
      if(r.status===404){ alive=false; statusCb&&statusCb('closed'); term.write('\r\n[terminal closed]\r\n'); return; }
      const j=await r.json();
      if(j.base>cursor) cursor=j.base;
      if(j.data){ const bin=atob(j.data); const bytes=new Uint8Array(bin.length); for(let i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i); term.write(bytes); cursor=j.cursor; }
      if(!j.alive){ statusCb&&statusCb('exited'); if(!j.data){ return; } }
    }catch(e){}
    setTimeout(poll, 120);
  }
  poll();
  return { term, close(){ alive=false; ro.disconnect(); fetch('/api/term/'+termId+'/close?x=1'+q,{method:'POST'}); } };
}
"""

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude Session Dashboard</title>
""" + XTERM_HEAD + r"""
<style>
  :root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2230;--border:#30363d;--txt:#e6edf3;
    --muted:#8b949e;--accent:#d97757;--active:#3fb950;--idle:#d29922;--ended:#6e7681;
    --prog:#388bfd;--done:#3fb950;--todo:#6e7681}
  *{box-sizing:border-box}
  body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--txt)}
  header{position:sticky;top:0;z-index:10;background:rgba(13,17,23,.92);backdrop-filter:blur(8px);
    border-bottom:1px solid var(--border);padding:14px 22px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
  h1{font-size:18px;margin:0;font-weight:650}h1 .dot{color:var(--accent)}
  .stats{display:flex;gap:16px;flex-wrap:wrap;margin-left:auto}
  .stat{display:flex;flex-direction:column;align-items:flex-end;line-height:1.1}
  .stat b{font-size:18px}.stat span{font-size:11px;color:var(--muted)}
  .controls{padding:12px 22px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  input[type=search]{background:var(--panel);border:1px solid var(--border);color:var(--txt);border-radius:8px;padding:8px 12px;min-width:240px;flex:1;max-width:420px}
  select,button{background:var(--panel);border:1px solid var(--border);color:var(--txt);border-radius:8px;padding:8px 10px;cursor:pointer}
  button:hover{border-color:#586069}
  .seg{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
  .seg button{border:0;border-radius:0;border-right:1px solid var(--border)}
  .seg button:last-child{border-right:0}
  .seg button.on{background:var(--accent);color:#fff}
  .toggle.on{background:var(--accent);color:#fff;border-color:var(--accent)}
  .grid{padding:8px 22px 40px;display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(360px,1fr))}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 16px;
    display:flex;flex-direction:column;gap:10px;transition:border-color .15s;cursor:pointer}
  .card:hover{border-color:#484f58}
  .card.active{border-left:3px solid var(--active)}.card.idle{border-left:3px solid var(--idle)}.card.ended{border-left:3px solid var(--ended)}
  .ctop{display:flex;align-items:flex-start;gap:8px}
  .title{font-weight:600;font-size:14.5px;flex:1;word-break:break-word}
  .title .nick{color:var(--accent)}
  .subtitle{font-weight:400;font-size:12px;color:var(--muted);margin-top:2px;word-break:break-word}
  .badge{font-size:10.5px;padding:2px 8px;border-radius:20px;white-space:nowrap;font-weight:600;text-transform:uppercase;letter-spacing:.4px}
  .badge.active{background:rgba(63,185,80,.16);color:var(--active)}
  .badge.idle{background:rgba(210,153,34,.16);color:var(--idle)}
  .badge.ended{background:rgba(110,118,129,.16);color:var(--ended)}
  .meta{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;color:var(--muted)}
  .meta code{background:var(--panel2);padding:1px 6px;border-radius:5px;color:#adbac7}
  .pbar{height:7px;border-radius:5px;background:var(--panel2);overflow:hidden;display:flex}
  .pbar i{display:block;height:100%}.pbar .d{background:var(--done)}.pbar .p{background:var(--prog)}.pbar .t{background:var(--todo)}
  .tlabel{display:flex;justify-content:space-between;font-size:11.5px;color:var(--muted)}
  .cardbtns{display:flex;gap:8px;flex-wrap:wrap}
  .cardbtns button{font-size:12px;padding:5px 10px}
  .cardbtns button:disabled{opacity:.4;cursor:not-allowed}
  .foot{font-size:11px;color:var(--muted);display:flex;justify-content:space-between;border-top:1px solid var(--border);padding-top:8px;margin-top:2px}
  .empty{padding:60px;text-align:center;color:var(--muted)}
  .pulse{width:8px;height:8px;border-radius:50%;background:var(--active);display:inline-block;margin-right:5px;animation:pulse 1.6s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
  #updated{font-size:11px;color:var(--muted)}
  /* drawer / modal */
  .overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;z-index:50}
  .overlay.show{display:block}
  .drawer{position:fixed;top:0;right:0;height:100%;width:min(720px,96vw);background:var(--bg);
    border-left:1px solid var(--border);z-index:51;transform:translateX(100%);transition:transform .2s;
    display:flex;flex-direction:column}
  .drawer.show{transform:none}
  .dhead{padding:14px 18px;border-bottom:1px solid var(--border);display:flex;align-items:flex-start;gap:10px}
  .dhead h2{font-size:16px;margin:0;flex:1}
  .dhead button{font-size:18px;line-height:1;padding:2px 9px}
  .dbody{padding:16px 18px;overflow:auto;flex:1}
  .dbtns{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
  .dbtns button{font-size:13px;padding:7px 12px}
  .dbtns button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
  .ev{padding:8px 10px;border-radius:8px;margin-bottom:8px;border:1px solid var(--border);background:var(--panel)}
  .ev .who{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:3px}
  .ev.user{border-left:3px solid var(--accent)}
  .ev.assistant{border-left:3px solid var(--prog)}
  .ev.tool{border-left:3px solid var(--idle)}
  .ev pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace}
  .ev.tool .who b{color:var(--idle)}
  .termwrap{position:fixed;inset:0;z-index:60;background:#0b0f14;display:none;flex-direction:column}
  .termwrap.show{display:flex}
  .termbar{padding:8px 14px;background:var(--panel);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .termbar .strat{font-size:12px;color:var(--muted);flex:1}
  .termbar .strat b{color:var(--txt)}
  #termhost{flex:1;min-height:0}
  .note{font-size:12px;color:var(--muted);background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:8px 12px;margin-bottom:10px}
  .copyrow{display:flex;gap:8px;margin-top:8px}
  .copyrow input{flex:1;background:var(--panel2);border:1px solid var(--border);color:var(--txt);border-radius:6px;padding:6px 8px;font-size:12px}
</style></head>
<body>
<header>
  <h1><span class="dot">●</span> Claude Session Dashboard</h1>
  <div class="stats" id="stats"></div>
</header>
<div class="controls">
  <input id="q" type="search" placeholder="Search title, project, branch, prompt…">
  <div class="seg" id="filter">
    <button data-f="all" class="on">All</button>
    <button data-f="active">Active</button>
    <button data-f="idle">Idle</button>
    <button data-f="ended">Ended</button>
    <button data-f="tasks">With tasks</button>
  </div>
  <button id="hideEnded" class="toggle">Hide ended</button>
  <select id="sort">
    <option value="recent">Sort: recent activity</option>
    <option value="started">Sort: start time</option>
    <option value="tasks">Sort: open tasks</option>
    <option value="msgs">Sort: messages</option>
  </select>
  <span id="updated"></span>
</div>
<div class="grid" id="grid"></div>

<div class="overlay" id="overlay"></div>
<div class="drawer" id="drawer">
  <div class="dhead"><h2 id="dtitle"></h2><button id="dclose">✕</button></div>
  <div class="dbody" id="dbody"></div>
</div>

<div class="termwrap" id="termwrap">
  <div class="termbar">
    <span class="strat" id="termstrat"></span>
    <button id="termclose">Close ✕</button>
  </div>
  <div id="termhost"></div>
</div>

<script>""" + TERM_JS + r"""
const $=s=>document.querySelector(s);
let DATA={sessions:[],summary:{}}, FILTER="all", Q="", HIDE_ENDED=false, ACTIVE_TERM=null;
function withTok(u){const m=location.search.match(/token=([^&]+)/);if(!m)return u;return u+(u.includes('?')?'&':'?')+'token='+m[1];}

function ago(ts){ if(!ts)return"—";const s=Date.now()/1000-ts;
  if(s<60)return Math.floor(s)+"s ago";if(s<3600)return Math.floor(s/60)+"m ago";
  if(s<86400)return Math.floor(s/3600)+"h ago";return Math.floor(s/86400)+"d ago";}
function when(ts){return ts?new Date(ts*1000).toLocaleString():"—";}
function num(n){return n>=1e6?(n/1e6).toFixed(1)+"M":n>=1e3?(n/1e3).toFixed(1)+"k":""+n;}
function esc(s){return(s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function shortModel(m){return(m||"").replace("claude-","").replace(/-\d{8}$/,"");}

function card(s){
  const t=s.tasks,tot=t.total||0,pct=n=>tot?(100*n/tot)+"%":"0%";
  const bar=tot?`<div class="pbar"><i class="d" style="width:${pct(t.completed)}"></i><i class="p" style="width:${pct(t.in_progress)}"></i><i class="t" style="width:${pct(t.pending)}"></i></div>
    <div class="tlabel"><span>${t.completed}/${tot} done${t.in_progress?` · ${t.in_progress} in progress`:''}</span><span>${t.pending} pending</span></div>`:'';
  const live=s.state==='active'?'<span class="pulse"></span>':'';
  const att=s.attachable;
  const name=s.nickname?`${live}<span class="nick">🏷 ${esc(s.nickname)}</span><div class="subtitle">${esc(s.title)}</div>`:`${live}${esc(s.title)}`;
  return `<div class="card ${s.state}" data-sid="${s.session_id}">
    <div class="ctop"><div class="title">${name}</div><span class="badge ${s.state}">${s.live_status||s.state}</span></div>
    <div class="meta">
      ${s.project?`<span>📁 <code>${esc(s.project)}</code></span>`:''}
      ${s.git_branch?`<span>⎇ <code>${esc(s.git_branch)}</code></span>`:''}
      <span>🆔 <code>${s.short_id}</code></span>${s.pid?`<span>pid ${s.pid}</span>`:''}
      ${s.models&&s.models.length?`<span>🤖 ${esc(s.models.map(shortModel).join(', '))}</span>`:''}
    </div>${bar}
    <div class="cardbtns">
      <button class="act-rename" data-sid="${s.session_id}">✏ Rename</button>
      <button class="act-ctx" data-sid="${s.session_id}">👁 Context</button>
      <button class="act-term" data-sid="${s.session_id}" ${att?'':'disabled'} title="${att?'Attach to live terminal':'Session not live'}">🖥 Connect</button>
      <button class="act-remote" data-sid="${s.session_id}" ${att?'':'disabled'} title="${att?'Make remote-control link':'Session not live'}">📡 Remote</button>
    </div>
    <div class="foot"><span>💬 ${s.user_msgs}/${s.assistant_msgs} · ↑${num(s.input_tokens)} ↓${num(s.output_tokens)} tok</span>
      <span title="${when(s.last_activity)}">${ago(s.last_activity)}</span></div>
  </div>`;
}

function render(){
  const sum=DATA.summary||{};
  $("#stats").innerHTML=[["total",sum.total,"sessions"],["active",sum.active,"active"],
    ["idle",sum.idle,"idle"],["ended",sum.ended,"ended"],["tasks",sum.in_progress_tasks,"tasks running"]]
    .map(([k,v,l])=>`<div class="stat"><b>${v??0}</b><span>${l}</span></div>`).join("");
  let list=DATA.sessions.slice();
  if(HIDE_ENDED) list=list.filter(s=>s.state!=='ended');
  if(FILTER==="tasks") list=list.filter(s=>s.tasks.total>0);
  else if(FILTER!=="all") list=list.filter(s=>s.state===FILTER);
  if(Q){const q=Q.toLowerCase();
    list=list.filter(s=>[s.nickname,s.title,s.project,s.git_branch,s.first_prompt,s.last_prompt,s.short_id].some(x=>(x||"").toLowerCase().includes(q)));}
  const sort=$("#sort").value;
  list.sort((a,b)=>{if(sort==="started")return(b.started_at||0)-(a.started_at||0);
    if(sort==="tasks")return(b.tasks.in_progress+b.tasks.pending)-(a.tasks.in_progress+a.tasks.pending);
    if(sort==="msgs")return b.assistant_msgs-a.assistant_msgs;return(b.last_activity||0)-(a.last_activity||0);});
  $("#grid").innerHTML=list.length?list.map(card).join(""):`<div class="empty">No sessions match.</div>`;
  $("#updated").textContent="updated "+new Date().toLocaleTimeString();
}

async function load(){
  try{ const r=await fetch(withTok("/api/sessions")); DATA=await r.json(); render(); }
  catch(e){ $("#updated").textContent="fetch error: "+e; }
}

// ---- context drawer ----
async function openContext(sid){
  const s=DATA.sessions.find(x=>x.session_id===sid)||{};
  $("#dtitle").textContent=s.nickname||s.title||sid;
  $("#dbody").innerHTML="<p style='color:var(--muted)'>loading…</p>";
  showDrawer(true);
  const att=s.attachable;
  let html=`<div class="dbtns">
    <button onclick="rename('${sid}')">✏ Rename</button>
    <button class="primary" onclick="connectTerm('${sid}')" ${att?'':'disabled'}>🖥 Connect terminal</button>
    <button onclick="makeRemote('${sid}')" ${att?'':'disabled'}>📡 Remote control</button></div>`;
  if(s.cwd) html+=`<div class="note">📁 ${esc(s.cwd)} &nbsp; ⎇ ${esc(s.git_branch||'')}</div>`;
  try{
    const ctx=await(await fetch(withTok('/api/session/'+sid+'/context'))).json();
    if(ctx.doing) html+=`<div class="note">▶ <b>Doing:</b> ${esc(ctx.doing)}</div>`;
    const evs=(ctx.events||[]);
    if(!evs.length) html+="<p style='color:var(--muted)'>No recent activity recorded.</p>";
    html+=evs.map(e=>{
      if(e.k==='tool') return `<div class="ev tool"><div class="who"><b>⚙ ${esc(e.name||'tool')}</b></div><pre>${esc(e.brief||'')}</pre></div>`;
      return `<div class="ev ${e.k}"><div class="who">${e.k}</div><pre>${esc(e.text||'')}</pre></div>`;
    }).join("");
  }catch(e){ html+="<p style='color:#f85149'>failed to load context: "+e+"</p>"; }
  $("#dbody").innerHTML=html;
}
function showDrawer(v){ $("#overlay").classList.toggle("show",v); $("#drawer").classList.toggle("show",v); }

async function rename(sid){
  const s=DATA.sessions.find(x=>x.session_id===sid)||{};
  const v=prompt("Nickname for this session (leave blank to clear):", s.nickname||"");
  if(v===null) return;
  await fetch(withTok('/api/session/'+sid+'/nickname'),{method:'POST',body:JSON.stringify({nickname:v.trim()})});
  load();
}

// ---- terminal ----
async function connectTerm(sid){
  showDrawer(false);
  const r=await fetch(withTok('/api/term/new'),{method:'POST',body:JSON.stringify({sid,mode:'auto'})});
  const j=await r.json();
  if(j.error){ alert("Cannot open terminal: "+j.error); return; }
  $("#termstrat").innerHTML=`<b>${j.strategy.toUpperCase()}</b> — ${esc(j.note)} <code style="color:#8b949e">${esc(j.cmd)}</code>`;
  $("#termwrap").classList.add("show");
  if(ACTIVE_TERM) ACTIVE_TERM.close();
  ACTIVE_TERM=await startTerm("termhost", j.term_id, st=>{});
}
function closeTerm(){ if(ACTIVE_TERM){ACTIVE_TERM.close();ACTIVE_TERM=null;} $("#termwrap").classList.remove("show"); }

async function makeRemote(sid){
  const r=await fetch(withTok('/api/remote'),{method:'POST',body:JSON.stringify({sid,mode:'auto'})});
  const j=await r.json();
  if(j.error){ alert("Cannot create remote control: "+j.error); return; }
  let html=`<div class="dbtns"><button class="primary" onclick="connectTerm('${sid}')">🖥 Open here</button></div>
    <div class="note">📡 <b>Remote-control link</b> (${j.strategy}). Open it on another device to drive this session.</div>`;
  if(j.warning) html+=`<div class="note" style="border-color:var(--idle);color:var(--idle)">⚠ ${esc(j.warning)}</div>`;
  html+=`<div class="copyrow"><input id="rurl" value="${esc(j.remote_url)}" readonly>
    <button onclick="navigator.clipboard.writeText(document.getElementById('rurl').value);this.textContent='copied'">copy</button></div>
    <p style="font-size:12px;color:var(--muted);margin-top:10px">Anyone with this link + token can type into the session. Treat it like a password.</p>`;
  $("#dtitle").textContent="Remote control";
  $("#dbody").innerHTML=html; showDrawer(true);
}

// ---- events ----
$("#q").addEventListener("input",e=>{Q=e.target.value;render();});
$("#sort").addEventListener("change",render);
$("#hideEnded").addEventListener("click",function(){HIDE_ENDED=!HIDE_ENDED;this.classList.toggle("on",HIDE_ENDED);render();});
document.querySelectorAll("#filter button").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll("#filter button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on");FILTER=b.dataset.f;render();}));
$("#dclose").addEventListener("click",()=>showDrawer(false));
$("#overlay").addEventListener("click",()=>showDrawer(false));
$("#termclose").addEventListener("click",closeTerm);
$("#grid").addEventListener("click",e=>{
  const b=e.target.closest("button"); const card=e.target.closest(".card");
  if(b){ e.stopPropagation(); const sid=b.dataset.sid;
    if(b.classList.contains("act-rename")) rename(sid);
    else if(b.classList.contains("act-ctx")) openContext(sid);
    else if(b.classList.contains("act-term")) connectTerm(sid);
    else if(b.classList.contains("act-remote")) makeRemote(sid);
    return; }
  if(card) openContext(card.dataset.sid);
});
load(); setInterval(load,5000);
</script>
</body></html>"""

REMOTE_PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude Remote Terminal</title>""" + XTERM_HEAD + r"""
<style>
  body{margin:0;background:#0b0f14;color:#e6edf3;font:14px -apple-system,Segoe UI,Roboto,sans-serif;height:100vh;display:flex;flex-direction:column}
  .bar{padding:8px 14px;background:#161b22;border-bottom:1px solid #30363d;font-size:13px}
  #host{flex:1;min-height:0}
</style></head><body>
<div class="bar">📡 Remote terminal — <span id="st">connecting…</span></div>
<div id="host"></div>
<script>""" + TERM_JS + r"""
const id=(location.search.match(/id=([^&]+)/)||[])[1];
if(!id){ document.getElementById('st').textContent='missing terminal id'; }
else startTerm('host', id, s=>{document.getElementById('st').textContent=s;}).then(t=>{document.getElementById('st').textContent='connected';});
</script></body></html>"""


def main():
    global REQUIRE_TOKEN, BASE_TOKEN
    ap = argparse.ArgumentParser(description="Claude session dashboard")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--remote", action="store_true",
                    help="bind 0.0.0.0 and require a token for non-local access")
    ap.add_argument("--token", default=None,
                    help="token required for remote access (auto-generated if omitted)")
    args = ap.parse_args()
    host = args.host
    if args.remote:
        host = "0.0.0.0"
    if host not in ("127.0.0.1", "localhost") or args.token:
        REQUIRE_TOKEN = True
        BASE_TOKEN = args.token or base64.urlsafe_b64encode(os.urandom(12)).decode().rstrip("=")
    srv = ThreadingHTTPServer((host, args.port), Handler)
    shown = host if host != "0.0.0.0" else _lan_ip()
    print(f"Claude session dashboard: http://{shown}:{args.port}/")
    print(f"  scanning: {PROJECTS_DIR}")
    if REQUIRE_TOKEN:
        print(f"  remote enabled — token: {BASE_TOKEN}")
        print(f"  remote URL: http://{shown}:{args.port}/?token={BASE_TOKEN}")
    print("  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
