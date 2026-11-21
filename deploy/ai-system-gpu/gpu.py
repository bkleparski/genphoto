#!/usr/bin/env python3
"""Unified GPU lease coordinator for s144 (RTX 3060 12 GB).

Serializes GPU access across:
- v1 TCP clients on 10.156.240.1:11435 (VM ai-system) and 127.0.0.1:11435 (pi, hermes-gpu-proxy)
- v2 Unix socket clients on /run/ai-system-gpu/lease.sock (GenPhoto: forge, krea2, comfyui)

Guarantees that:
1. Only one GPU owner holds a lease at any time.
2. Before READY: ollama-embed-hot is stopped, qwen3-embedding is unloaded, any stray
   GenPhoto user backends are stopped, and nvidia-smi confirms free VRAM.
3. On FINISH, ABORT, client disconnect/crash, timeout, or coordinator restart:
   all GenPhoto backends are stopped, Ollama chat models are unloaded, VRAM drains,
   ollama-embed-hot is started, and qwen3-embedding:latest is verified back in GPU
   before releasing the lock for the next client.
"""

import grp
import json
import os
from pathlib import Path
import socket
import socketserver
import subprocess
import sys
import threading
import time
import uuid
from urllib.request import Request, urlopen

# TCP v1: listen address -> only permitted client IP (backwards compatible)
CLIENTS = {"10.156.240.1": "10.156.240.2", "127.0.0.1": "127.0.0.1"}
TCP_PORT = 11435

# Unix v2: local GenPhoto coordinator socket
UNIX_SOCK_PATH = Path(os.environ.get("AI_GPU_SOCK", "/run/ai-system-gpu/lease.sock"))
SOCKET_GROUP = os.environ.get("AI_GPU_SOCKET_GROUP", "bartek")

# User-level systemd services owned by GenPhoto (bartek, UID 1000, Linger=yes)
USER_NAME = os.environ.get("AI_GPU_USER", "bartek")
USER_UID = int(os.environ.get("AI_GPU_UID", "1000"))
USER_RUNTIME_DIR = f"/run/user/{USER_UID}"
GENPHOTO_SERVICES = ("forge", "krea2", "comfyui")

# Ollama & embedding keeper
OLLAMA_URL = os.environ.get("AI_GPU_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
KEEPER = "ollama-embed-hot.service"
EMBED = "qwen3-embedding:latest"
CHAT_MODELS = {
    "gemma4-12b-agent:latest",
    "qwen3.5:9b",
    "gemma4-12b-agent-pi:latest",
    "qwen3.5-pi:9b",
    "qwen2.5vl:7b",
    "qwen2.5vl:32b",
}

# VRAM thresholds (RTX 3060 12288 MiB; Xorg/KDE desktop uses ~40-300 MiB)
MIN_FREE_VRAM_MB = int(os.environ.get("AI_GPU_MIN_FREE_MB", "9500"))
EMBED_MIN_FREE_VRAM_MB = int(os.environ.get("AI_GPU_EMBED_MIN_FREE_MB", "7000"))

V2_HEARTBEAT_TIMEOUT_S = 90.0
V2_MIN_BUDGET_S = 60
V2_MAX_BUDGET_S = 14400  # 4h ceiling for large MOSS/all-models batches

marker = Path("/var/lib/ai-system-gpu/restore-keeper")
lock = threading.Lock()
_state_lock = threading.Lock()
_active_lease = None


def log(msg):
    print(msg, flush=True)


def command(*args, check=True, timeout=90):
    return subprocess.run(args, check=check, timeout=timeout, capture_output=True, text=True)


def user_systemctl(*args, check=False, timeout=45):
    cmd = [
        "runuser", "-u", USER_NAME, "--",
        "env",
        f"XDG_RUNTIME_DIR={USER_RUNTIME_DIR}",
        f"DBUS_SESSION_BUS_ADDRESS=unix:path={USER_RUNTIME_DIR}/bus",
        "systemctl", "--user", *args,
    ]
    return subprocess.run(cmd, check=check, timeout=timeout, capture_output=True, text=True)


def active_genphoto_services():
    active = []
    for svc in GENPHOTO_SERVICES:
        try:
            r = user_systemctl("is-active", "--quiet", svc, timeout=10)
            if r.returncode == 0:
                active.append(svc)
        except Exception:
            pass
    return active


def vram_info():
    """Return (used_mb, free_mb, total_mb) from nvidia-smi."""
    r = command(
        "nvidia-smi",
        "--query-gpu=memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
        timeout=10,
    )
    parts = [int(x.strip()) for x in r.stdout.strip().splitlines()[0].split(",")[:3]]
    return parts[0], parts[1], parts[2]


def gpu_compute_procs():
    """Return list of dicts for active CUDA compute processes on GPU."""
    r = command(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
        timeout=10,
    )
    procs = []
    for line in r.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        name = parts[1]
        try:
            mem_mb = int(parts[2])
        except ValueError:
            mem_mb = 0
        cmdline = ""
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\x00", " ").strip()
        except OSError:
            pass
        procs.append({"pid": pid, "name": name, "used_mb": mem_mb, "cmdline": cmdline[:200]})
    return procs


def loaded():
    with urlopen(f"{OLLAMA_URL}/api/ps", timeout=10) as response:
        return {model["name"] for model in json.load(response).get("models", [])}


def unload(name):
    endpoints = ["/api/embed", "/api/generate"] if "embed" in name.lower() else ["/api/generate", "/api/embed"]
    last_err = None
    for ep in endpoints:
        payload = {"model": name, "keep_alive": 0}
        if ep == "/api/embed":
            payload["input"] = ""
        try:
            request = Request(
                f"{OLLAMA_URL}{ep}",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=90) as response:
                response.read()
            break
        except Exception as exc:
            last_err = exc
    else:
        raise RuntimeError(f"Ollama unload failed for {name}: {last_err}")

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if name not in loaded():
            return
        time.sleep(0.5)
    if name in loaded():
        raise RuntimeError(f"Ollama did not confirm unload of {name}")


def stop_all_genphoto_backends(wait_s=35):
    """Stop forge, krea2, comfyui user services and confirm their compute PIDs exit."""
    try:
        user_systemctl("stop", *GENPHOTO_SERVICES, timeout=45)
    except Exception as exc:
        log(f"warn_stop_backends={exc}")

    deadline = time.monotonic() + wait_s
    killed = False
    while time.monotonic() < deadline:
        still_active = active_genphoto_services()
        procs = gpu_compute_procs()
        gen_procs = [
            p for p in procs
            if any(k in (p["cmdline"] + " " + p["name"]).lower()
                   for k in ("forge", "launch.py", "comfyui", "krea-2", "krea2"))
        ]
        if not still_active and not gen_procs:
            return True
        if not killed and time.monotonic() > deadline - (wait_s / 2):
            killed = True
            for svc in still_active or GENPHOTO_SERVICES:
                try:
                    user_systemctl("kill", "-s", "SIGKILL", svc, timeout=10)
                except Exception:
                    pass
            for p in gen_procs:
                try:
                    os.kill(p["pid"], 9)
                except OSError:
                    pass
        time.sleep(1)

    return not active_genphoto_services()


def start_genphoto_backend(svc):
    if svc not in GENPHOTO_SERVICES:
        raise ValueError(f"Unsupported backend service: {svc}")
    others = [s for s in GENPHOTO_SERVICES if s != svc]
    if others:
        user_systemctl("stop", *others, timeout=30)
    user_systemctl("start", svc, check=True, timeout=30)
    log(f"backend_started service={svc}")


def wait_gpu_clean(min_free_mb=MIN_FREE_VRAM_MB, timeout_s=35):
    """Confirm no Ollama models, no compute processes, and free VRAM >= min_free_mb."""
    deadline = time.monotonic() + timeout_s
    last_reason = "vram_check_timeout"
    while True:
        try:
            ollama_models = loaded()
        except Exception as exc:
            return False, f"ollama_unreachable:{exc}"
        if ollama_models:
            last_reason = f"ollama_loaded:{','.join(sorted(ollama_models))}"
        else:
            try:
                _used, free_mb, _total = vram_info()
                procs = gpu_compute_procs()
            except Exception as exc:
                return False, f"nvidia_smi_error:{exc}"
            if procs:
                names = ",".join(f"{p['pid']}:{p['name']}({p['used_mb']}MB)" for p in procs)
                last_reason = f"compute_procs:{names}"
            elif free_mb < min_free_mb:
                last_reason = f"low_vram:{free_mb}MB<{min_free_mb}MB"
            else:
                return True, "ok"
        if time.monotonic() >= deadline:
            return False, last_reason
        time.sleep(1)


def _write_marker(meta):
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(marker)


def _set_active_lease(info):
    global _active_lease
    with _state_lock:
        _active_lease = info


def get_status():
    with _state_lock:
        lease = dict(_active_lease) if _active_lease else None
    try:
        used_mb, free_mb, total_mb = vram_info()
    except Exception:
        used_mb = free_mb = total_mb = None
    try:
        ol = sorted(loaded())
    except Exception:
        ol = []
    return {
        "busy": lease is not None or lock.locked(),
        "owner": lease["owner"] if lease else None,
        "job_id": lease["job_id"] if lease else None,
        "backend": lease["backend"] if lease else None,
        "lease_id": lease["lease_id"] if lease else None,
        "started_at": lease["started_at"] if lease else None,
        "vram_used_mb": used_mb,
        "vram_free_mb": free_mb,
        "vram_total_mb": total_mb,
        "embedding_loaded": EMBED in ol,
        "ollama_models": ol,
        "active_backends": active_genphoto_services(),
    }


def prepare_gpu(owner, job_id="", backend="", lease_id="", budget_s=1800, on_state=None):
    """Prepare exclusive clean GPU for a lease holder (both v1 and v2)."""
    if loaded() - {EMBED}:
        return False, "ollama_chat_active"

    if on_state:
        on_state("unloading_embedding")

    _write_marker({
        "owner": owner,
        "job_id": job_id,
        "backend": backend,
        "lease_id": lease_id,
        "started_at": int(time.time()),
        "budget_s": int(budget_s),
    })

    # Stop keeper so it cannot reload qwen3-embedding mid-job
    command("systemctl", "stop", KEEPER, check=False, timeout=30)

    if EMBED in loaded():
        unload(EMBED)
    if loaded():
        return False, "ollama_not_empty"

    # Ensure no stray GenPhoto backends occupy VRAM
    if active_genphoto_services() or gpu_compute_procs():
        stop_all_genphoto_backends(wait_s=35)

    ok, reason = wait_gpu_clean(MIN_FREE_VRAM_MB, timeout_s=35)
    if not ok:
        return False, reason

    _set_active_lease({
        "owner": owner,
        "job_id": job_id,
        "backend": backend,
        "lease_id": lease_id,
        "started_at": int(time.time()),
        "budget_s": int(budget_s),
    })
    log(f"gpu_ready owner={owner} backend={backend or 'ollama'} job_id={job_id or '-'}")
    return True, "ok"


def keeper_active():
    return command("systemctl", "is-active", "--quiet", KEEPER, check=False, timeout=10).returncode == 0


def cleanup_and_restore(force=False, on_state=None):
    """Idempotent teardown of all GPU workloads and restoration of embedding model."""
    try:
        active_svcs = active_genphoto_services()
        try:
            ol = loaded()
        except Exception:
            ol = set()
        need = force or marker.exists() or bool(active_svcs) or bool(ol - {EMBED}) or (EMBED not in ol) or (not keeper_active())
        if not need:
            return

        _set_active_lease(None)

        # 1. Stop all GenPhoto user services first and wait for GPU release
        if active_svcs or force:
            if on_state:
                on_state("stopping_backend")
            stop_all_genphoto_backends(wait_s=35)

        # 2. Unload any Ollama chat/non-embedding models
        try:
            for name in list(loaded() - {EMBED}):
                unload(name)
        except Exception as exc:
            log(f"warn_ollama_unload={exc}")

        # 3. Wait until enough VRAM is free for embedding (~6.2 GB)
        if on_state:
            on_state("restoring_embedding")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                _u, free_mb, _t = vram_info()
                if free_mb >= EMBED_MIN_FREE_VRAM_MB or EMBED in loaded():
                    break
            except Exception:
                break
            time.sleep(1)

        # 4. Start embedding keeper and confirm qwen3-embedding:latest is loaded
        if not keeper_active():
            command("systemctl", "start", KEEPER, check=True, timeout=60)
        for _ in range(60):
            try:
                if EMBED in loaded():
                    log("embedding_restored")
                    marker.unlink(missing_ok=True)
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("Embedding nie wrócił do pamięci GPU w ciągu 60s")
    finally:
        _set_active_lease(None)


def restore():
    """Backwards-compatible entrypoint for --restore and startup reconciliation."""
    cleanup_and_restore(force=False)


class LeaseV1(socketserver.StreamRequestHandler):
    """Backwards-compatible TCP 11435 handler for VM ai-system, pi, hermes-gpu-proxy."""

    def reply(self, line):
        self.wfile.write((line + "\n").encode())
        self.wfile.flush()

    def handle(self):
        client_ip = self.client_address[0]
        server_ip = self.server.server_address[0]
        if client_ip != CLIENTS.get(server_ip) or not lock.acquire(blocking=False):
            self.reply("BUSY")
            return
        prepared = False
        try:
            self.connection.settimeout(120)
            if marker.exists() or active_genphoto_services():
                cleanup_and_restore(force=True)
            self.reply("SEARCH")
            if self.rfile.readline(16) != b"GENERATE\n":
                return
            lease_id = uuid.uuid4().hex[:12]
            ok, reason = prepare_gpu(
                owner=f"v1:{client_ip}",
                backend="ollama",
                lease_id=lease_id,
                budget_s=1800,
            )
            prepared = True
            if not ok:
                log(f"v1_gpu_busy client={client_ip} reason={reason}")
                self.reply("GPU_BUSY")
                return
            self.reply("READY")
            self.connection.settimeout(1800)
            self.rfile.readline(16)
        finally:
            try:
                if prepared or marker.exists():
                    cleanup_and_restore(force=True)
            except Exception as exc:
                log(f"v1_restore_error={exc}")
            finally:
                lock.release()


class LeaseV2(socketserver.StreamRequestHandler):
    """Unix socket v2 handler for GenPhoto (ACQUIRE / START / STOP / PING / FINISH / ABORT / STATUS / CLEANUP)."""

    def reply(self, line):
        self.wfile.write((line + "\n").encode())
        self.wfile.flush()

    def _safe_reply(self, line):
        try:
            self.reply(line)
        except OSError:
            pass

    def handle(self):
        self.connection.settimeout(15)
        raw = self.rfile.readline(512)
        if not raw:
            return
        line = raw.decode("utf-8", errors="replace").strip()
        parts = line.split()
        if not parts:
            return
        cmd = parts[0].upper()

        if cmd == "STATUS":
            self._safe_reply(json.dumps(get_status(), ensure_ascii=False))
            return

        if cmd == "CLEANUP":
            if not lock.acquire(blocking=False):
                st = get_status()
                self._safe_reply(f"BUSY {st.get('owner') or 'busy'}")
                return
            try:
                cleanup_and_restore(force=True)
                self._safe_reply("RESTORED")
            except Exception as exc:
                self._safe_reply(f"ERR {exc}")
            finally:
                lock.release()
            return

        if cmd != "ACQUIRE" or len(parts) < 4:
            self._safe_reply("ERR usage: ACQUIRE <job_id> <backend> <budget_s>")
            return

        job_id = parts[1][:64]
        backend = parts[2].lower()
        try:
            budget_s = max(V2_MIN_BUDGET_S, min(V2_MAX_BUDGET_S, int(parts[3])))
        except ValueError:
            self._safe_reply("ERR invalid_budget")
            return

        if backend not in GENPHOTO_SERVICES:
            self._safe_reply(f"ERR invalid_backend:{backend}")
            return

        if not lock.acquire(blocking=False):
            st = get_status()
            self._safe_reply(f"BUSY {st.get('owner') or 'busy'}")
            return

        lease_id = uuid.uuid4().hex[:12]
        prepared = False
        try:
            self.connection.settimeout(120)
            ok, reason = prepare_gpu(
                owner=f"genphoto:{backend}",
                job_id=job_id,
                backend=backend,
                lease_id=lease_id,
                budget_s=budget_s,
                on_state=lambda s: self._safe_reply(f"STATE {s}"),
            )
            prepared = True
            if not ok:
                log(f"v2_gpu_busy job_id={job_id} backend={backend} reason={reason}")
                self._safe_reply(f"GPU_BUSY {reason}")
                return

            self.reply(f"READY {lease_id}")
            deadline = time.monotonic() + budget_s

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log(f"v2_budget_exceeded job_id={job_id} backend={backend}")
                    break
                self.connection.settimeout(min(V2_HEARTBEAT_TIMEOUT_S, max(1.0, remaining)))
                try:
                    nxt = self.rfile.readline(256)
                except socket.timeout:
                    log(f"v2_heartbeat_timeout job_id={job_id} backend={backend}")
                    break
                except OSError:
                    log(f"v2_socket_error job_id={job_id} backend={backend}")
                    break
                if not nxt:
                    log(f"v2_client_eof job_id={job_id} backend={backend}")
                    break
                req_parts = nxt.decode("utf-8", errors="replace").strip().split()
                if not req_parts:
                    continue
                op = req_parts[0].upper()
                if op == "PING":
                    self.reply("PONG")
                elif op == "START":
                    svc = req_parts[1].lower() if len(req_parts) > 1 else backend
                    if svc != backend:
                        self.reply(f"ERR backend_mismatch:{svc}!={backend}")
                        continue
                    try:
                        start_genphoto_backend(svc)
                        self.reply(f"STARTED {svc}")
                    except Exception as exc:
                        self.reply(f"ERR start_failed:{exc}")
                elif op == "STOP":
                    stop_all_genphoto_backends(wait_s=35)
                    self.reply("STOPPED")
                elif op in ("FINISH", "ABORT"):
                    log(f"v2_{op.lower()} job_id={job_id} backend={backend}")
                    break
                else:
                    self.reply(f"ERR unknown_op:{op}")
        finally:
            try:
                if prepared or marker.exists():
                    cleanup_and_restore(
                        force=True,
                        on_state=lambda s: self._safe_reply(f"STATE {s}"),
                    )
                    self._safe_reply("RESTORED")
            except Exception as exc:
                log(f"v2_restore_error={exc}")
                self._safe_reply(f"ERR restore_failed:{exc}")
            finally:
                lock.release()


class ThreadingUnixStreamServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def _start_unix_server():
    UNIX_SOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if UNIX_SOCK_PATH.exists():
        UNIX_SOCK_PATH.unlink()
    server = ThreadingUnixStreamServer(str(UNIX_SOCK_PATH), LeaseV2)
    try:
        gid = grp.getgrnam(SOCKET_GROUP).gr_gid
        os.chown(UNIX_SOCK_PATH, 0, gid)
        os.chmod(UNIX_SOCK_PATH, 0o660)
    except Exception as exc:
        log(f"warn_socket_perms={exc}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"unix_coordinator_listening path={UNIX_SOCK_PATH}")
    return server


if __name__ == "__main__":
    if "--restore" in sys.argv:
        restore()
        sys.exit(0)

    # Startup reconciliation: ensure clean GPU & hot embedding even after unclean reboot/crash
    try:
        restore()
    except Exception as exc:
        log(f"startup_restore_error={exc}")

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    servers = []
    try:
        servers.append(_start_unix_server())
        for host in CLIENTS:
            server = socketserver.ThreadingTCPServer((host, TCP_PORT), LeaseV1)
            server.daemon_threads = True
            servers.append(server)
        for server in servers[1:-1]:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        servers[-1].serve_forever()
    finally:
        for server in servers:
            server.server_close()
        UNIX_SOCK_PATH.unlink(missing_ok=True)
