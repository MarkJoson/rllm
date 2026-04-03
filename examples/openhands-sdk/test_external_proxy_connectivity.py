#!/usr/bin/env python3
"""
快速诊断 external LiteLLM 代理是否可达、OpenHands 同款 /meta/ slug 路径是否通。

不启动 Ray / 训练；适合反复调试「没有请求打到代理」的问题。

用法（在任意目录，需能 import 到本文件或复制运行）::

    cd /path/to/rllm/repo
    python examples/openhands-sdk/test_external_proxy_connectivity.py \\
        --base-url http://127.0.0.1:4000/v1 \\
        --model openai/your-model-name

若 OpenHands 在 Docker 内访问宿主机代理，宿主机侧应监听 0.0.0.0；
容器内 URL 一般为 http://host.docker.internal:4000/v1 ，可用 --rewrite-host 模拟。

external 模式注意：训练进程会先 POST /admin/reload 注入 model_list；
若你手搓启动的代理从未 reload，/v1/chat/completions 可能 4xx —— 先跑一次训练让 reload 发生，
或自行 curl /admin/reload（需 admin token）。
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse, urlunparse

_SLUG_PREFIX = "rllm1:"


def _encode_slug(metadata: dict) -> str:
    body = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    enc = base64.urlsafe_b64encode(body.encode("utf-8")).rstrip(b"=")
    return f"{_SLUG_PREFIX}{enc.decode('ascii')}"


def _build_meta_url(base_v1_url: str, metadata: dict) -> str:
    """与 openhands_agent._build_proxied_base_url 一致。"""
    slug = _encode_slug(metadata)
    parsed = urlparse(base_v1_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    new_path = f"{path}/meta/{slug}/v1"
    if not new_path.startswith("/"):
        new_path = "/" + new_path
    return urlunparse(parsed._replace(path=new_path))


def _to_host_docker_internal(url: str) -> str:
    p = urlparse(url)
    h = p.hostname or ""
    if h in ("localhost", "127.0.0.1"):
        netloc = p.netloc.replace(h, "host.docker.internal", 1)
        return urlunparse(p._replace(netloc=netloc))
    return url


def _tcp_probe(host: str, port: int, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True, "ok"
    except OSError as e:
        return False, str(e)


def _http_json(
    method: str,
    url: str,
    payload: dict | None,
    headers: dict[str, str],
    timeout: float = 60.0,
) -> tuple[int, str, dict | None]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, raw, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw, None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        return e.code, raw, None
    except urllib.error.URLError as e:
        return -1, str(e.reason), None


def main() -> int:
    p = argparse.ArgumentParser(description="Probe external LiteLLM (OpenHands-compatible slug path).")
    p.add_argument(
        "--base-url",
        default="http://127.0.0.1:4000/v1",
        help="与训练一致，例如 http://HOST:PORT/v1",
    )
    p.add_argument("--model", required=True, help="LiteLLM model_list 里注册的 model name")
    p.add_argument(
        "--rewrite-host",
        action="store_true",
        help="把 URL 里的 localhost/127.0.0.1 换成 host.docker.internal（看容器视角）",
    )
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()

    base = args.base_url.strip()
    if args.rewrite_host:
        base = _to_host_docker_internal(base)
        print(f"[info] rewrite base-url -> {base}")

    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    ok, msg = _tcp_probe(host, port)
    print(f"[1] TCP {host}:{port} -> {'OK' if ok else 'FAIL'} ({msg})")
    if not ok:
        print("    代理未监听或防火墙/安全组拦截；external 请确认 uvicorn 绑定 0.0.0.0。")
        return 1

    headers = {"Content-Type": "application/json"}
    chat_body = {
        "model": args.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 4,
    }

    plain_chat = urlunparse(urlparse(base)._replace(path="/v1/chat/completions"))
    code, text, _ = _http_json("POST", plain_chat, chat_body, headers, timeout=args.timeout)
    print(f"[2] POST {plain_chat} (no meta slug) -> HTTP {code}")
    if code != 200:
        print(f"    body (truncated): {text[:800]}")
        if code in (401, 403):
            print("    检查 API key / 代理鉴权。")
        if code == 404 or "model" in text.lower():
            print("    常见原因：未 /admin/reload 或 model 名与 model_list 不一致。")
    else:
        print("    OK（说明代理与后端至少有一条通路）")

    meta = {
        "session_uids": ["ctx_debugpytest000"],
        "session_name": "debug-task:0:1",
    }
    meta_url = _build_meta_url(base, meta)
    code2, text2, _ = _http_json("POST", meta_url, chat_body, headers, timeout=args.timeout)
    print(f"[3] POST {meta_url[:100]}... (OpenHands 同款 meta 路径) -> HTTP {code2}")
    if code2 != 200:
        print(f"    body (truncated): {text2[:800]}")
        print("    若 [2] 成功而 [3] 失败：检查 MetadataRoutingMiddleware 是否挂在 litellm_app。")
    else:
        print("    OK（slug 路径与中间件正常）")

    print()
    print("容器内 OpenHands 使用 host.docker.internal；若在 Linux 无该域名，需 --add-host=host.docker.internal:host-gateway。")
    print("训练侧 external：确认 rllm.sdk.proxy.host/port 与手启代理一致；AgentSdkEngine 会把 base_url 传给 rollout。")
    return 0 if code == 200 else 2


if __name__ == "__main__":
    sys.exit(main())
