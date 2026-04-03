#!/usr/bin/env python3
"""
全链路最小复现脚本 —— 模拟 AgentSdkEngine → wrap → rollout → Docker 的完整调用链。
逐层诊断为什么外部 LiteLLM 没有收到请求。

Usage:
    python test_rollout_e2e.py                           # 默认 proxy http://127.0.0.1:4000/v1
    python test_rollout_e2e.py --proxy http://10.0.0.1:4000/v1
    python test_rollout_e2e.py --layer all               # 运行所有层（默认）
    python test_rollout_e2e.py --layer docker             # 只测 Docker
    python test_rollout_e2e.py --layer network            # 只测容器网络
    python test_rollout_e2e.py --layer rollout            # 只测 rollout 函数
    python test_rollout_e2e.py --layer wrap               # 模拟 wrap_with_session_context
    python test_rollout_e2e.py --layer proxy              # 测 proxy HTTP
    python test_rollout_e2e.py --layer container_run      # 实际启动 rllm-openhands 容器

诊断层：
    1. Docker     ─ Docker daemon + image 是否存在？
    2. Network    ─ 容器内能否 TCP 连通 host proxy？
    3. Proxy      ─ proxy HTTP /v1/models 是否返回正常？
    4. URL Build  ─ 构造的 proxied URL 是否正确？
    5. Wrap       ─ wrap_with_session_context → rollout 参数传递是否正确？
    6. Rollout    ─ 调用 rollout() 并捕获完整输出
    7. Container  ─ 实际启动容器并捕获完整 stdout/stderr
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def banner(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{RESET}\n")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓ {msg}{RESET}")


def fail(msg: str) -> None:
    print(f"  {RED}✗ {msg}{RESET}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠ {msg}{RESET}")


def info(msg: str) -> None:
    print(f"  {msg}")


# ─────────────────────────── Layer 1: Docker ────────────────────────────

def check_docker(image: str = "rllm-openhands") -> bool:
    banner("Layer 1: Docker 环境检查")
    all_ok = True

    # Docker daemon
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        if r.returncode == 0:
            ok("Docker daemon 可用")
        else:
            fail("Docker daemon 不可用")
            info(f"  stderr: {r.stderr.decode()[:200]}")
            all_ok = False
    except FileNotFoundError:
        fail("docker 命令未找到 (未安装或不在 PATH)")
        all_ok = False
    except subprocess.TimeoutExpired:
        fail("docker info 超时")
        all_ok = False

    # Image existence
    try:
        r = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", image],
            capture_output=True, timeout=10,
        )
        lines = r.stdout.decode().strip().splitlines()
        if lines:
            ok(f"镜像 '{image}' 存在: {lines[0]}")
        else:
            fail(f"镜像 '{image}' 不存在！需要先 build:")
            info(textwrap.dedent(f"""\
                docker build -t {image} \\
                    -f examples/openhands-sdk/workspace/Dockerfile \\
                    examples/openhands-sdk/workspace
            """))
            all_ok = False
    except Exception as e:
        fail(f"检查镜像失败: {e}")
        all_ok = False

    # Quick container smoke test
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", image, "python", "-c", "print('SMOKE_OK')"],
            capture_output=True, timeout=30,
        )
        out = r.stdout.decode().strip()
        if "SMOKE_OK" in out:
            ok("容器冒烟测试 OK (能启动并执行 Python)")
        else:
            fail(f"容器冒烟测试失败 (exit={r.returncode})")
            info(f"  stdout: {out[:300]}")
            info(f"  stderr: {r.stderr.decode()[:300]}")
            all_ok = False
    except subprocess.TimeoutExpired:
        fail("容器冒烟测试超时 (30s)")
        all_ok = False
    except Exception as e:
        fail(f"容器冒烟测试异常: {e}")
        all_ok = False

    # Test if entrypoint.py imports work inside container
    test_import = (
        "import sys; sys.argv=['test']; "
        "from openhands.sdk import LLM, Agent, Conversation; "
        "print('IMPORTS_OK')"
    )
    try:
        r = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "python", image, "-c", test_import],
            capture_output=True, timeout=30,
        )
        out = r.stdout.decode().strip()
        if "IMPORTS_OK" in out:
            ok("容器内 OpenHands SDK 导入正常")
        else:
            fail("容器内 OpenHands SDK 导入失败")
            info(f"  stdout: {out[:300]}")
            info(f"  stderr: {r.stderr.decode()[:300]}")
            all_ok = False
    except Exception as e:
        fail(f"测试导入异常: {e}")
        all_ok = False

    return all_ok


# ─────────────────────── Layer 2: Container Network ─────────────────────

def check_network(proxy_url: str) -> bool:
    banner("Layer 2: 容器 → Host 网络连通性")

    from urllib.parse import urlparse
    parsed = urlparse(proxy_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 4000

    # host.docker.internal resolution
    container_host = "host.docker.internal" if host in ("localhost", "127.0.0.1") else host

    all_ok = True

    # TCP from host
    info(f"从 Host 侧测试 TCP → {host}:{port}")
    try:
        s = socket.create_connection((host, port), timeout=5)
        s.close()
        ok(f"Host → {host}:{port} TCP 连通")
    except Exception as e:
        fail(f"Host → {host}:{port} TCP 失败: {e}")
        all_ok = False

    # TCP from inside container
    tcp_cmd = f"python -c \"import socket; s=socket.create_connection(('{container_host}', {port}), timeout=5); s.close(); print('TCP_OK')\""
    try:
        r = subprocess.run(
            ["docker", "run", "--rm",
             "--add-host", "host.docker.internal:host-gateway",
             "--entrypoint", "python",
             "rllm-openhands",
             "-c",
             f"import socket; s=socket.create_connection(('{container_host}', {port}), timeout=5); s.close(); print('TCP_OK')"],
            capture_output=True, timeout=15,
        )
        if "TCP_OK" in r.stdout.decode():
            ok(f"容器内 → {container_host}:{port} TCP 连通")
        else:
            fail(f"容器内 → {container_host}:{port} TCP 不通")
            info(f"  stderr: {r.stderr.decode()[:200]}")
            warn("可能原因: 防火墙、Docker 网络配置、proxy 未绑定 0.0.0.0")
            all_ok = False
    except Exception as e:
        fail(f"容器网络测试异常: {e}")
        all_ok = False

    return all_ok


# ─────────────────────── Layer 3: Proxy HTTP ────────────────────────────

def check_proxy(proxy_url: str) -> bool:
    banner("Layer 3: LiteLLM Proxy HTTP 检查")
    all_ok = True

    # /v1/models
    base = proxy_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    models_url = f"{base.rstrip('/v1')}/v1/models"

    info(f"GET {models_url}")
    try:
        req = urllib.request.Request(models_url, headers={"Authorization": "Bearer EMPTY"})
        resp = urllib.request.urlopen(req, timeout=5)
        body = json.loads(resp.read().decode())
        model_ids = [m.get("id", "?") for m in body.get("data", [])]
        if model_ids:
            ok(f"Proxy 返回 {len(model_ids)} 个模型: {model_ids[:3]}")
        else:
            warn("Proxy 返回 0 个模型！可能需要 /admin/reload 加载配置")
            all_ok = False
    except Exception as e:
        fail(f"Proxy /v1/models 请求失败: {e}")
        all_ok = False

    # /v1/chat/completions (简单测试)
    chat_url = f"{base}/chat/completions"
    chat_body = json.dumps({
        "model": os.environ.get("OPENHANDS_MODEL_NAME", "openai/openhands-model"),
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
    }).encode()
    info(f"POST {chat_url} (无 meta slug)")
    try:
        req = urllib.request.Request(
            chat_url, data=chat_body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        body = json.loads(resp.read().decode())
        ok(f"Chat completion 返回正常: {str(body.get('choices', [{}])[0].get('message', {}).get('content', ''))[:50]}")
    except Exception as e:
        fail(f"Chat completion 失败: {e}")
        warn("如果是 404/500, 检查 model_list 配置和 vLLM 后端")
        all_ok = False

    # With meta slug
    from openhands_agent import _build_proxied_base_url, _encode_metadata_slug
    test_meta = {"session_uids": ["test_uid_123"], "session_name": "test:0:1"}
    slugged = _build_proxied_base_url(base, test_meta)
    slugged_chat = f"{slugged}/chat/completions"
    info(f"POST {slugged_chat[:80]}... (带 meta slug)")
    try:
        req = urllib.request.Request(
            slugged_chat, data=chat_body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        ok("带 meta slug 的 chat completion 返回正常")
    except urllib.error.HTTPError as e:
        fail(f"带 meta slug 的请求失败 (HTTP {e.code})")
        if e.code == 404:
            warn("404 说明 proxy 不识别 /meta/<slug>/ 路径")
            warn("你启动的是 rllm 的 litellm_server 还是原生 LiteLLM？")
            warn("需要: python -m rllm.sdk.proxy.litellm_server --host 0.0.0.0 --port 4000")
        all_ok = False
    except Exception as e:
        fail(f"带 meta slug 的请求异常: {e}")
        all_ok = False

    return all_ok


# ─────────────── Layer 4: URL 构造验证 ──────────────────────────────────

def check_url_build(proxy_url: str) -> bool:
    banner("Layer 4: URL 构造验证")
    from openhands_agent import (
        _build_proxied_base_url,
        _to_container_url,
        _proxy_slug_metadata_for_agent_sdk,
    )

    info(f"原始 proxy_url: {proxy_url}")

    # Simulate slug with explicit UIDs
    meta = {"session_uids": ["ctx_abcdef1234567890"], "session_name": "task1:0:1"}
    proxied = _build_proxied_base_url(proxy_url, meta)
    info(f"加 slug 后:     {proxied}")

    container_url = _to_container_url(proxied)
    info(f"容器内 URL:     {container_url}")

    # Verify slug can be decoded
    from rllm.sdk.proxy.metadata_slug import extract_metadata_from_path
    from urllib.parse import urlparse
    parsed = urlparse(container_url)
    result = extract_metadata_from_path(parsed.path)
    if result:
        clean_path, decoded_meta = result
        ok(f"Slug 解码成功: session_name={decoded_meta.get('session_name')}")
        ok(f"清理后路径: {clean_path}")
    else:
        fail("Slug 解码失败！URL 构造有问题")
        return False

    # Test _proxy_slug_metadata_for_agent_sdk with no context
    info("\n  测试 _proxy_slug_metadata_for_agent_sdk (无 session context):")
    meta2, label2 = _proxy_slug_metadata_for_agent_sdk()
    warn(f"无 context 时回退到随机 UUID: {label2}")
    info(f"  metadata: {meta2}")

    ok("URL 构造逻辑正确")
    return True


# ──────────── Layer 5: wrap_with_session_context 模拟 ────────────────────

def check_wrap(proxy_url: str) -> bool:
    banner("Layer 5: wrap_with_session_context 模拟")

    from rllm.sdk.session.base import wrap_with_session_context
    from rllm.sdk.session import get_active_session_uids, get_current_session_name

    captured: dict[str, Any] = {}

    def fake_rollout(*args: Any, **kwargs: Any):
        """捕获 rollout 收到的所有参数"""
        captured["args"] = args
        captured["kwargs"] = dict(kwargs)
        captured["session_uids"] = list(get_active_session_uids())
        captured["session_name"] = get_current_session_name()
        return [{"name": "test", "steps": [], "reward": 0.0}]

    wrapped = wrap_with_session_context(fake_rollout, tracer_service_name="test")

    # Simulate what AgentSdkEngine does
    task = {"instruction": "Test instruction", "scenario": "npu_operator"}
    task_id = "test-task-id-1234"
    metadata = {"session_name": f"{task_id}:0:1", "task": task}

    info("模拟 AgentSdkEngine 调用:")
    info(f"  metadata = {json.dumps(metadata, ensure_ascii=False)[:100]}...")
    info(f"  **task    = {json.dumps(task, ensure_ascii=False)}")

    # Run in thread pool (same as AgentSdkEngine)
    executor = ThreadPoolExecutor(max_workers=1)
    bound = functools.partial(wrapped, metadata, **task)
    future = executor.submit(bound)
    output, session_uid = future.result(timeout=5)
    executor.shutdown(wait=False)

    info(f"\n  wrap 返回:")
    info(f"  session_uid = {session_uid}")
    info(f"\n  rollout 收到的参数:")
    info(f"  args   = {captured.get('args', '?')}")
    info(f"  kwargs 的 keys = {list(captured.get('kwargs', {}).keys())}")

    # Check session propagation
    uids = captured.get("session_uids", [])
    name = captured.get("session_name")

    if uids:
        ok(f"ContextVar session UIDs 传播正确: {uids}")
    else:
        fail("ContextVar session UIDs 为空！Context 未传播到线程")
        warn("这会导致 proxy slug 使用随机 UUID，SQLite traces 无法关联")

    if name == f"{task_id}:0:1":
        ok(f"ContextVar session name 传播正确: {name}")
    else:
        fail(f"session name 不匹配: 期望 '{task_id}:0:1', 实际 '{name}'")

    # Check if _rllm_proxy_session_uids was injected
    kw = captured.get("kwargs", {})
    if "_rllm_proxy_session_uids" in kw:
        ok(f"_rllm_proxy_session_uids 已注入: {kw['_rllm_proxy_session_uids']}")
    else:
        warn("_rllm_proxy_session_uids 未注入 (wrap 未 patch)")
        info("  ContextVar 传播在 ThreadPoolExecutor 中通常有效,")
        info("  但如果你发现 traces 仍然关联不上, 需要 patch base.py")

    if "_rllm_proxy_session_name" in kw:
        ok(f"_rllm_proxy_session_name 已注入: {kw['_rllm_proxy_session_name']}")
    else:
        warn("_rllm_proxy_session_name 未注入")

    # Check task fields are present as kwargs
    if "instruction" in kw and "scenario" in kw:
        ok(f"task 字段正确传入: instruction='{kw['instruction'][:40]}...', scenario='{kw['scenario']}'")
    else:
        fail("task 字段未传入 kwargs！")

    return True


# ──────────────── Layer 6: rollout() 直接调用 ───────────────────────────

def check_rollout(proxy_url: str) -> bool:
    banner("Layer 6: rollout() 直接调用 (实际启动 Docker 容器)")

    from openhands_agent import rollout, _proxy_slug_metadata_for_agent_sdk

    # Simulate the context that wrap_with_session_context would set up
    from rllm.sdk.session.contextvar import ContextVarSession

    task = {"instruction": "Test: implement a simple hello_world function.", "scenario": "npu_operator"}

    info(f"task = {json.dumps(task, ensure_ascii=False)}")
    info(f"proxy_url (默认回退) = {proxy_url}")
    info("")

    with ContextVarSession(name="test_task:0:1") as session:
        info(f"Session UID: {session._uid}")
        info(f"Session name: {session.name}")

        meta, label = _proxy_slug_metadata_for_agent_sdk()
        info(f"Slug meta: {meta}")
        info(f"Slug label: {label}")

        info("\n  开始调用 rollout()... (这会启动 Docker 容器)")
        t0 = time.time()
        try:
            result = rollout(**task)
            elapsed = time.time() - t0
            info(f"\n  rollout 完成, 耗时 {elapsed:.2f}s")

            if elapsed < 5.0:
                warn(f"耗时仅 {elapsed:.2f}s — 太快了！Docker 容器可能没有真正运行")
                warn("正常的 OpenHands agent 运行至少需要 30s+")
            else:
                ok(f"耗时 {elapsed:.2f}s — 看起来合理")

            if result:
                traj = result[0]
                info(f"  返回 Trajectory: name={traj.name}, reward={traj.reward}")
                info(f"  steps count: {len(traj.steps)}")
                if traj.reward == 0.0 and elapsed < 5.0:
                    fail("reward=0 + 快速完成 → 容器大概率启动失败")
            else:
                fail("rollout 返回空结果")

        except Exception as e:
            elapsed = time.time() - t0
            fail(f"rollout 异常 ({elapsed:.2f}s): {e}")
            traceback.print_exc()

    return True


# ──────────────── Layer 7: 实际容器运行 ─────────────────────────────────

def check_container_run(proxy_url: str) -> bool:
    banner("Layer 7: 实际启动 rllm-openhands 容器 (详细日志)")

    import tempfile
    from openhands_agent import (
        _setup_npu_operator_workspace,
        _build_proxied_base_url,
        _to_container_url,
        _OPENHANDS_IMAGE,
        _MODEL_NAME,
    )

    task = {"instruction": "Test: implement hello_world.", "scenario": "npu_operator"}
    workspace = _setup_npu_operator_workspace(task)

    meta = {"session_uids": ["test_debug_uid"], "session_name": "debug:0:1"}
    proxied_url = _build_proxied_base_url(proxy_url, meta)
    container_url = _to_container_url(proxied_url)

    info(f"Workspace:     {workspace}")
    info(f"Container URL: {container_url}")
    info(f"Image:         {_OPENHANDS_IMAGE}")
    info(f"Model:         {_MODEL_NAME}")

    # List workspace contents
    for f in os.listdir(workspace):
        info(f"  workspace/{f}")
    tools_dir = os.path.join(workspace, "tools")
    if os.path.isdir(tools_dir):
        for f in os.listdir(tools_dir):
            info(f"  workspace/tools/{f}")

    cmd = [
        "docker", "run", "--rm",
        "-e", f"LLM_BASE_URL={container_url}",
        "-e", "LLM_API_KEY=EMPTY",
        "-e", f"LLM_MODEL={_MODEL_NAME}",
        "-e", "NPU_OPERATOR_TASK=1",
        "-e", "WORKSPACE_BASE=/opt/workspace",
        "-e", "MAX_ITERATIONS=2",  # 限制迭代数
        "-v", f"{workspace}:/opt/workspace",
        "--add-host", "host.docker.internal:host-gateway",
        _OPENHANDS_IMAGE,
    ]

    info(f"\n  Docker 命令:\n  {' '.join(cmd[:6])} \\\n    {'   '.join(cmd[6:])}")
    info("\n  启动容器中...\n")

    try:
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        elapsed = time.time() - t0

        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")

        info(f"  退出码: {result.returncode}")
        info(f"  耗时:   {elapsed:.2f}s")
        info(f"\n  ──── STDOUT ({len(stdout)} chars) ────")
        print(stdout[:2000] if stdout else "  (空)")
        info(f"\n  ──── STDERR ({len(stderr)} chars) ────")
        print(stderr[:2000] if stderr else "  (空)")

        if result.returncode != 0:
            fail(f"容器退出码 {result.returncode}")
            if "No module named" in stderr:
                warn("Python 模块缺失 — 镜像可能需要重新 build")
            if "LLM_BASE_URL is not set" in (stdout + stderr):
                warn("LLM_BASE_URL 环境变量未传入容器")
            if "No task instruction" in (stdout + stderr):
                warn("TASK_INSTRUCTION 为空且 INSTRUCTIONS.md 读取失败")
        elif elapsed < 3.0:
            warn("完成太快 — 可能没有真正执行 agent step")
        else:
            ok("容器运行完成")

    except subprocess.TimeoutExpired:
        fail("容器运行超时 (120s)")
    except Exception as e:
        fail(f"容器运行异常: {e}")
    finally:
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)

    return True


# ─────────────────────── Main ───────────────────────────────────────────

LAYERS = {
    "docker": check_docker,
    "network": check_network,
    "proxy": check_proxy,
    "url": check_url_build,
    "wrap": check_wrap,
    "rollout": check_rollout,
    "container_run": check_container_run,
}


def main():
    parser = argparse.ArgumentParser(description="全链路 rollout 诊断")
    parser.add_argument("--proxy", default="http://127.0.0.1:4000/v1", help="LiteLLM proxy URL")
    parser.add_argument("--layer", default="all",
                        choices=["all"] + list(LAYERS.keys()),
                        help="指定测试层")
    parser.add_argument("--image", default="rllm-openhands", help="Docker 镜像名")
    args = parser.parse_args()

    banner(f"rllm-openhands 全链路诊断  proxy={args.proxy}")

    if args.layer == "all":
        layers_to_run = list(LAYERS.keys())
    else:
        layers_to_run = [args.layer]

    results = {}
    for name in layers_to_run:
        fn = LAYERS[name]
        try:
            if name == "docker":
                results[name] = fn(args.image)
            elif name in ("network", "proxy", "url", "wrap", "rollout", "container_run"):
                results[name] = fn(args.proxy)
            else:
                results[name] = fn(args.proxy)
        except Exception as e:
            fail(f"Layer '{name}' 异常: {e}")
            traceback.print_exc()
            results[name] = False

        # 如果基础层失败，提前结束
        if name in ("docker",) and not results[name]:
            warn("Docker 基础检查失败，后续层测试将跳过")
            break

    banner("诊断总结")
    for name, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {name:20s}  {status}")

    print()
    if all(results.values()):
        ok("所有测试层通过")
    else:
        failed = [n for n, v in results.items() if not v]
        fail(f"失败的层: {', '.join(failed)}")
        print()
        info("常见根因排查:")
        info("  1. docker 层失败 → 镜像未 build 或 Docker daemon 未启动")
        info("  2. network 层失败 → 容器无法连通 host proxy (防火墙/绑定地址)")
        info("  3. proxy 层失败  → LiteLLM 配置问题或未使用 rllm 的 litellm_server")
        info("  4. wrap 层失败   → session context 未正确传播到 rollout")
        info("  5. container 失败 → 查看 STDOUT/STDERR 详细日志")


if __name__ == "__main__":
    main()
