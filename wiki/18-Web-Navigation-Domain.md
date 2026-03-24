# Web 导航域 (Web Navigation Domain)

Web 导航域训练 Agent 在浏览器环境中完成网页交互任务。

> 源码参考：[rllm/environments/web/](../rllm/environments/web/)、[examples/webarena/](../examples/webarena/)

## 概述

Web 导航训练让 LLM Agent 学习如何通过控制浏览器完成复杂的 Web 任务，如表单填写、信息检索、在线购物等。

## Agent-Environment 交互

```
┌──────────────┐   action (click/type/nav)  ┌───────────────────┐
│  Web Agent   │─────────────────────────→   │  Browser Env      │
│  (解析LLM    │                             │  (Playwright/     │
│   输出为      │   ←─────────────────────   │   Selenium)       │
│   浏览器动作) │    obs (DOM/screenshot)     │  ├── 页面渲染      │
└──────────────┘                             │  ├── 元素交互      │
                                             │  └── 状态提取      │
                                             └───────────────────┘
```

## 观察空间

| 观察模式 | 说明 | 适用 |
|----------|------|------|
| DOM Tree | 简化的 HTML DOM 树 | 文本 LLM |
| Accessibility Tree | 可访问性树 | 结构化交互 |
| Screenshot | 页面截图 | VLM (多模态) |
| 混合模式 | DOM + Screenshot | VLM + 文本 |

## 动作空间

```python
# 支持的浏览器动作
actions = [
    "click(element_id=123)",          # 点击元素
    "type(element_id=456, text='hello')",  # 输入文本
    "goto(url='https://...')",        # 导航到 URL
    "scroll(direction='down')",       # 滚动页面
    "go_back()",                      # 返回上一页
    "wait(seconds=2)",                # 等待
    "select_option(element_id=789, option='value')",  # 下拉选择
]
```

## 奖励计算

Web 导航任务的奖励通常在所有步完成后计算，基于最终页面状态：

```python
def compute_final_reward(self):
    current_url = self.browser.url
    page_content = self.browser.content()
    
    # URL 匹配
    if self.expected_url and current_url == self.expected_url:
        return 1.0
    
    # 内容匹配
    if self.expected_content in page_content:
        return 1.0
    
    return 0.0
```

## 数据集

| 数据集 | 任务数 | 特点 |
|--------|--------|------|
| WebArena | 812 | 真实网站（GitLab、Reddit、购物） |
| VisualWebArena | 910 | WebArena + 视觉观察 |
| MiniWoB++ | 100+ | 简化 Web 任务（表单、按钮） |
| WorkArena | 33 | 企业应用（ServiceNow） |

## 安装

```bash
# 安装浏览器自动化依赖
uv pip install -e ".[web]"

# 安装 Playwright 浏览器
playwright install chromium

# WebArena 需要额外部署网站服务
docker compose up -d  # 启动测试网站
```
