# RecallOS 界面设计文档（UI）

> 版本：V1.0 · 更新日期：2026-08-17
>
> 本文档基于 `app.py`（Streamlit）实际实现编写；样式见 `.streamlit/config.toml` 与 `app.py::inject_css`。

## 1. 设计规范

### 1.1 主题色板

| 用途 | 色值 | 说明 |
| --- | --- | --- |
| 主色 primaryColor | `#D4875E` | 按钮、强调 |
| 背景 backgroundColor | `#FDFBF7` | 页面底色（暖白） |
| 次级背景 secondaryBackgroundColor | `#EFEBE4` | 输入框、分组背景 |
| 文字 textColor | `#1A1A1A` | 正文 |
| AI 气泡 | `#F5EFE6` | 助手消息底 |
| 用户气泡 | `#EFEBE4` | 用户消息底 |
| 行分隔线 | `rgba(128,128,128,0.35)` | 分组表格行 |

页面强制 `headless = true`（本地直接渲染，不显示部署菜单）。

### 1.2 字体与排版

- 全局 16px（覆盖 Streamlit 默认 14px，更大更易读），`inject_css` 对所有主要控件统一设置。
- 排版节奏：`h1` 用小字号标题（20px 灰），正文用大号（26px 起）提问式文案。

### 1.3 全局布局

- `layout="centered"`，单列居中。
- 侧边栏固定：🏠 主页 / 📚 历史回顾 / 📊 用量统计 / 🔑 重新配置 API Key。
- 页面流转由 `st.session_state.step` 驱动，所有跳转按钮走 `_navigate` / `go_home`。

## 2. 页面结构

```
Render 入口 main()
├─ 未配置 Key → render_api_key_setup（首次引导）
├─ 已配置 → 侧边栏 + 按 step 分发：
│   ├─ learning        → render_learning（新/旧流程）
│   ├─ connections     → render_connections（旧流程连接推荐）
│   ├─ summary         → render_summary（今日总结）
│   ├─ review_list     → render_review_list（今日复习列表）
│   ├─ review          → render_review（复习会话）
│   ├─ concept_detail  → render_concept_detail（概念详情，历史页进入）
│   ├─ history         → render_history（我的知识）
│   ├─ usage_stats     → render_usage_stats（用量统计）
│   └─ reconfigure     → render_reconfigure（重配 API Key）
│   默认 step="home"   → render_home
```

## 3. 页面详细设计

### 3.1 首次配置（render_api_key_setup / render_reconfigure）

- 居中标题「📚 RecallOS」+「首次使用，请配置 DeepSeek API Key」。
- 说明文案：`Key 只会保存在本机 ~/.recallos/config.json，不会上传。`
- 密码框 + 「保存」主按钮；空 Key 报错。
- `render_reconfigure` 同构：覆盖旧 Key，保存后自动回首页。

### 3.2 首页（render_home）

```
📚 RecallOS
今天想弄懂什么？
[📝 今日复习（N）]           ← 有到期复习时
[📖 继续学习：{标题}]       ← 有「学习中」概念时
─────────────────────────
👋 先从一个精选概念开始——「机会成本」…  ← 首次打开自动推荐
✨ 精选概念         │   ✍️ 自定义
点击即可开始        │   名称输入 + 原文粘贴
[机会成本]         │   [学习目标 radio（可选，横向）]
[复利]            │   [💡 预热] [开始]
…
─────────────────────────
昨天你搞懂了：…
连续学习：第 N 天
```

要点：

- **两栏布局**（`st.columns([1,1])`）：左「精选概念」，右「自定义」。
- 预热：输入概念名后出现 `💡 预热` 按钮，AI 生成 1–2 句大白话解释（`warmup_concept`），结果以 `💡` 气泡展示，并作为开场消息带入学习。
- 学习目标用横向 radio，可选不阻塞。
- 底部展示最近学到概念 + streak。
- 「开始」在新流程下**不调用 AI**，直接进入阅读（`_start_new_session`）。

### 3.3 学习页 · 新流程（render_learning → _render_learning_new）

顶部有阶段指示器（`_render_stage_indicator`），按 `stage` 展示五种状态：
📖 阅读中 / 🧠 验证理解 / 💡 最小干预 / ✅ 已完成 / 🔄 需要重新学习。

**阅读阶段**

```
### 📖 阅读原文
[📋 只看重点] toggle          ← 只显示关键句（关键词加粗），原文收进展开
████████░░░░  2/5 段
**第 2 段 / 共 5 段**
（关键词加粗的段落正文 / 只看重点时的关键句列表）
🗝️ 这段在说什么？（用自己的话总结一句）
[◀ 上一段] [下一段 ▶]  ✅ 已记录 1/5 段的理解
[我读完了（主按钮）]
```

- 一次一段，进度条 + 「第 N 段 / 共 M 段」；只看重点模式抽 3 句关键句。
- 每段总结实时保存（`record_reading_answer`），作为验证与总结上下文。
- 未贴原文时显示「直接开始验证」兜底按钮。

**验证阶段**

```
### 📝 验证你的理解
[📄 再看一眼原文]（展开面板）
{validation_task}      ← AI 设计的任务（可含难度）
[😵 我看不懂，帮我解释]
（对话气泡；此处用 chat_input 收集闭卷回答）
```

- 后台执行：AI 分析回答 → Learner State；无缺口 → 直接完成（`验证过即完`）。
- 有缺口 → 先给「有对比的反馈」，再进入最小干预。
- 30% 概率在此阶段前多问一个「你觉得你能解释清楚吗」预判（元认知学习）。

**最小干预阶段**

```
### 💡 最小干预
跟着提示想一想，用自己的话回答就好，不用追求标准答案。
（干预气泡，图标按类型：💡提示 🌰例子 🎭类比 ⚠️反例 🤔直问 🔁复述 ✅结束）
（chat_input 收集回答）
```

- 干预按强度阶梯递进：hint → example → analogy → counterexample → question。
- 干预被判无效则上升一级；连续 2 次无效 → 直接完成，附一句温情的结束语。
- 缺口消失 → 完成。AI 失败显示可重试错误。

**完成 / 重新学习**

- 完成：`🎉 理解验证通过，今天学完啦！` + 说明 + 「查看今日总结」。
- 重新学习：连续 3 次未通过 → 提示回看原文，按钮「重新读一遍，再试一次」。

### 3.4 今日总结（render_summary）

```
## ✅ 今天学完了
### ✨ 你搞懂了什么
> {breakthrough}
────────────
### 💡 说白了我就是
> {plain}
────────────
### 📌 明天 AI 会问你
> {tomorrow_hook}
────────────
📊 累计掌握 N 个概念，建立 M 条连接
[👋 明天继续（主按钮）]
[🔍 还想再挖深一点？（可选）]    展开
[🔗 查看相关概念（可选）]       展开
```

- 新流程自动生成（`finish_auto`），旧流程保留「输入自己的话再生成」。
- 生成后才 streak+1（当天首次时）。

### 3.5 复习（render_review_list / render_review）

**列表页**

```
## 📝 今日复习
以下概念到了复习时间，AI 会用上次学完时的追问来检验你。
**概念标题**  ✅/🔄 掌握度     [开始复习]
```

- 空状态：`今天没有到期待复习的概念，先去学点新东西吧。`

**复习会话**

```
## 📝 复习：{标题}
（AI 提问气泡）
（chat_input「你的回答…」）
✓/🤔 判分反馈   ← 答对立即通过；答错继续直至 3 次
三连错 → 警告「这个知识点建议重新学一遍。」
复习通过 → `复习通过，掌握更牢固了！` + 返回按钮
```

### 3.6 我的知识（render_history）

```
## 📚 我的知识
### ✅ 搞懂了            ┐
│ 概念名称  操作[查看|✕] │ 三个分组表格，按掌握度排序
│ ────────行分隔────────│ 行内展开详情（点击「查看」）
│ **机会成本** 查看 ✕   ┘
│  ──────────────
│  📖 机会成本 ✅ 搞懂了
│  我的理解：…
│  ── 学习记录 ──（追问 / 验证 / 干预 / 深化 记录）
│  ── 知识连接 ──
│  ── 每日总结 ──
```

- 三组：✅ 搞懂了 → 🔄 模糊 → 📖 学习中。
- 行内删除两步确认；行内展开（不跳页），默认展开第一条。
- 紧凑 Excel 风：低留白、行分隔线（`inject_css` 中的 `row-divider`）。

### 3.7 用量统计（render_usage_stats）

```
## 📊 用量统计
[今日 Token/次/成本] [本月] [累计]     ← 3 个 metric 卡片
近 7 天消耗趋势
| 日期 | Token | 调用次数 | 成本(元) |   ← st.table
说明：成本按 DeepSeek 公开价估算（输入 ¥0.27/1M tokens，输出 ¥1.10/1M tokens）。
```

### 3.8 旧流程学习页（_render_learning_old，保留）

开场问题 + 四层追问 + 连接推荐页面（render_connections：AI 推荐连接 + 可编辑关系文本 + 保存）。仅 `RECALLOS_NEW_FLOW=0` 时启用。

## 4. 交互与消息气泡规范

- 所有 AI 回复按 role 渲染为圆角气泡：
  - 助手：`#F5EFE6` 底，白色区，`line-height 1.6`。
  - 用户：`#EFEBE4` 底，右对齐，`max-width 85%`。
- 文案风格：口语化、有画面感；多用 emoji 表意（💡📖📝🎉🔗📊）。
- 状态提示统一前缀：`✓` 正确 / `🤔` 需改进 / `❌` 错误 · 可重试 / `💡` 提示 / `📖` 参考。
- 错误处理指导思想：**AI 失败可重试、不中断页面**——所有 AI 调用包裹 try/except，标注 `DeepSeekAuthError`（Key 无效）与其他错误，均有重试/返回路径。

## 5. 实现备注

| 项 | 值 |
| --- | --- |
| 主题来源 | `.streamlit/config.toml`；打包版由 `launcher.py` 写入 `flag_options` |
| 全局 CSS | `app.py::inject_css`（字号、气泡、紧凑表格、行分隔线） |
| 状态 | `st.session_state.step` + 各页面局部 key（`v_*`、`review_*`、`home_*`） |
| 无 rerun 的提交 | 利用 Streamlit 同 run 处理链：先 `_run_pending` 再 `_capture_chat_answer`，避免 streamlit#7629 |