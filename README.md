# RecallOS - 你的苏格拉底式学习伴侣

> "AI amplifies learning. It never replaces it."

RecallOS 是一个基于苏格拉底追问法的 AI 学习工具。它不会替你总结、不会替你记忆，而是通过不断追问"为什么"，引导你自己抵达思考的瞬间。

---

## ✨ 核心哲学

| 传统学习 | RecallOS |
| --- | --- |
| 被动阅读 | 主动回答 |
| 记忆知识点 | 理解底层逻辑 |
| 孤立的概念 | 连接的知识网络 |
| 工具替你干活 | AI 逼你思考 |

RecallOS 相信：

> 真正的学习，不是"我知道了"，而是"我搞懂了"。

那个瞬间——当你突然说出 **"哦！原来是这样！"** ——就是 RecallOS 交付的核心价值。

---

## 🎯 功能概览（V0.2.0 核心功能）

**核心流程：**

```text
📝 输入文本
  ↓
🤖 苏格拉底追问（AI 生成 4 层递进问题）
  ↓
💬 用户回答 + AI 实时反馈（正确推进 / 错误给提示 / 连错 3 次给参考）
  ↓
🔗 知识链接推荐（"这个概念和你之前学的 X 有关"）
  ↓
📊 每日总结（"你今天真正搞懂了：XXX" + 明日钩子）
  ↓
💾 本地存储（你的知识资产，永远属于你）
```

**三大核心能力：**

| 功能 | 描述 | 状态 |
| --- | --- | --- |
| 苏格拉底追问 | AI 生成层层递进的"为什么"问题，引导深度思考 | ✅ 已完成 |
| 主动回忆 | 将学习内容转化为复盘，逼迫大脑输出而非输入 | ✅ 已完成 |
| 知识链接 | 自动发现概念之间的联系，构建个人知识网络 | ✅ 已完成 |

**技术栈：**

| 层级 | 技术 | 说明 |
| --- | --- | --- |
| AI 引擎 | DeepSeek API | 中文理解力强，成本极低（≈4 元/本书） |
| 存储 | SQLite | 本地数据库，用户拥有全部数据 |
| 后端 | Python 3.10+ | 核心逻辑（异步 httpx + pydantic） |
| 前端 | Streamlit / CLI | Web 界面 + 命令行双模式 |

---

## 🚀 快速开始（开发版）

### 1. 克隆项目

```bash
git clone https://github.com/YMsad/-RecallOS-Working-Title-.git
cd RecallOS
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
# Windows:
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 API 密钥

复制环境变量模板：

```bash
cp .env.example .env
```

然后编辑 `.env`，填入你的 DeepSeek API Key（到 [platform.deepseek.com](https://platform.deepseek.com) 注册获取）：

```ini
DEEPSEEK_API_KEY="sk-你的密钥"
DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
DEEPSEEK_MODEL="deepseek-chat"
```

> ⚠️ **注意**：不要将 `.env` 文件提交到 Git 或分享给他人，否则 API Key 可能泄露。

### 5. 运行

**方式一：一键启动脚本（自动检查 `.env`）**

- Windows：双击 `run.bat`，或命令行执行
  ```bat
  run.bat
  ```
- macOS / Linux：先赋予执行权限，再运行
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

**方式二：Streamlit Web 界面（推荐）**

```bash
streamlit run app.py
```

**方式三：命令行界面**

```bash
python -m cli.main             # 开始一次学习
python -m cli.main --history   # 回顾历史记录
```

### 6. 开始学习

📖 输入概念名称 + 粘贴想学习的内容，剩下的交给 AI 追问：

```text
第 1 层：这个概念的核心本质是什么？
第 2 层：为什么它重要？
第 3 层：如果没有它，世界会怎样？
第 4 层：它和你之前学的 X 有什么联系？
```

每日总结：

```text
✨ 你今天真正搞懂了：XXX
🌱 明天你可以继续深挖：ZZZ
```

---

## 📁 项目结构（实际代码）

```text
RecallOS/
├── core/                    # 核心引擎
│   ├── __init__.py          # 统一导出
│   ├── config.py            # 环境变量配置（.env）
│   ├── client.py            # DeepSeek API 客户端（重试+超时+异常）
│   ├── database.py          # SQLite 五表 + CRUD 操作
│   ├── models.py            # Pydantic 数据模型 + 验证
│   ├── prompts.py           # 苏格拉底式提示词模板
│   └── session.py           # LearningSession 状态机
│
├── cli/                     # 命令行界面
│   ├── __init__.py
│   └── main.py              # CLI 主程序（--history 历史回顾）
│
├── backend/                 # 后端支撑模块（预留扩展）
│   ├── ai/
│   ├── anki/                # 未来 Anki 同步
│   ├── database/
│   ├── parser/              # PDF/EPUB 解析（V0.2）
│   └── utils/
│
├── tests/                   # 测试（85 个全部通过）
│   ├── test_client.py       # 10 个 mock 测试
│   ├── test_database.py     # 数据库测试
│   ├── test_models.py       # 18 个模型测试
│   ├── test_prompts.py      # 23 个提示词测试
│   ├── test_session.py      # 7 个会话测试
│   ├── test_cli.py          # 8 个 CLI 测试
│   ├── test_app.py          # 3 个 Streamlit 测试
│   └── conftest.py          # pytest 配置
│
├── docs/                    # 设计文档
│   ├── prd.md               # 产品需求文档（V0.1 方案 A）
│   ├── ui.md                # UI / 交互设计
│   └── 技术文档.md           # 技术架构演进
│
├── prompts/                 # 提示词模板备份
│   └── README.md
│
├── app.py                   # Streamlit UI（5 个页面）
├── run.bat                  # Windows 一键启动（自动检查 .env）
├── run.sh                   # macOS/Linux 一键启动（chmod +x run.sh）
├── setup.py                 # pip install -e . 本地安装（recallos-socratic）
├── requirements.txt         # 锁定版本的依赖清单
├── .env.example             # 环境变量模板
├── .gitignore               # Git 忽略规则
├── conftest.py              # pytest 全局配置
├── README.md                # 项目说明（就是你正在看的）
└── LICENSE                  # MIT 开源协议
```

---

## 📊 成功指标（北极星指标）

| 指标 | 目标 | 说明 |
| --- | --- | --- |
| 理解触发次数 | ↑ | 用户标记"我搞懂了"的次数 |
| 次日回来学习 | >40% | 用户是否主动回来 |
| 平均学习时长 | 8-15 分钟 | 深度但不疲惫 |
| 每周使用天数 | >5 天 | 形成习惯 |

RecallOS 不关注：

- ❌ 学习时长
- ❌ 打卡天数
- ❌ 卡片数量

---

## 🧭 开发路线图

| 版本 | 目标 | 时间预估 |
| --- | --- | --- |
| V0.1 | MVP：追问 + 问答 + 本地存储 + CLI + Web | ✅ 已完成 |
| V0.2.0 | 对话感升级：降维追问、零基础模式、解释模式、动态开场、认知反差 | ✅ 已完成 |
| V0.2.1 | 思维模型：黄金圈、场景化提问、结构性类比 | +3 小时 |
| V0.2.2 | 精细化：理解可编辑、Token 记账、历史删除 | +3 小时 |
| V0.3 | 内容 + 平台扩展：PDF/EPUB/网页导入、多端打包 | +10 小时 |
| V1.0 | 完整学习 OS（多模式 + 云同步 + 分析） | 待定 |

---

## 🤝 贡献指南

这是一个独立开发项目，但欢迎任何形式的反馈：

- 🐛 Bug 报告：开 Issue 描述问题
- 💡 功能建议：开 Discussion 讨论
- 📖 测试用例：用自己的学习资料测试，反馈效果
- ☕ 精神支持：Star 这个项目就是最大的鼓励

---

## 📜 许可证

[MIT License](LICENSE) - 你可以自由使用、修改、分发。

---

## 🙏 致谢

- 纳瓦尔 - "做别人需要的，而不是别人想要的"
- 苏格拉底 - "我唯一知道的就是我一无所知"
- DeepSeek - 让高质量 AI 触手可及

---

## 📬 联系

- 作者：YMsad
- 项目状态：✅ V0.2.0 已完成
- 反馈通道：GitHub Issues / Discussions

> "真正的学习，不是收集知识，而是内化知识。"

RecallOS - 让你真正搞懂。