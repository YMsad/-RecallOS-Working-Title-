RecallOS - 你的苏格拉底式学习伴侣

> "AI amplifies learning. It never replaces it."

RecallOS 是一个基于苏格拉底追问法的 AI 学习工具。它不会替你总结、不会替你记忆，而是通过不断追问"为什么"，引导你自己抵达理解的瞬间。

---

✨ 核心哲学

| 传统学习 | RecallOS |
|----------|----------|
| 被动阅读 | 主动回答 |
| 记忆知识点 | 理解底层逻辑 |
| 孤立的概念 | 连接的知识网络 |
| 工具替你干活 | AI逼你思考 |

RecallOS 相信：

> 真正的学习，不是"我知道了"，而是"我搞懂了"。

那个瞬间——当你突然说出 **"哦！原来是这样！"** ——就是 RecallOS 交付的核心价值。

---

🎯 功能概览 V0.1 核心功能

```text
📝 输入文本
    ↓
🤔 苏格拉底追问（AI生成4层递进问题）
    ↓
💬 用户回答 + AI实时反馈（正确推进 / 错误给提示 / 连错3次给参考）
    ↓
🔗 知识链接推荐（"这个概念和你之前学的X有关"）
    ↓
📊 每日总结（"你今天真正搞懂了：XXX" + 明日钩子）
    ↓
💾 本地存储（你的知识资产，永远属于你）三大核心能力
功能	描述	状态
苏格拉底追问	AI生成层层递进的"为什么"问题，引导深度思考	✅ 已完成
主动回忆	将学习内容转化为问题，逼迫大脑输出而非输入	✅ 已完成
知识链接	自动发现概念间的关联，构建个人知识网络	✅ 已完成
🛠️ 技术栈

层级	技术	说明
AI引擎	DeepSeek API	中文理解力强，成本极低（≈4元/本书）
存储	SQLite	本地数据库，用户拥有全部数据
后端	Python 3.14+	核心逻辑（异步 httpx + pydantic）
前端	Streamlit / CLI	Web界面 + 命令行双模式
🚀 快速开始（开发版）

1. 克隆项目
bash
git clone https://github.com/YMsad/-RecallOS-Working-Title-.git
cd RecallOS
2. 创建虚拟环境（推荐）
bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
3. 安装依赖
bash
pip install -r requirements.txt
4. 配置 API 密钥
bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，填入你的 DeepSeek API Key
# 去 https://platform.deepseek.com 注册获取
.env 文件内容示例：

ini
DEEPSEEK_API_KEY="sk-你的密钥"
DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
DEEPSEEK_MODEL="deepseek-chat"
5. 运行
方式一：Streamlit Web 界面（推荐）
bash
streamlit run app.py
方式二：命令行界面
bash
# 开始学习
python -m cli.main

# 查看学习历史
python -m cli.main --history
6. 开始学习
text
📖 输入概念名称 + 粘贴你想学习的内容

🤔 第1层：这个概念的核心是什么？
💭 你的回答：...

🤔 第2层：为什么它重要？
💭 你的回答：...

🤔 第3层：如果没有它，世界会怎样？
💭 你的回答：...

🤔 第4层：它和你之前学的X有什么联系？
💭 你的回答：...

✨ 每日总结：
"你今天真正搞懂了：XXX"
"明天你可以继续深挖：ZZZ"
📁 项目结构（实际代码）

text
RecallOS/
├── core/                    # 核心引擎
│   ├── __init__.py          # 统一导出
│   ├── config.py            # 环境变量配置（.env）
│   ├── client.py            # DeepSeek API客户端（重试+超时+异常）
│   ├── database.py          # SQLite五表 + CRUD操作
│   ├── models.py            # Pydantic数据模型 + 验证
│   ├── prompts.py           # 苏格拉底式提示词模板
│   └── session.py           # LearningSession状态机
│
├── cli/                     # 命令行界面
│   ├── __init__.py
│   └── main.py              # CLI主程序 + --history历史回顾
│
├── tests/                   # 测试（85个全部通过）
│   ├── test_client.py       # 10个mock测试
│   ├── test_database.py     # 数据库测试
│   ├── test_models.py       # 18个模型测试
│   ├── test_prompts.py      # 23个提示词测试
│   ├── test_session.py      # 7个会话测试
│   ├── test_cli.py          # 4个CLI测试
│   ├── test_app.py          # 3个Streamlit测试
│   └── conftest.py          # pytest配置
│
├── docs/                    # 设计文档
│   ├── prd.md               # 产品需求文档（V0.1纳瓦尔版）
│   ├── ui.md                # UI设计规范
│   └── 技术文档.md           # 技术架构设计
│
├── backend/                 # 后端支撑模块（预留扩展）
│   ├── ai/
│   ├── anki/                # 未来Anki同步
│   ├── database/
│   ├── parser/              # PDF/EPUB解析（V0.2）
│   └── utils/
│
├── prompts/                 # 提示词模板备份
│   └── README.md
│
├── app.py                   # Streamlit UI（5个页面）
├── requirements.txt         # 依赖清单
├── .env.example             # 环境变量模板
├── .gitignore               # Git忽略规则
├── conftest.py              # pytest全局配置
├── README.md                # 项目说明（就是你正在看的）
└── LICENSE                  # MIT开源协议
📊 成功指标（北星指标）

指标	目标	说明
理解瞬间完成次数	↑	用户标记"我搞懂了"的次数
次日留存率	>40%	用户是否主动回来
平均学习时长	8-15分钟	深度但不疲惫
每周使用天数	>5天	形成习惯
RecallOS 不关注：

❌ 学习时长

❌ 打卡天数

❌ 卡片数量

🧭 开发路线图

版本	目标	时间预估
V0.1	MVP：追问+问答+本地存储+CLI+Web	✅ 已完成（2.5小时）
V0.2	复习调度 + 知识图谱可视化	+20小时
V0.5	Web界面完善 + 多文档管理	+30小时
V1.0	完整学习OS（多模式+云同步+分析）	待定
🤝 贡献指南

这是一个独立开发项目，但欢迎任何形式的反馈：

🐛 Bug报告：开Issue描述问题

💡 功能建议：开Discussion讨论

📖 测试用例：用自己的学习资料测试，反馈效果

☕ 精神支持：Star这个项目就是最大的鼓励

📜 许可证

MIT License - 你可以自由使用、修改、分发。

🙏 致谢

纳瓦尔 - "做别人需要的，而不是别人想要的"

苏格拉底 - "我唯一知道的就是我一无所知"

DeepSeek - 让高质量AI触手可及

📬 联系

作者：16岁独立开发者

项目状态：✅ V0.1 已完成

反馈通道：GitHub Issues / Discussions

"真正的学习，不是收集知识，而是内化知识。"

RecallOS - 让你真正搞懂。

text

---

### ✅ 你只需要做

1. **打开 `README.md`**，全选删除，粘贴上面这段
2. **保存**
3. **提交推送**：
```bash
git add README.md
git commit -m "docs: 更新 README 与实际项目结构同步"
git push
