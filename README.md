RecallOS - 你的苏格拉底式学习伴侣 "AI amplifies learning. It never replaces it."

RecallOS 是一个基于苏格拉底追问法的AI学习工具。它不会替你总结、不会替你记忆，而是通过不断追问"为什么"，引导你自己抵达理解的瞬间。

✨ 核心哲学 传统学习 RecallOS 被动阅读 主动回答 记忆知识点 理解底层逻辑 孤立的概念 连接的知识网络 工具替你干活 AI逼你思考 RecallOS 相信：

真正的学习，不是"我知道了"，而是"我搞懂了"。

那个瞬间——当你突然说出 "哦！原来是这样！" ——就是 RecallOS 交付的核心价值。

🎯 功能概览 V0.1 核心功能（30-50小时开发） text 📝 输入文本 ↓ 🤔 苏格拉底追问（AI生成3-5层递进问题） ↓ 💬 用户回答 + AI实时反馈 ↓ 🔗 知识链接提醒（"这个概念和你之前学的X有关"） ↓ 📊 每日总结（"你今天真正搞懂了：XXX"） ↓ 💾 本地存储（你的知识资产，永远属于你） 三大核心能力 功能 描述 状态 苏格拉底追问 AI生成层层递进的"为什么"问题，引导深度思考 🚧 开发中 主动回忆 将学习内容转化为问题，逼迫大脑输出而非输入 🚧 开发中 知识链接 自动发现概念间的关联，构建个人知识网络 🚧 开发中 🛠️ 技术栈 层级 技术 说明 AI引擎 DeepSeek API 中文理解力强，成本极低（≈4元/本书） 存储 SQLite 本地数据库，用户拥有全部数据 后端 Python 3.10+ 核心逻辑 前端（可选） Streamlit / 命令行 MVP阶段先用CLI验证体验 🚀 快速开始（开发版）

克隆项目 bash git clone https://github.com/yourusername/RecallOS.git cd RecallOS
安装依赖 bash pip install -r requirements.txt
配置API密钥 bash
去 platform.deepseek.com 注册，获取API Key
export DEEPSEEK_API_KEY="sk-你的密钥" 4. 运行命令行原型 bash python cli.py 5. 开始学习 text 📖 粘贴你想学习的内容（或输入文件路径）：

机会成本是指为了得到某种东西而放弃的其他东西的最大价值...

🤔 让我问你几个问题：

为什么"放弃的"比"得到的"更能定义成本？ 💭 你的回答：...
如果没有稀缺性，机会成本还存在吗？ 💭 你的回答：... ... 📁 项目结构 text RecallOS/ ├── src/ │ ├── ai/ │ │ ├── deepseek_client.py # DeepSeek API封装 │ │ ├── socratic.py # 苏格拉底追问引擎 │ │ ├── qa_generator.py # 问题生成器 │ │ └── linker.py # 知识链接引擎 │ ├── storage/ │ │ ├── models.py # 数据结构定义 │ │ └── database.py # SQLite操作 │ ├── learning/ │ │ └── session.py # 学习会话编排 │ └── cli/ │ └── interface.py # 命令行交互 ├── data/ │ └── recallos.db # 本地数据库（自动生成） ├── tests/ # 测试用例 ├── requirements.txt └── README.md 📊 成功指标（北星指标） 指标 目标 说明 理解瞬间完成次数 ↑ 用户标记"我搞懂了"的次数 次日留存率 >40% 用户是否主动回来 平均学习时长 8-15分钟 深度但不疲惫 每周使用天数 >5天 形成习惯 RecallOS 不关注：
❌ 学习时长

❌ 打卡天数

❌ 卡片数量

🧭 开发路线图 版本 目标 时间预估 V0.1 MVP 命令行原型：追问+问答+本地存储 30-50小时 V0.2 增加复习调度 + 知识链接可视化 +20小时 V0.5 Web界面 + 多文档管理 +30小时 V1.0 完整学习OS（多模式+云同步+分析） 待定 🤝 贡献指南 这是一个独立开发项目，但欢迎任何形式的反馈：

🐛 Bug报告：开Issue描述问题

💡 功能建议：开Discussion讨论

📖 测试用例：用自己的学习资料测试，反馈效果

☕ 精神支持：Star这个项目就是最大的鼓励

📜 许可证 MIT License - 你可以自由使用、修改、分发。

🙏 致谢 纳瓦尔 - "做别人需要的，而不是别人想要的"

苏格拉底 - "我唯一知道的就是我一无所知"

DeepSeek - 让高质量AI触手可及

📬 联系 作者：16岁独立开发者

项目状态：🚧 积极开发中

反馈通道：GitHub Issues / Discussions

"真正的学习，不是收集知识，而是内化知识。"

RecallOS - 让你真正搞懂。