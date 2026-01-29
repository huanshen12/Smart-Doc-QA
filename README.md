# 🚀 Enterprise RAG Knowledge Base

> 基于 DeepSeek-V3 与 LangChain 的企业级本地知识库问答系统。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)](https://www.langchain.com/)

这是一个经过工程化重构的 RAG（检索增强生成）应用。区别于普通的 Demo，本项目专注于解决**数据持久化**、**文档解析健壮性**以及**会话状态管理**等实际落地痛点。

## ✨ 核心亮点

- **🧠 混合模型架构**：
  - **大脑**：接入 **DeepSeek-V3** (via OpenAI SDK)，实现高性价比的推理能力。
  - **眼睛**：使用阿里 **DashScope (通义千问)** Embedding，精准捕获中文语义。
- **💾 企业级数据处理**：
  - **MD5 幂等性去重**：上传文件时自动计算哈希指纹，防止重复入库，节省 Token 与存储空间。
  - **乱码防御机制**：内置 `UTF-8`/`GBK`/`UTF-16` 梯队解码策略，完美支持 Windows 老旧文本文件。
- **🗂️ 持久化记忆系统**：
  - 自研文件级 Session 管理 (`FileChatMessageHistory`)，重启服务后依然能通过 Session ID 找回历史对话。
- **⚡ 极致交互体验**：
  - 基于 LCEL (LangChain Expression Language) 构建流式管道，实现打字机式实时响应。

## 🛠️ 目录结构

```text
.
├── src/
│   ├── app_face_to_user.py      # 用户对话主界面 (Streamlit)
│   ├── app_file_uploader.py     # 知识库管理后台 (Streamlit)
│   ├── config_data.py           # 全局配置与模型参数
│   ├── knowledge_base.py        # 知识库核心服务 (切分、去重、入库)
│   ├── rag.py                   # RAG 核心链路 (LCEL Chain)
│   ├── vector.py                # 向量数据库封装 (ChromaDB)
│   └── file_chat_messages_history.py # 自定义历史记录管理
├── chroma_db/                   # 向量数据库持久化目录 (自动生成)
├── .env                         # 环境变量配置文件
├── requirements.txt             # 项目依赖
└── README.md                    # 项目文档
``` 
## 🚀 快速开始
1. 克隆项目
```Bash
git clone [https://github.com/YourUsername/Enterprise-RAG-Knowledge-Base.git](https://github.com/YourUsername/Enterprise-RAG-Knowledge-Base.git)
cd Enterprise-RAG-Knowledge-Base
```
2. 环境配置
建议使用 Conda 或 venv 创建独立的虚拟环境（推荐 Python 3.10+）：

```Bash
conda create -n rag_env python=3.10
conda activate rag_env
pip install -r requirements.txt
```
3. 配置密钥
在项目根目录创建 .env 文件，填入你的 API Key：

```Ini, TOML
# DeepSeek API 配置 (兼容 OpenAI 格式)
DEEPSEEK_BASE_URL=[https://api.deepseek.com/v1](https://api.deepseek.com/v1)
DeepSeek_api_key=sk-xxxxxxxxxxxxxxxx
# 阿里 DashScope 配置 (用于 Embedding)
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```
4. 运行系统
启动知识库管理后台（上传文档）：

```Bash
streamlit run src/app_file_uploader.py
```
启动智能问答客服（开始对话）：

```Bash
streamlit run src/app_face_to_user.py
```
