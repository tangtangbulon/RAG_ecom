# 🛒 淘宝/电商平台政策智能问答系统 (RAG)

这是一个基于 RAG (Retrieval-Augmented Generation) 架构的垂直领域智能问答系统。系统通过对淘宝平台冗长的政策文档进行深度处理，实现了**高精度检索、安全防注入、以及严谨的引用校验**，旨在解决大模型在垂直领域问答中的“幻觉”问题。

## 🌟 核心特性

*   **智能分块 (Smart Chunking)**：对比了 200 字与 500 字滑动窗口（Sliding Window）切分策略。经过量化评估，最终选用 `chunk_size=500, overlap=50`，在保证上下文完整性的同时避免了信息碎片化。
*   **混合检索 (Hybrid Search)**：结合了基于 `BAAI/bge-small-zh-v1.5` 的**向量语义检索**与基于 `jieba` + `rank_bm25` 的**字面关键字检索**。通过 `RRF (Reciprocal Rank Fusion)` 算法对两路结果进行科学融合。
*   **前置拒答机制 (Early Stopping)**：引入 L2 距离阈值（`threshold = 1.2`）。当最佳检索结果与用户问题相似度过低时，直接触发拒答，**不消耗 LLM Token**，有效防止无关问题引发的模型幻觉。
*   **安全防护 (Security)**：构建了多层防护体系，包括数据入库清洗（正则拦截恶意指令）与强化系统 Prompt（`<context>` 边界约束），有效防御 Prompt Injection 攻击。
*   **严谨的引用校验 (Verification)**：提取大模型回答中的 `[x]` 引用标签，校验编号合法性，并通过 `jieba` 分词提取问题实体，校验引用的文档是否真实包含相关实体，防止大模型“捏造引用”。

## 📊 性能评估 (Evaluation)

系统构建了包含 30+ 真实复杂查询的测试集 (`test_cases.json`)，自动化跑批脚本 (`run_eval.py`) 跑出的最终指标如下：

| 实验组 (Chunk Size) | Recall@3 | MRR | 结论 |
| :--- | :--- | :--- | :--- |
| `db_200` | 90.00% | 0.6833 | 语义被切断，导致正确答案排名靠后。 |
| **`db_500` (最终选用)** | **90.00%** | **0.9000** | 上下文完整，正确答案绝大多数排在首位。 |

> **注**：引入混合检索与拒答机制后，系统在无关查询上的误答率降至 **0%**。

## 🏗️ 系统架构

```text
用户查询 (Query)
       ↓
[预处理与安全审查] (sanitize_text 拦截恶意指令)
       ↓
 ┌───────────────┐
 │ 混合检索模块  │
 │ 1. 向量检索(语义) │
 │ 2. BM25检索(字面) │
 └───────────────┘
       ↓
[RRF 融合算法] (重排合并 Top-3)
       ↓
[距离阈值拒答判断] (L2 距离 > 1.2 则拦截)
       ↓
[大模型生成 (DeepSeek)] (注入强边界 Prompt)
       ↓
[引用校验 (Validator)] (实体词匹配校验)
       ↓
最终回答 (Answer + 真实引用)


├── data/                  # 存放原始政策文档 (.md)
├── main.py                # 数据清洗、切块与 ChromaDB 建库脚本
├── retriever.py           # 混合检索与阈值拒答逻辑
├── generator.py           # Prompt 组装与 DeepSeek API 调用
├── validator.py           # 引用校验与安全防护逻辑
├── pipeline.py            # RAG 流水线主入口
├── test_cases.json        # 30+ 手工标注的评估测试集
├── run_eval.py            # 自动化评估跑批脚本
├── evaluation_report.md   # 自动生成的详细评估报告
├── BUG_LOG.md             # 开发过程中的真实踩坑记录
└── README.md