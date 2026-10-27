import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import time
from retriever import HybridRetriever
from generator import generate_answer
from validator import verify_and_show_citations

def calculate_metrics(retrieved_ids, expected_ids, k=3):
    """计算单条数据的 Recall@K 和 MRR"""
    if not expected_ids:
        return 1.0 if not retrieved_ids else 0.0,1.0 if not retrieved_ids else 0.0
    recall=1.0 if any(doc_id in retrieved_ids for doc_id in expected_ids) else 0.0
    mrr=0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected_ids:
            mrr=1/rank
            break
    return recall, mrr

def run_evaluation():
    with open(Path(__file__).resolve().parent / "test_cases.json", "r", encoding="utf-8") as f:
        test_cases=json.load(f)
    retriever = HybridRetriever("pdb_200")

    total_recall,total_mrr=0.0,0.0
    markdown_report = "# RAG 系统评估报告\n\n"
    markdown_report += "| 序号 | Query | 是否拒答 | 检索到的 IDs | 期望 IDs | Recall@3 | MRR | 生成状态 |\n"
    markdown_report += "|---|---|---|---|---|---|---|---|\n"
    print(f"🚀 开始跑批测试，共 {len(test_cases)} 条...")
    for i, case in enumerate(test_cases, start=1):
        query=case["query"]
        expected_ids = case.get("expected_id", [])
        print(f"[{i}/{len(test_cases)}] 正在测试: {query}")

        docs, metas, distance = retriever.retrieve(query)

        retrieved_ids = [meta.get("source_file") + "_" + f"{meta.get('chunk_id'):03d}" for meta in metas] if metas else []
        is_rejected = (docs is None)
        if is_rejected:
            answer = "抱歉，系统未找到与您问题高度相关的政策信息。"
            gen_status = "成功拒答"
            recall, mrr = (1.0, 1.0) if not expected_ids else (0.0, 0.0)
        else:
            answer = generate_answer(query, docs, metas)
            gen_status = "生成成功" if "生成失败" not in answer else "API 报错"
            verify_and_show_citations(answer, docs, metas, query)
            recall, mrr = calculate_metrics(retrieved_ids, expected_ids)

        total_recall += recall
        total_mrr += mrr

        hit_icon="✅" if recall > 0 else "❌"
        markdown_report +=f"| {i} | {query} | {'是' if is_rejected else '否'} | {retrieved_ids} | {expected_ids} | {hit_icon} {recall} | {mrr:.2f} | {gen_status} |\n"
        markdown_report += f"\n<details><summary>点击查看第 {i} 题详细回答</summary>\n\n**AI 回答:**\n{answer}\n\n</details>\n\n"

        time.sleep(1)


    avg_recall = total_recall / len(test_cases)
    avg_mrr = total_mrr / len(test_cases)
    report_header = f"## 综合指标\n- **Total Cases:** {len(test_cases)}\n- **Recall@3:** {avg_recall:.2%}\n- **MRR:** {avg_mrr:.4f}\n\n"
    markdown_report = markdown_report.replace("# RAG 系统评估报告\n\n", "# RAG 系统评估报告\n\n" + report_header)  # pyright: ignore[reportPossiblyUnboundVariable]
    with open("evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(markdown_report)
        
    print(f"\n🎉 测试完成！平均 Recall@3: {avg_recall:.2%}, MRR: {avg_mrr:.4f}")
    print("报告已保存至 evaluation_report.md")

if __name__ == "__main__":
    run_evaluation()