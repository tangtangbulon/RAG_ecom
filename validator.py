import re
import jieba
from generator import generate_answer
from retriever import HybridRetriever


def verify_and_show_citations(answer,docs, metadatas, query):
    """
    严谨的引用校验逻辑
    """
    cited_indices = []
    for match in re.findall(r'\[(\d+)\]', answer):
        idx = int(match)

        if 1 <= idx <= len(metadatas):
            if idx not in cited_indices:
                cited_indices.append(idx)
        else:
            print(f"⚠️ 警告：模型捏造了无效引用 [{idx}]，该文献不存在！")

    if not cited_indices:
        print("⚠️ 警告：模型生成的答案中没有有效的引用！")
        return 
    
    print("\n📚 校验通过的参考来源：")
    for idx in cited_indices:
        meta = metadatas[idx-1]
        doc_text = docs[idx - 1]

        query_words=[w for w in jieba.cut(query) if len(w) > 1 and w not in ["什么", "怎么", "如何", "政策", "规定"]]
        doc_matched = False
        missing_words = []
        for word in query_words:
            if word in doc_text:
                doc_matched = True
                break
            else:
                missing_words.append(word)
        status = "✅ 实体匹配通过" if doc_matched or not query_words else "❌ 实体词不匹配(可能幻觉)"
        print(f"  [{idx}] {meta.get('source_file')} (chunk={meta.get('chunk_id')})")
        print(f"      校验状态：{status}")
        if not doc_matched and query_words:
            print(f"      缺失词汇：{missing_words}")