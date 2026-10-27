from retriever import HybridRetriever
from generator import generate_answer
from validator import verify_and_show_citations
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('BAAI/bge-reranker-base')



# 如果有校验模块也可以引入
# from validator import verify_and_show_citations 
def run_rag_pipeline(query):
    print(f"\n🔍 用户提问: {query}")
    
    # 1. 实例化检索器
    retriever = HybridRetriever("pdb_200")
    
    # 2. 执行检索（带拒答逻辑）
    docs, metas, distance = retriever.retrieve(query, top_k=10)
    # 3. 判断距离，如果太远直接拒绝
    if not docs:
        print(f"🛑 [拦截] 最佳匹配距离为 {distance:.4f}，超过阈值，触发拒答。")
        return "抱歉，系统未找到与您问题高度相关的政策信息。"
        
    print(f"✅ [检索成功] 最佳距离: {distance:.4f}")
    
    # 4. 生成回答
    pairs = [[query, doc] for doc in docs]
    pairs = [[query, doc] for doc in docs]
    scores = [float(s) for s in reranker.predict(pairs)]
    meta_list = metas or []
    ranked = sorted(zip(scores, docs, meta_list), key=lambda x: x[0], reverse=True)[:3]
    docs = [item[1] for item in ranked]
    metas = [item[2] for item in ranked]
    answer = generate_answer(query, docs, metas)
    verify_and_show_citations(answer, docs, metas, query)  # pyright: ignore[reportCallIssue]

    print(f"\n🤖 回答：\n{answer}")
    
    # 5. 引用校验（从 verify_qa.py 抽离）
    # verify_and_show_citations(answer, metas, query)
    
    return answer

