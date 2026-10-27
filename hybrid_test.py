import chromadb
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


client=chromadb.PersistentClient(path="chroma_db")
model=SentenceTransformer("BAAI/bge-small-zh-v1.5")
collection = client.get_collection("pdb_500")
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

print("正在加载数据并分词...")
all_data=collection.get()
docs=all_data["documents"] or []
ids=all_data["ids"]

tokenized_corpus=[list(jieba.cut(doc)) for doc in docs]

bm25=BM25Okapi(tokenized_corpus)

def bm25_retrieve(query, top_k=3):
    """基于 BM25 的关键字检索"""
    # 同样需要对用户的提问进行分词
    tokenized_query=list(jieba.cut(query))
    # 获取所有文档的 BM25 得分
    doc_scores=bm25.get_scores(tokenized_query)
    # 获取得分最高的 Top-K 索引
    top_n=sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]

    results=[]
    for idx in top_n:
        results.append({
            "id": ids[idx],
            "score": doc_scores[idx],
            "text": docs[idx]
        })
    return results


def rrl_fusion(vector_ids, bm25_ids,k=60):
    """RRL 融合"""
    # 计算向量检索的 RRF 得分
    rrf_score={}
    for rank, doc_id in enumerate(vector_ids, start=1):
        rrf_score[doc_id] = rrf_score.get(doc_id, 0) + 1 / (rank * k)

    # 计算 BM25 检索的 RRF 得分
    for rank, doc_id in enumerate(bm25_ids, start=1):
        rrf_score[doc_id] = rrf_score.get(doc_id, 0) + 1 / (rank * k)
    # 按总分降序排列
    sorted_docs=sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)

    return sorted_docs

def hybrid_retrieve(query, top_k=3):
    # 1. 向量检索 Top-K
    vector_results=collection.query(
        query_embeddings=[model.encode(QUERY_PREFIX + query).tolist()],
        n_results=top_k,
    )
    vector_ids=vector_results["ids"][0]
    # 2. BM25 检索 Top-K
    bm25_results=bm25_retrieve(query, top_k=top_k)
    bm25_ids=[result["id"] for result in bm25_results]
    # 3. RRL 融合
    fusion_results=rrl_fusion(vector_ids, bm25_ids, top_k)
    final_ids=[doc_id for doc_id,score in fusion_results][:top_k]
    return final_ids

if __name__ == "__main__":
    complex_queries=[
        "什么是「双百分」规则？"
        "卖家编辑SKU应符合什么要求？"
        "假设买家于2025年6月1日下单购买主商品，卖家于2025年6月5日在平台“点击发货”操作，则卖家须在什么时间赠品发货？"
    ]
    for query in complex_queries:
        print("\n" + "="*50)
        print(f"🔍 复杂查询: {query}")

        v_res=collection.query(
            query_embeddings=[model.encode(QUERY_PREFIX + query).tolist()],
            n_results=3,
        )
        print(f"🧠 纯向量检索 ID: {v_res['ids'][0]}")

        b_res = bm25_retrieve(query, top_k=3)
        print(f"🔤 纯 BM25 检索 ID: {[r['id'] for r in b_res]}")
        h_res = hybrid_retrieve(query, top_k=3)
        print(f"🔥 混合检索 ID: {h_res}")
        