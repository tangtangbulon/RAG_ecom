import os
import chromadb
from sentence_transformers import SentenceTransformer
from main import QUERY_PREFIX


client=chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="policy_documents")
model=SentenceTransformer("BAAI/bge-small-zh-v1.5")
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

def retrieve_policy(query, top_k=3,platform=None,date_filter=None):

    query_params={
        "query_embeddings": [model.encode(QUERY_PREFIX + query).tolist()],
        "n_results": top_k,
    }
    results = collection.query(**query_params)
    return results


def evaluate_retrieval(collection_name,test_cases, k=3):
    collection = client.get_collection(collection_name)
    recall_hits = 0
    mrr_sum=0.0
    print(f"🚀 开始评估，共 {len(test_cases)} 个测试用例，评估指标：Recall@{k} 和 MRR\n")
    
    for i, case in enumerate(test_cases, start=1):
        query=case["query"]
        expected_id=case["expected_id"]
        results = collection.query(
            query_embeddings=[model.encode(QUERY_PREFIX + query).tolist()],
            n_results=k
        )
        retrieved_ids=results["ids"][0] if results["ids"] else []
        if expected_id[0] in retrieved_ids:
            recall_hits+=1
            hit_status = "✅ 命中"
        else:
            hit_status = "❌ 未命中"
        
        rank=0
        for idx, retrieved_id in enumerate(retrieved_ids):
            if retrieved_id==expected_id[0]:
                rank=idx+1
                break
        if rank>0:
            mrr_sum+=1/rank
        
        print(f"[{i}] 问题：{query}")
        print(f"    🎯 期望 ID: {expected_id}")
        print(f"    📦 检索结果: {retrieved_ids}")
        print(f"    📊 状态: {hit_status} | 排名: {rank if rank > 0 else '未找到'}\n")
    
    total_cases = len(test_cases)
    recall_at_k = recall_hits / total_cases if total_cases > 0 else 0
    mrr = mrr_sum / total_cases if total_cases > 0 else 0
    print("=" * 40)
    print(f"🏆 最终评估报告 (基于 {total_cases} 个测试用例)")
    print(f"🔹 Recall@{k}: {recall_at_k:.2%} ({recall_hits}/{total_cases})")
    print(f"🔹 MRR: {mrr:.4f}")
    print("=" * 40)
if __name__ == "__main__":
    test_cases_200=[
    {
        "query": "店铺真实体验分应用在哪里？",
        "expected_id": ["淘宝平台店铺真实体验分规范.md_010"],
    },
    {
        "query": "卖家发布商品或设置优惠时所填写的价格应该符合什么要求？",
        "expected_id": ["淘宝平台价格管理规则.md_021"],
    },
    {
        "query": "淘宝交互平台包括哪些？",
        "expected_id": ["淘宝平台交互风险信息管理规则.md_017"],
    },
    {
        "query": "基础质量分<60分时,商品会受到什么影响",
        "expected_id": ["淘宝平台商品信息质量分管理规范.md_005"],
    },
    {
        "query": "阿里旺旺服务管理违规处理情节一般场景，是指？",
        "expected_id": ["淘宝网阿里旺旺服务管理规范.md_000"],
    },
    {
        "query": "如何开通账期保障？",
        "expected_id": ["淘宝网保证金管理规范.md_020"],
    },
    {
        "query": "发布代购，商家需要具备什么样的条件？",
        "expected_id": ["淘宝网代购商品发布规范.md_000"],
    },
    {
        "query": "淘宝网店铺经营主体变更包括什么类型？",
        "expected_id": ["淘宝网店铺经营主体变更规范.md_000"],
    },
    {
        "query": "淘宝网会员须同时满足什么条件且通过淘宝网综合评估，方可创建店铺：？",
        "expected_id": ["淘宝网开店规范.md_000"],
    },
    {
        "query": "淘宝网收费规则是什么？",
        "expected_id": ["淘宝网卖家基础软件服务费规则.md_001"],
    }
]

    test_cases_500_v1=[
    {
        "query": "店铺真实体验分应用在哪里？",
        "expected_id": ["淘宝平台店铺真实体验分规范.md_003"],
    },
    {
        "query": "卖家发布商品或设置优惠时所填写的价格应该符合什么要求？",
        "expected_id": ["淘宝平台价格管理规则.md_006"],
    },
    {
        "query": "淘宝交互平台包括哪些？",
        "expected_id": ["淘宝平台交互风险信息管理规则.md_005"],
    },
    {
        "query": "基础质量分<60分时,商品会受到什么影响",
        "expected_id": ["淘宝平台商品信息质量分管理规范.md_001"],
    },
    {
        "query": "阿里旺旺服务管理违规处理情节一般场景，是指？",
        "expected_id": ["淘宝网阿里旺旺服务管理规范.md_003"],
    },
    {
        "query": "如何开通账期保障？",
        "expected_id": ["淘宝网保证金管理规范.md_006"],
    },
    {
        "query": "发布代购，商家需要具备什么样的条件？",
        "expected_id": ["淘宝网代购商品发布规范.md_000"],
    },
    {
        "query": "淘宝网店铺经营主体变更包括什么类型？",
        "expected_id": ["淘宝网店铺经营主体变更规范.md_000"],
    },
    {
        "query": "淘宝网会员须同时满足什么条件且通过淘宝网综合评估，方可创建店铺：？",
        "expected_id": ["淘宝网开店规范.md_000"],
    },
    {
        "query": "淘宝网收费规则是什么？",
        "expected_id": ["淘宝网卖家基础软件服务费规则.md_000"],
    }
]

    evaluate_retrieval("pdb_200",test_cases_200, k=3)
    evaluate_retrieval("pdb_500",test_cases_500_v1, k=3)