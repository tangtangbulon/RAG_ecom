import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import chromadb
import re
import urllib.request
import urllib.error

def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_dotenv()

cilent=chromadb.PersistentClient(path="chroma_db")

# 删除旧集合（开发阶段使用，会清空数据）
try:
    cilent.delete_collection(name="policy_documents")
    print("🗑️ 已清空旧数据库")
except Exception:
    pass

# 重新创建集合
collection = cilent.get_or_create_collection(name="policy_documents")

print("正在加载模型，请稍候...")
model=SentenceTransformer("BAAI/bge-small-zh-v1.5")

file_path="data"

def process_file(file_path):
    chunks_with_metadata=[]
    filename=os.path.basename(file_path)
    filename_parts=filename.split("_")
    platform=filename_parts[0]
    date_val=filename_parts[1] if len(filename_parts) > 1 else ""
    name_without_extension=filename.split(".")[0]
    parts=name_without_extension.split("-")
    if len(parts)>=2:
        date_match=re.search(r'\d{4}-\d{2}-\d{2}', parts[0])
        if date_match:
            date_val= date_match.group(0)
        platform = parts[1]
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text.strip():
            print(f"文件 {file_path} 为空，跳过")
            return [], [], [], []
        documents, metadatas, ids, embeddings = [], [], [], []
        for i, chunk in enumerate(text.split('\n\n')):
            chunk = chunk.strip()
            if not chunk: continue
            documents.append(chunk)
            metadatas.append({
                "source_file":filename,
                "chunk_id":i,
                "platform":platform,
                "date":date_val
            })
            ids.append(f"{filename}_{i:03d}")
            embeddings.append(model.encode(chunk).tolist())

        print(f"✅ 成功处理：{filename}，切分出 {len(documents)} 块。")
        return documents, metadatas, ids, embeddings
    except Exception as e:
        print(f"❌ 处理 {filename} 时出错: {e}")
        return [], [], [], []

data_dir = "data"
for name in sorted(os.listdir(data_dir)):
    if not name.endswith((".md", ".txt")):
        continue
    file_path = os.path.join(data_dir, name)
    documents, metadatas, ids, embeddings = process_file(file_path)
    if not documents:
        continue
    print(f"✅ 成功处理：{name}，切分出 {len(documents)} 块。")
    collection.upsert(
        documents=documents,
        metadatas=metadatas, 
        ids=ids,
        embeddings=embeddings
            )
print(f"入库成功！当前数据库中一共有 {collection.count()} 条数据。") 


def retrieve_policy(query, top_k=3,platform=None,date_filter=None):
    """
    输入: 用户问题 (String)
    输出: 最相关的 Top-K 个文档块 (包含原本的文本和元数据)
    """

    where_conditions={}
    if platform:
        where_conditions["platform"]=platform
    if date_filter:
        where_conditions["date"]={"$gt": date_filter}

    query_params={
        "query_embeddings": [model.encode(query).tolist()],
        "n_results": top_k,
    }
    if where_conditions:
        query_params["where"]=where_conditions
        print(f"🔎 正在带条件检索: {where_conditions}")
    else:
        print("🔎 正在全库检索...")

    results = collection.query(**query_params)
    return results



def generate_answer(query, docs,metadata):
    context_parts=[]
    for i,(doc,meta) in enumerate(zip(docs,metadata),1):
        src=meta.get("source_file","unknown")
        context_parts.append(f"第{i}份文件:{src}\n{doc}")
    context="\n\n".join(context_parts)
    
    prompt = f"""
    你是一个专业的客服政策问答助手。请严格根据以下【参考信息】回答用户问题。
    
    【参考信息】：
    {context}
    
    【用户问题】：
    {query}
    
    【回答要求】：
    1. 如果参考信息中没有答案，或信息不足，请直接回答"抱歉，在现有资料中未找到相关答案"，绝不允许编造。
    2. 回答要简明扼要。
    3. 在回答的最后，用 [1][2] 这样的格式标注你引用的参考信息编号（可连续，如 [1][3]），不要使用其他引用格式。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请在 .env 中设置 DEEPSEEK_API_KEY")

    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek 请求失败: {e.code} {detail}") from e

    return body["choices"][0]["message"]["content"]

def show_citations(references,metadatas):
    cited=[]
    for n in re.findall(r'\[(\d+)\]',references):
        idx=int(n)
        if 1<=idx<=len(metadatas) and idx not in cited:
            cited.append(idx)
        if not cited:
            print("🔍 未找到引用信息")
            return
        print("\n📚 参考来源：")
        for idx in cited:
            meta=metadatas[idx-1]   
            print(f"  [{idx}] {meta.get('source_file')} "
                f"(chunk={meta.get('chunk_id')}, platform={meta.get('platform')}, date={meta.get('date')})")


def answer_query(query,top_k=3,platform=None,date_filter=None):
    results=retrieve_policy(query,top_k,platform,date_filter)
    documents = results.get("documents")
    docs = documents[0] if documents else []
    metadata = results.get("metadatas")
    metadatas = metadata[0] if metadata else []
    if not docs:
        print("抱歉，未检索到相关文档。")
        return None
    answer = generate_answer(query,docs,metadatas)
    print("\n🤖 回答：\n" + answer)
    show_citations(answer,metadatas)
    return answer