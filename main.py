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

client=chromadb.PersistentClient(path="chroma_db")

# 删除旧集合（开发阶段使用，会清空数据）
try:
    client.delete_collection(name="policy_documents")
    print("🗑️ 已清空旧数据库")
except Exception:
    pass

# 重新创建集合
collection = client.get_or_create_collection(name="policy_documents")

print("正在加载模型，请稍候...")
model=SentenceTransformer("BAAI/bge-small-zh-v1.5")

file_path="data"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
HEADING_LINE = re.compile(
    r"^(第[一二三四五六七八九十百0-9]+[章节条]|[一二三四五六七八九十]+、).{0,30}$"
)
SECTION_LINE = re.compile(r"^[一二三四五六七八九十]+、")
MAX_CHARS = 400

def is_heading(text):
    text = text.strip()
    return len(text) <= 40 and bool(HEADING_LINE.match(text))

def split_policy(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces = []
    pending_heading = ""
    for para in paragraphs:
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        if len(lines) >= 2 and is_heading(lines[-1]):
            body = "\n".join(lines[:-1]).strip()
            if body:
                pieces.append((pending_heading, body))
                pending_heading = ""
            pending_heading = "\n".join(
                part for part in (pending_heading, lines[-1]) if part
            )
            continue
        if is_heading(para):
            pending_heading = "\n".join(
                part for part in (pending_heading, para) if part
            )
            continue
        pieces.append((pending_heading, para))
        pending_heading = ""
    if pending_heading:
        pieces.append((pending_heading, ""))

    chunks = []
    for heading, body in pieces:
        sections = [body] if body else [""]
        if body:
            buf = []
            sections = []
            for line in body.splitlines():
                if buf and SECTION_LINE.match(line.strip()):
                    sections.append("\n".join(buf).strip())
                    buf = [line]
                else:
                    buf.append(line)
            if buf:
                sections.append("\n".join(buf).strip())
        for section in sections:
            chunk = "\n".join(part for part in (heading, section) if part).strip()
            if not chunk:
                continue
            if len(chunk) <= MAX_CHARS:
                chunks.append(chunk)
                continue
            buf, size = [], 0
            for line in chunk.splitlines():
                if buf and size + len(line) > MAX_CHARS:
                    chunks.append("\n".join(buf))
                    buf = [heading, line] if heading else [line]
                    size = sum(len(x) for x in buf)
                else:
                    buf.append(line)
                    size += len(line)
            if buf:
                chunks.append("\n".join(buf))
    return chunks

data_dir = "data"

def sliding_window_split(text, filename, chunk_size=200,overlap=50):
    """按固定长度切分，并保留重叠部分防止上下文断裂"""
    if not text:
        return []
    chunks = []
    start=0
    text_length=len(text)

    while start< text_length:
        end=min(start+chunk_size, text_length)
        chunk=text[start:end].strip()
        if chunk:
            chunks.append(f"{filename}\n{chunk}")
        if end==text_length:
            break
        start=end-overlap
    return chunks

def process_file(file_path,chunk_size=200,overlap=50):
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
            documents, metadatas, ids, embeddings = [], [], [], []
            text = f.read()
        if not text.strip():
            print(f"文件 {file_path} 为空，跳过")
            return [], [], [], []
        chunks = sliding_window_split(text, filename, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "source_file": filename,
                "chunk_id": i,
                "platform": platform,
                "date": date_val
            })
            ids.append(f"{filename}_{i:03d}")
            embeddings.append(model.encode(chunk).tolist())

        print(f"✅ 成功处理：{filename}，切分出 {len(documents)} 块。")
        return documents, metadatas, ids, embeddings
    except Exception as e:
        print(f"❌ 处理 {filename} 时出错: {e}")
        return [], [], [], []

def build_db(chunk_size,collection_name):
    print(f"\n================ 开始建库: {collection_name} (Size={chunk_size}) ================")
    try:
        client.delete_collection(name=collection_name)
    except Exception as e:
        pass
    collection=client.get_or_create_collection(name=collection_name)
    data_dir="data"
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".md"):
            continue
        file_path=os.path.join(data_dir, name)
        docs, metas, ids, embs = process_file(file_path, chunk_size=chunk_size)
        if not docs:continue
        collection.upsert(documents=docs, metadatas=metas, ids=ids, embeddings=embs)
        print(f"✅ 入库：{name}，切分出 {len(docs)} 块。")

if __name__ == "__main__":
    build_db(chunk_size=200,collection_name="pdb_200")

    build_db(chunk_size=500,collection_name="pdb_500")