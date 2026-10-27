import os
import re
import json
import urllib.request
import urllib.error
import jieba
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

client=chromadb.PersistentClient(path="chroma_db")
model=SentenceTransformer("BAAI/bge-small-zh-v1.5")
collection = client.get_collection("pdb_500")
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class HybridRetriever:
    def __init__(self, collection_name="pdb_500"):
        self.collection = client.get_collection(collection_name)
        all_data = self.collection.get()
        self.docs = all_data["documents"] or []
        self.ids = all_data["ids"]
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.threshold = 1.2

    def retrieve(self, query, top_k=3):
        candidate_k = 20
        v_res =self.collection.query(
            query_embeddings=[model.encode(QUERY_PREFIX + query).tolist()],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"]
        )

        min_distance=v_res["distances"][0][0] if v_res["distances"] and len(v_res["distances"]) > 0 else None
        cosine_sim = 1 - min_distance / 2 if min_distance is not None else None
        if min_distance > self.threshold:  # pyright: ignore[reportOptionalOperand]
            return None, None, min_distance
        vector_ids = v_res["ids"][0]

    
        tokenized_query = list(jieba.cut(query))
        doc_scores = self.bm25.get_scores(tokenized_query)  # pyright: ignore[reportAttributeAccessIssue]
        top_n = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:candidate_k]
        bm25_ids = [self.ids[idx] for idx in top_n]

        rrf_score = {}
        for rank, doc_id in enumerate(vector_ids, start=1):
            rrf_score[doc_id] = rrf_score.get(doc_id, 0) + 1 / (rank * 60)

        for rank, doc_id in enumerate(bm25_ids, start=1):
            rrf_score[doc_id] = rrf_score.get(doc_id, 0) + 1 / (rank * 60)

        sorted_docs = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
        final_ids = [doc_id for doc_id, _ in sorted_docs]

        final_docs, final_metas = [], []
        for doc_id in final_ids:
            idx = self.ids.index(doc_id)
            final_docs.append(self.docs[idx])
            final_metas.append(self.collection.get(ids=[doc_id])["metadatas"][0])  # pyright: ignore[reportOptionalSubscript]

        return final_docs, final_metas, min_distance


        

