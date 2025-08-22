import numpy as np
import faiss
import jieba 
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from config import Settings
from src.utils.logger import app_logger

class HybridRetriever:
    def __init__(self, documents: List[Dict], settings: Settings):
        self.documents = documents
        self.settings = settings
        self.corpus = [doc['content'] for doc in documents]
        
        app_logger.info("Initializing Jieba for Chinese tokenization...")
        jieba.set_dictionary(os.path.join(settings.PROJECT_ROOT,'src','dict.txt.big'))
        jieba.initialize()

        self.embedding_model = SentenceTransformer(self.settings.EMBEDDING_MODEL_PATH, device=self.settings.DEVICE)
        self.bm25 = None
        self.faiss_index = None
        self._build_indices()

    def _build_indices(self):
        app_logger.info(f"Building BM25 and FAISS indices for {len(self.corpus)} documents...")
        
        app_logger.info("Tokenizing corpus with Jieba for BM25...")
        tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        # ------------------------------------
        
        app_logger.info("Encoding documents for FAISS index...")
        embeddings = self.embedding_model.encode(self.corpus, show_progress_bar=True, normalize_embeddings=True)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings.astype(np.float32))
        app_logger.info("Indices built successfully.")

    def _reciprocal_rank_fusion(self, results: List[List[Tuple[int, float]]], k=60) -> Dict[int, float]:
        fused_scores = {}
        for result_list in results:
            for rank, (doc_index, _) in enumerate(result_list):
                if doc_index not in fused_scores:
                    fused_scores[doc_index] = 0
                fused_scores[doc_index] += 1 / (rank + k)
        
        reranked_results = {k: v for k, v in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)}
        return reranked_results

    def search(self, query: str) -> List[int]:
        tokenized_query = list(jieba.cut_for_search(query))
        
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_results = sorted(
            [(i, score) for i, score in enumerate(bm25_scores)], 
            key=lambda x: x[1], 
            reverse=True
        )[:self.settings.HYBRID_SEARCH_TOP_K]

        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), self.settings.HYBRID_SEARCH_TOP_K)
        faiss_results = list(zip(indices[0], scores[0]))

        fused_results = self._reciprocal_rank_fusion([bm25_results, faiss_results])
        
        final_indices = list(fused_results.keys())
        return final_indices[:self.settings.HYBRID_SEARCH_TOP_K]


class Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = CrossEncoder(self.settings.RERANKER_MODEL_NAME, device=self.settings.DEVICE)
        app_logger.info(f"Reranker model '{self.settings.RERANKER_MODEL_NAME}' loaded on device '{self.settings.DEVICE}'.")
    
    def rerank(self, query: str, documents: List[Dict], original_indices: List[int]) -> List[Dict]:
        pairs = [[query, documents[i]['content']] for i in original_indices]
        if not pairs:
            return []
            
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        reranked_docs = []
        for i, score in enumerate(scores):
            doc = documents[original_indices[i]]
            doc['rerank_score'] = score
            reranked_docs.append(doc)
            
        reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        return reranked_docs[:self.settings.RERANK_TOP_N]