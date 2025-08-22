import random
import uuid
import json
import jieba
import jieba.posseg as pseg 
import numpy as np
import re
from opencc import OpenCC
from typing import List, Dict
from config import Settings
from src.data_loader import DataLoader
from src.rag_pipeline import HybridRetriever, Reranker
from src.llm_handler import LLMHandler
from src.utils.logger import app_logger, rag_logger, log_conversation

class JTCG_RAG_Orchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        app_logger.info("Initializing JTCG RAG Orchestrator...")
        with open(self.settings.PROMPT_PATH, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
        self.s2t_converter = OpenCC('s2t.json')
        loader = DataLoader(self.settings)
        self.faq_docs, self.product_docs = loader.load_and_chunk()
        self.faq_retriever = HybridRetriever(self.faq_docs, self.settings)
        self.reranker = Reranker(self.settings)
        self.llm_handler = LLMHandler(self.settings)
        app_logger.info("Orchestrator initialized successfully.")

    def _extract_critical_keywords(self, query: str) -> List[str]:
        allowed_pos = {'n', 'nr', 'ns', 'nt', 'eng'}
        keywords = [
            word.lower() for word, flag in pseg.cut(query) 
            if flag in allowed_pos and len(word.strip()) > 0
        ]
        return list(dict.fromkeys(keywords)) 

    def process_query(self, query: str) -> str:
        conversation_id = uuid.uuid4()
        rag_log_extra = {'conv_id': conversation_id}
        
        app_logger.info(f"CONV_ID: {conversation_id} - Processing new query: '{query}'")

        intent_result = self.llm_handler.classify_intent(query)
        intent = intent_result.get("intent", "policy_inquiry")
        rag_logger.debug(f"Classified intent: '{intent}'", extra=rag_log_extra)

        if intent == "handoff":
            app_logger.info(f"CONV_ID: {conversation_id} - Handoff intent detected. Skipping RAG.")
            final_answer = "已為您轉接真人客服，請稍候。"
        else:
            app_logger.info(f"CONV_ID: {conversation_id} - Executing Verified Golden Ticket RAG flow.")

            tokenized_query_for_bm25 = list(jieba.cut_for_search(query))
            bm25_scores = self.faq_retriever.bm25.get_scores(tokenized_query_for_bm25)
            
            top_bm25_index = np.argmax(bm25_scores)
            top_bm25_score = bm25_scores[top_bm25_index]
            
            rag_logger.debug(f"Top BM25 candidate index: {top_bm25_index}, Score: {top_bm25_score:.4f}", extra=rag_log_extra)
            
            golden_ticket_doc = None
            
            if top_bm25_score > self.settings.BM25_CONFIDENCE_THRESHOLD:
                critical_keywords = self._extract_critical_keywords(query)
                rag_logger.debug(f"Extracted critical keywords for verification: {critical_keywords}", extra=rag_log_extra)
                
                top_doc_content = self.faq_docs[top_bm25_index]['content'].lower()

                if critical_keywords and all(keyword in top_doc_content for keyword in critical_keywords):
                    app_logger.info(f"CONV_ID: {conversation_id} - Verified Golden Ticket MATCH! BM25 score ({top_bm25_score:.4f}) is above threshold AND all keywords found.")
                    golden_ticket_doc = self.faq_docs[top_bm25_index]
                    top_score = 1.0 
                    faq_results = [golden_ticket_doc]
            
            if golden_ticket_doc is None:
                app_logger.info(f"CONV_ID: {conversation_id} - No Golden Ticket. Proceeding with full hybrid search and reranking.")
                hybrid_indices = self.faq_retriever.search(query)
                final_candidate_indices = list(dict.fromkeys([top_bm25_index] + hybrid_indices))
                faq_results = self.reranker.rerank(query, self.faq_docs, final_candidate_indices)
                top_score = faq_results[0].get('rerank_score', 0) if faq_results else 0

            if faq_results:
                rag_logger.debug(f"Reranked Top-{len(faq_results)} FAQ(s). Top score: {top_score:.4f}", extra=rag_log_extra)
                rag_logger.debug(f"Top reranked doc content: {faq_results[0]['content'][:200]}...", extra=rag_log_extra)
            else:
                rag_logger.debug("No relevant FAQs found after reranking.", extra=rag_log_extra)
            
            if top_score >= self.settings.FAQ_CONFIDENCE_THRESHOLD:
                app_logger.info(f"CONV_ID: {conversation_id} - High confidence path triggered. Top score: {top_score:.4f}")
                user_prompt = self._build_direct_prompt(query, faq_results)
                rag_logger.debug(f"Final User Prompt (Direct):\n{user_prompt}", extra=rag_log_extra)
            else:
                app_logger.info(f"CONV_ID: {conversation_id} - Low confidence path triggered. Top score: {top_score:.4f}")
                if intent == "product_inquiry":
                    user_prompt = self._build_product_fallback_prompt(query)
                else: 
                    user_prompt = self._build_generic_fallback_prompt(query)
                rag_logger.debug(f"Final User Prompt (Fallback):\n{user_prompt}", extra=rag_log_extra)

            raw_answer, _ = self.llm_handler.generate_response(self.system_prompt, user_prompt, conversation_id)
            final_answer = self.s2t_converter.convert(raw_answer) if raw_answer else raw_answer

        log_conversation(conversation_id, query, final_answer)
        app_logger.info(f"CONV_ID: {conversation_id} - Successfully processed query.")
        
        return final_answer

    def _get_product_samples(self, count=3) -> List[Dict]:
        return random.sample(self.product_docs, min(count, len(self.product_docs)))
        
    def _build_direct_prompt(self, query: str, context_docs: List[Dict]) -> str:
        context = ""
        for i, doc in enumerate(context_docs):
            metadata = doc['metadata']
            context += f"--- 參考資料 {i+1} ---\n"
            context += f"內容: {doc['content']}\n"
            if metadata.get('url'):
                context += f"參考連結: {metadata['url']}\n"
            context += "-----------------\n\n"
        return f"情境模式: 直接回答模式\n\n[參考資料]\n{context}\n\n[提問]\n{query}"
        
    def _build_product_fallback_prompt(self, query: str) -> str:
        product_samples = self._get_product_samples()
        context = ""
        for i, doc in enumerate(product_samples):
            context += f"--- 產品範例 {i+1} ---\n"
            context += f"{doc['content']}\n"
            context += f"參考連結: {doc['metadata']['url']}\n"
            context += "-----------------\n\n"
        return f"情境模式: 購物引導模式\n\n使用者的原始問題是：「{query}」。\n由於在 FAQ 中找不到直接答案，請根據以下「產品範例」，生成一段友善的回應，向使用者介紹我們產品的大致方向，並引導他們提供更具體的需求。\n\n[產品範例]\n{context}"
        
    def _build_generic_fallback_prompt(self, query: str) -> str:
        return f"情境模式: 通用協助模式\n\n使用者的原始問題是：「{query}」。\n由於在 FAQ 中找不到直接答案，請根據你在通用協助模式下的行為準則來回應。"