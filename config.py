import os
import torch
from dotenv import load_dotenv

class Settings:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
    CONVERSATION_LOGS_DIR = os.path.join(LOGS_DIR, "conversations")
    PROMPT_PATH = os.path.join(PROJECT_ROOT, "prompts", "system_prompt.txt")
    KNOWLEDGE_BASE_PATH = "/data/jp-storage/Peter/agent/data/ai-eng-test-sample-knowledges.csv"
    PRODUCTS_PATH = "/data/jp-storage/Peter/agent/data/ai-eng-test-sample-products.csv"
    TEST_QUERIES_PATH = "/data/jp-storage/Peter/agent/data/test.json"
    
    AZURE_ENDPOINT = ""
    API_KEY = ""
    MODEL_TYPE = "gpt-4.1"
    MAX_RETRIES = 1
    PROMPT_PRICE_PER_1K_TOKENS = 0.03 
    COMPLETION_PRICE_PER_1K_TOKENS = 0.06

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    EMBEDDING_MODEL_PATH = "/data/jp-storage/model/embedding_model/bge-m3"
    RERANKER_MODEL_NAME = 'BAAI/bge-reranker-large'

    HYBRID_SEARCH_TOP_K = 10
    RERANK_TOP_N = 3
    FAQ_CONFIDENCE_THRESHOLD = 0.5
    BM25_CONFIDENCE_THRESHOLD = 12.0