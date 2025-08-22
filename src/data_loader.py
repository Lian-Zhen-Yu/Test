import pandas as pd
from typing import List, Dict, Tuple
from config import Settings
from src.utils.logger import app_logger

class DataLoader:
    def __init__(self, settings: Settings):
        self.settings = settings
        app_logger.info(f"DataLoader initialized with knowledge_base: {self.settings.KNOWLEDGE_BASE_PATH} and products: {self.settings.PRODUCTS_PATH}")

    def load_and_chunk(self) -> Tuple[List[Dict], List[Dict]]:
        app_logger.info("Starting data loading and chunking process...")
        product_documents = self._chunk_products()
        faq_documents = self._chunk_knowledge_base()
        app_logger.info(f"Chunking complete. Loaded {len(faq_documents)} FAQ documents and {len(product_documents)} product documents.")
        return faq_documents, product_documents

    def _chunk_products(self) -> List[Dict]:
        documents = []
        try:
            df_products = pd.read_csv(self.settings.PRODUCTS_PATH)
            for _, row in df_products.iterrows():
                specs = f"- 類型: {row.get('specs/arm_type', 'N/A')}\n" \
                        f"- 最大支援尺寸: {row.get('specs/size_max_inch', 'N/A')} 吋\n" \
                        f"- VESA: {row.get('specs/vesa/0', 'N/A')}\n"
                content = f"產品名稱: {row['name']}\nSKU: {row['sku']}\n核心規格:\n{specs}相容性說明: {row.get('compatibility_notes', '無')}"
                metadata = {"source": "product", "sku": row['sku'], "name": row['name'], "url": f"/products/{row['sku']}", "image": row.get('images/0')}
                documents.append({"content": content, "metadata": metadata})
        except FileNotFoundError:
            app_logger.error(f"Product data file not found at {self.settings.PRODUCTS_PATH}")
            raise
        return documents
    
    def _chunk_knowledge_base(self) -> List[Dict]:
        documents = []
        try:
            df_kb = pd.read_csv(self.settings.KNOWLEDGE_BASE_PATH)
            for _, row in df_kb.iterrows():
                content = f"問題分類: {row['title']}\n詳細內容: {row['content']}"
                metadata = {"source": "faq", "title": row['title'], "url": row.get('urls/0/href'), "image": row.get('images/0')}
                documents.append({"content": content, "metadata": metadata})
        except FileNotFoundError:
            app_logger.error(f"Knowledge base file not found at {self.settings.KNOWLEDGE_BASE_PATH}")
            raise
        return documents