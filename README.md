由於對 Agent coding 那些的 function calling 沒用過，只用過 local RAG model，因此針對 FAQ 小題進行回答，並用提供的 json 檔跑，且 coding 大部分是自己完成，而非直接丟 LLM。我後來還是做了簡易的 Routing 去判斷是產品&客服&問題的哪個，因為也只是簡易的分工流程而已。但由於時間問題，我直接麻煩 LLM 幫我摘要我的 coding 去寫 markdown，因此以下是由 LLM 協助摘要為主。

```
使用者查詢 (User Query)
      │
      ▼
┌─────────────────────┐
│ 1. 意圖路由層       │
│ (Intent Routing)    │
└─────────┬───────────┘
          │
          ├─ (意圖: handoff) ───>  [直接回應轉接話術]
          │
          └─ (意圖: inquiry) ───>  ┌───────────────────────┐
                                  │ 2. 進階 RAG 管線        │
                                  │ (Advanced RAG Pipeline) │
                                  └───────────┬───────────┘
                                              │
                                              ▼
                                  ┌───────────────────────┐
                                  │ 3. LLM 回應生成        │
                                  │ (Response Generation) │
                                  └───────────┬───────────┘
                                              │
                                              ▼
                                        [最終回答]
```

### 1\. 意圖路由層 (Intent Routing)

這是系統的「大腦」，在進行任何昂貴的 RAG 搜尋之前，會先對使用者的查詢進行快速分類。

  - **實作位置**: `src/llm_handler.py` -\> `LLMHandler.classify_intent()`
  - **運作方式**:
    1.  使用一個輕量級、專為分類任務設計的 Prompt。
    2.  呼叫 GPT-4.1 模型，將使用者查詢分類為以下幾種核心意圖之一：
          - `handoff`: 使用者明確表示需要人工協助（例如「找真人」、「轉客服」）。
          - `product_inquiry`: 查詢與產品特性、推薦、規格相關。
          - `policy_inquiry`: 查詢與非產品的政策相關（例如付款、運送、保固）。
  - **核心邏輯**:
      - 如果意圖為 `handoff`，系統會**完全跳過 RAG 流程**，直接回傳預設的轉接話術，實現快速、準確的服務升級。
      - 如果意圖為其他類型，則將查詢及分類結果交給下一階段的 RAG 管線處理。

### 2\. 進階 RAG 管線 (Advanced RAG Pipeline)

這是系統的核心資料檢索模組，專為解決關鍵字與語意混合查詢的挑戰而設計。

#### A. 資料分塊 (Chunking)

  - **實作位置**: `src/data_loader.py`
  - **運作方式**:
      - **結構化分塊 (Structured Chunking)**: 針對 `products.csv`，將每個產品的規格（如 SKU, VESA, 尺寸）轉換為一段人類可讀的 Markdown 格式描述，以利模型理解。
      - **語意分塊 (Semantic Chunking)**: 針對 `knowledges.csv`，將每一行（一個完整的問答對）視為一個獨立的語意單元，確保上下文的完整性。

#### B. 「已驗證的黃金門票」檢索策略 (Verified Golden Ticket)

為了解決 RAG 中「關鍵字匹配」被「語意模型」錯誤否決的問題，我們設計了一套更強硬的檢索策略。

  - **實作位置**: `src/orchestrator.py` -\> `JTCG_RAG_Orchestrator.process_query()`
  - **運作方式**:
    1.  **純關鍵字搜尋 (BM25-only Search)**:
          - 使用 `jieba` 進行中文斷詞後，先透過 BM25 演算法找到關鍵字匹配分數最高的**冠軍文件**。
    2.  **雙重驗證 (Verification)**:
          - **分數驗證**：檢查冠軍文件的 BM25 分數是否超過一個預設的信心門檻（`BM25_CONFIDENCE_THRESHOLD`）。
          - **內容驗證**：使用**詞性標註 (POS Tagging)** 提取查詢中的核心名詞，並驗證這些核心詞是否**全部**出現在冠軍文件的內容中。
    3.  **授予黃金門票**:
          - 若**同時通過**分數與內容驗證，該文件將獲得「黃金門票」。它會被直接視為最高信心的答案，**部分繞過**後續流程，強制進入高信心度的回答路徑。
    4.  **常規混合檢索 (Fallback to Hybrid Search)**:
          - 若沒有文件獲得黃金門票，系統則退回到常規的混合檢索流程：
              - **混合搜尋**: 結合 BM25 (關鍵字) 和 FAISS (語意向量) 的搜尋結果，並使用**倒數排序融合 (RRF)** 演算法智慧地合併排序。
              - **重排序 (Reranking)**: 將融合後的候選文件列表交給 `Cross-Encoder` 模型進行最終的精細排序，找出與問題上下文最相關的前 N 篇文件。

### 3\. LLM 回應生成 (Response Generation)

  - **實作位置**: `src/llm_handler.py` -\> `LLMHandler.generate_response()`
  - **運作方式**:
    1.  Orchestrator 根據 RAG 管線的結果（高信心度或低信心度），從 `prompts/system_prompt.md` 中選擇對應的「情境模式」指示。
    2.  將指示、檢索到的上下文（參考資料）和使用者查詢組合為最終的 Prompt。
    3.  呼叫 Azure OpenAI API (`gpt-4.1`) 生成回答。此處包含**內容過濾錯誤的偵測與重試機制**。
    4.  Orchestrator 接收到原始回答後，使用 `OpenCC` 進行**簡轉繁**處理，確保輸出的語言一致性。

-----

## 專案結構

```
jtcg_rag_project/
├── main.py                 # 專案主入口，執行批次處理
├── requirements.txt        # 依賴套件
├── config.py               # 集中管理所有設定
├── .env                    # (需手動建立) 儲存 API 金鑰等敏感資訊
│
├── prompts/
│   └── system_prompt.md    # LLM 的核心行為準則與回應模式定義
│
└── src/
    ├── data_loader.py      # 負責載入與分塊原始資料
    ├── llm_handler.py      # 處理所有與 LLM 的互動 (意圖分類、生成回答)
    ├── rag_pipeline.py     # 實現混合檢索 (BM25+FAISS) 與重排序
    ├── orchestrator.py     # 總指揮，實現意圖路由與「黃金門票」等核心業務邏輯
    └── utils/
        └── logger.py       # 設定與管理所有日誌
```

-----

## 如何執行

1.  **環境設定**

      - 確保 Python 3.10+ 環境。
      - 執行 `pip install -r requirements.txt` 安裝所有依賴。

2.  **配置設定**

      - 在專案根目錄建立 `.env` 檔案，並填入 `AZURE_OPENAI_API_KEY="YOUR_API_KEY"`。
      - 開啟 `config.py`，確認所有檔案路徑 (`KNOWLEDGE_BASE_PATH`, `PRODUCTS_PATH`, `EMBEDDING_MODEL_PATH`, `TEST_QUERIES_PATH`) 均指向正確的位置。

3.  **準備資料**

      - 依照 `config.py` 中的路徑，放置 `ai-eng-test-sample-knowledges.csv`, `ai-eng-test-sample-products.csv`, 和 `test.json`。
      - **(重要)** 從 [Jieba 官方庫](https://www.google.com/search?q=https://raw.githubusercontent.com/fxsjy/jieba/master/jieba/dict.txt.big) 下載 `dict.txt.big` 檔案，並將其放置於 `src/` 目錄下，以達到最佳中文斷詞效果。

4.  **執行批次處理**

    ```bash
    python main.py
    ```

5.  **查看結果**

      - **終端機**: 即時顯示每個問題的處理結果。
      - **`logs/`**: 查看詳細的應用程式日誌、RAG 檢索細節、以及 Token 花費。
      - **`logs/conversations/`**: 每個對話的獨立存檔。
      - **`output/`**: 最終產出的 `result.csv` 和 `result.txt` 報告。
