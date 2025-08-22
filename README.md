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

## 以下是針對整份考題的預期回答準備方式 (無準備coding)


```
/jtcg-crm-agent-python-list
├── main.py                 
├── config/
│   ├── models.yaml           # LLM, Embedding, Reranker 模型選型與 API Keys
│   ├── thresholds.yaml       # RAG 信心度、轉真人...等業務規則門檻
│   └── language.yaml         # 語言設定
│
├── agent/                     # 整個 AI-agnet 核心架構
│   ├── tool_executor.py       # Tool Execution : 選擇工具
│   ├── intent_router.py       # Intent Recognition : 行為判斷
│   ├── state_manager.py       # State Management : 上下文狀態與對話流暢
│   ├── response_generation.py # 產出回應
│   └── memory.py              # Memory : 對話記憶
│
├── prompts/
│   ├── system_prompt.md       # Agent system prompt
│   ├── handoff_prompt.md      # Agent 轉客服的摘要 prompt
│   └── few_shot_examples.json # 用於 Dynamic few-shot
│
├── tools/
│   ├── __init__.py          
│   ├── product_tools.py      # 產品搜尋、比較、過濾等 Function Calling
│   ├── order_tools.py        # 訂單查詢、追蹤等 Function Calling
│   └── handoff_tools.py      # 真人客服轉接、生成過程摘要等 Function Calling。包含收集 Email 以便轉接真人客服，以及回應給使用者的 format 格式。
│   ├── safety.py             # 惡意行為等 filter。
│
├── rag_pipeline/
│   ├── index_builder.py      # 資料 chunk ... 等
│   ├── retriever.py          # 檢索器：Hybrid Search ... 等。
│   └── reranker.py           # Rerank base 的 二次檢索
│
├── data/
│   ├── products.csv          # 原始產品資料
│   ├── faqs.csv              # 原始 FAQ 知識庫
│   └── others.csv            # if need
│
├── utils/
│   ├── locale.py              # 語系偵測/繁簡轉換
│   └── citations.py           # 針對 URL 等整理
│
└── evaluation/               
    ├── golden_dataset.jsonl  # 用於評估的標準問答對與預期行為
    ├── metrics.py            # 忠誠度、相關性、recall ... 等
    └── run_evaluation.py     # 自動化評估腳本
│
└── requirements.txt
```

## 核心設計：混合式 ReAct 框架

預計採用 **混合式 ReAct 框架**。只有在需要進行複雜決策、綜合資訊或處理模糊語意時，才會啟動 LLM 的 Reasoning 能力。在此專案應可兼顧效率、成本與模型能力。

### Agent 核心運作內容

1.  **Intent Recognition (`intent_router.py`)**:
    由 LLM 擔任意圖識別的角色，快速分辨使用者意圖(查訂單 or 問問題 or 推薦產品) 來提取關鍵 Entities。包含真人轉客服的關鍵字機制，以及負面語意判斷、Agnet 針對同一題循環問答超過 n 次...等。

2.  **State Management (`state_manager.py`)**:
    處理多輪互動中的上下文狀態，保持對話流暢，也並避免重複詢問已知資訊。

3.  **Tool Execution (`tool_executor.py`)**:
    根據識別出的意圖，調用對應的 Tool 或啟動 RAG 流程。

4.  **Response Generation (`response_generation.py`)**:
    整合所有檢索到的上下文、Tool 與對話歷史，由 LLM 根據 System Prompt 中定義的品牌口吻和格式要求，產出最終回覆。同時設定 RAG threshold, 超過指定分數判定為回答信心不足，改用繼續引導詢問深入問題等回答。

5.  **Memory (`memory.py`)**:

      * **儲存**: 每次對話結束後，觸發 LLM 評估是否需將此次互動的關鍵資訊摘要後存入長期記憶。
      * **檢索**: 在新對話開始時，動態檢索該使用者的過往互動記憶，作為 Dynamic Few-shot 。

-----

## 模型選擇與實驗步驟

### 1\. Embedding Model

  * **首選模型**: `text-embedding-3-small`，主要考量其高性價比。(但其實我都用 local model，沒用過雲端 model)
  * **候選模型**: 我也會快速尋找市場上在中英文客服場景中表現優異的其他雲端模型。
  * **實驗設計**:
    1.  建立一個 JTCG 專屬的標準評測資料集，並由人工標註正確答案。
    2.  使用 **Mean Reciprocal Rank (MRR)** 與 **Normalized Discounted Cumulative Gain (NDCG)** 作為核心指標，量化評估各候選模型的檢索效能。最後再挑選出合適的 Embedding model。


### 2\. Agent LLM Model

  * **首選模型**: `GPT-4.1`。
  * **選擇理由**: 速度快、成本相對較低。根據個人實測，其在工具調用和遵循指令的成效上優於 `GPT-4o`。
  * **實驗設計**: 比照 Embedding model 設計一個資料集。
  * **評估指標**:
      * 任務成功率 (Task Success Rate)
      * 工具呼叫準確率 (Tool Calling Accuracy)
      * 忠誠度 (Faithfulness)

接著根據實驗結果決定應如何改善流程、RAG、Prompt...等，並控制變因進行實驗。

-----

## 可優化方向

1.  **檢索策略 (Hybrid Search)**: 結合 **BM25 (關鍵字搜索)** 和 **Dense Vector Search (語意搜索)**。前者用於精準捕捉領域術語，後者用於理解語意相似度。

2.  **重排序 (Reranking)**: 在初步檢索 Top-20 文件後，使用一個輕量級的 **Cross-Encoder 模型（如 `bge-reranker`）** 進行二次排序，僅將最相關的 Top-3 結果送入 LLM。預期能提升上下文品質、降低 Token 成本，並顯著提高忠誠度。同時針對關鍵字搜尋，也會使用"保送"的方式進入排序，避免被 Reranker 機制清掉。

3.  **分塊策略 (Chunking)**:

      * **FAQ**: 採用 **語意分塊 (Semantic Chunking)**，確保每個 Chunk 都是一個完整的問答對。
      * **產品資料**: 採用 **結構化分塊 (Structured Chunking)**，將規格、描述、相容性等欄位轉換為結構化的 Markdown 文本，讓 LLM 更易於解析。

4.  **自動化評估框架**: 使用 **RAGAS** 等框架，建立自動化 RAG 來監控 **忠誠度**、**答案相關性** 和**上下文 recall **，作為 RAG 優化基礎。

5.  **知識圖譜 (Knowledge Graph)**: 對於產品推薦等複雜關聯場景，可引入圖資料庫 (如 Neo4j) 或直接建立 Knowledge Graph。相關的 **Graph RAG** 技術已有發展，預期能提供比傳統結構化資料更佳的推薦效果。例如：(JTCG-ARM-DUAL-PRO-32) -[支援最大尺寸]-> (16吋)。

6.  **從記憶中發現趨勢**: 設計一個分析型 Agent，定期掃描並分析 Memory 中儲存的對話紀錄，以發現潛在的客戶需求、產品問題或新的商業趨勢。

7.  **Prompt Cache**: 對於高頻且重複的查詢降低 API 費用與延遲。

8.  **Dynamic Few-shot**: 在對話開始時，不僅是從歷史紀錄，也可以從高品質範例庫 (`few_shot_examples.json`) 中動態檢索與當前問題最相關的範例，注入到 Prompt 中以提升模型表現。

9.  **框架選擇考量**: 雖然 **Tree of Thoughts** 或 **Monte Carlo Tree Search** 等複雜搜索算法在通用問題解決中非常強大，但對於 CRM 這類使用者旅程相對線性的場景，我自己認為可能會帶來巨大的延遲和成本。因此，我考慮更直接的 **ReAct-style 混合框架**，在效率和能力之間應該可以有更佳的平衡。

10. **Model Routing** : 若效能允許，可考慮同時使用 GPT-4.1 and GPT 3.5 等，前者負責複雜情境或需 Reasoning 等，後者則為常態使用。如此預期可讓效能更佳成本也更便宜。但需經實驗確保。

-----

## 實際預期案例

**使用者輸入:** 「我需要一個新螢幕，大概 16 吋，但我的桌面有點厚，怕夾不住。」

-----

**Step 1: `intent_router.py` (意圖識別)**

  * **預期行為**: 偵測到使用者意圖為**產品推薦**並伴隨一個**疑問**。
  * **預期 Output**:
    ```json
    {
      "intents": ["product_recommendation", "faq_query"],
      "entities": {
        "screen_size_inch": 16,
        "concern": "desk_thickness_compatibility"
      }
    }
    ```

-----

**Step 2: `state_manager.py` (狀態管理)**

  * **預期行為**: 記錄已知的 Entities，並將對話狀態更新為等待工具執行。
  * **預期 Output**:
    ```json
    {
      "session_id": "test",
      "known_entities": {
        "screen_size_inch": 16,
        "concern": "desk_thickness_compatibility"
      },
      "status": "AWAITING_TOOL_EXECUTION"
    }
    ```

-----

**Step 3: `tool_executor.py` (工具執行)**

  * **預期行為**: 在此案例中並行觸發兩個任務：
    1.  **任務 A (產品探索)**: 呼叫 `product_tools.search_products`，參數為 `{'size_inch': '>=16'}`。
    2.  **任務 B (FAQ 檢索)**: 呼叫 `rag_pipeline.retriever.search`，查詢為 `'桌面厚度 夾具'`。
  * **預期 Output (範例)**:
    ```json
    {
      "tool_outputs": [
        {
          "tool_name": "search_products",
          "result": "[找到的產品列表...]"
        },
        {
          "tool_name": "search_faq",
          "result": "[關於桌面夾具厚度限制的 FAQ 內容...]"
        }
      ]
    }
    ```

-----

**Step 4: `response_generation.py` (回應生成)**

  * **預期行為**: 將工具輸出、對話歷史與 System Prompt 一起打包 (且考慮 dynamic few-shot)，發送給核心 Agent LLM 產出最終回覆。
  * **預期 Output**:
    > 好的，針對您 16 吋螢幕與桌板厚度的需求，我推薦這款產品： **[產品名稱 JTCG-ULTRA-16]**, **推薦原因**: ... , **關於桌板厚度**, : 根據我們的資料，這款螢幕支架的夾具支援厚度為 .... 能適用於市面上絕大多數的桌板。 **產品連結**: [產品頁面連結] .... 。為了給您更精準的建議，可以告訴我您桌面的大概厚度與材質嗎？

-----

**Step 5: `memory.py` (記憶儲存)**

  * **預期行為**: 對話結束後，LLM 判斷此次互動具有價值，生成一個結構化的記憶摘要。
  * **預期 Output (存入 VectorDB 的內容)**:
    ```json
    {
      "intent": ["product_discovery", "faq"],
      "query_keywords": ["16吋", "桌板厚"],
      "successful_recommendation": "JTCG-ULTRA-16",
      "resolved_faq": "FAQ-DESK-006"
    }
    ```
    這個摘要會被 Embedding 並儲存到長期記憶的 VectorDB 中，用於未來記憶體控管搜尋與 Dynamic Few-shot 檢索。
