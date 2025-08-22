import openai
import time
import json
from typing import Tuple, Dict, Any
from config import Settings
from src.utils.logger import app_logger, cost_logger

class LLMHandler:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not self.settings.API_KEY:
            raise ValueError("Azure OpenAI API Key is not set in environment variables.")
        self.client = openai.AzureOpenAI(
            azure_endpoint=self.settings.AZURE_ENDPOINT,
            api_key=self.settings.API_KEY,
            api_version="2024-02-01"
        )
        app_logger.info(f"LLMHandler initialized for model '{self.settings.MODEL_TYPE}'.")

    def classify_intent(self, query: str) -> Dict[str, str]:
        intent_prompt = f"""
        Analyze the user's query and classify it into ONE of the following intents.
        Respond ONLY with a valid JSON object.

        Intents:
        - "handoff": The user explicitly asks for a human agent. Keywords: 真人, 人工, 客服, agent, human.
        - "product_inquiry": The user is asking about product features, recommendations, comparisons, or specific SKUs.
        - "policy_inquiry": The user is asking about non-product related company policies like shipping, payment, warranty, returns, etc.

        User Query: "{query}"

        JSON Response:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.MODEL_TYPE,
                messages=[
                    {"role": "system", "content": "You are an expert in classifying user intent."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.0,
                max_tokens=50,
                response_format={"type": "json_object"} 
            )
            
            intent_json = json.loads(response.choices[0].message.content)
            app_logger.info(f"Query '{query}' classified with intent: {intent_json.get('intent')}")
            return intent_json

        except Exception as e:
            app_logger.error(f"Intent classification failed for query '{query}'. Error: {e}", exc_info=True)
            return {"intent": "policy_inquiry"}
    
    def generate_response(self, system_prompt: str, user_prompt: str, conversation_id: str) -> Tuple[str, Dict[str, Any]]:
        for attempt in range(self.settings.MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.MODEL_TYPE,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1024
                )

                if response.choices[0].finish_reason == 'content_filter':
                    raise openai.APIError("Response flagged by content filter.", response=None, body=None)

                content = response.choices[0].message.content
                usage = response.usage
                
                self._log_usage(conversation_id, usage)
                return content, usage

            except openai.APIError as e:
                app_logger.warning(f"CONV_ID: {conversation_id} - OpenAI APIError on attempt {attempt + 1}: {e}. Retrying...")
                if "content filter" in str(e):
                    
                    app_logger.error(f"CONV_ID: {conversation_id} - Content filter triggered. The prompt may contain sensitive words.")
                if attempt >= self.settings.MAX_RETRIES:
                    app_logger.error(f"CONV_ID: {conversation_id} - Max retries reached. Failing.")
                    return "抱歉，系統暫時無法處理您的請求。請稍後再試或調整您的問題。", None
                time.sleep(1) 
            except Exception as e:
                app_logger.error(f"CONV_ID: {conversation_id} - An unexpected error occurred: {e}")
                return "抱歉，系統發生未預期的錯誤，請稍後再試。", None
        
        return "抱歉，系統目前無法回應，請稍後再試。", None 

    def _log_usage(self, conversation_id: str, usage: Dict[str, Any]):
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        prompt_cost = (prompt_tokens / 1000) * self.settings.PROMPT_PRICE_PER_1K_TOKENS
        completion_cost = (completion_tokens / 1000) * self.settings.COMPLETION_PRICE_PER_1K_TOKENS
        total_cost = prompt_cost + completion_cost

        cost_logger.info(
            f"CONV_ID: {conversation_id}, "
            f"PROMPT_TOKENS: {prompt_tokens}, COMPLETION_TOKENS: {completion_tokens}, TOTAL_TOKENS: {total_tokens}, "
            f"ESTIMATED_COST_USD: {total_cost:.6f}"
        )