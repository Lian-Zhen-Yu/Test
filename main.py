import json
import os
import pandas as pd
from config import Settings
from src.orchestrator import JTCG_RAG_Orchestrator
from src.utils.logger import app_logger

def main():
    """
    Main execution function, reads a test queries JSON file and performs batch processing.
    This version is updated to parse the {"messages": [...]} structure.
    """
    app_logger.info("=============================================")
    app_logger.info("=== JTCG RAG Batch Processing Job Started ===")
    app_logger.info("=============================================")

    try:
        settings = Settings()
        orchestrator = JTCG_RAG_Orchestrator(settings)
        
        app_logger.info(f"Loading test queries from {settings.TEST_QUERIES_PATH}")
        with open(settings.TEST_QUERIES_PATH, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        results = []
        for i, conversation_obj in enumerate(test_data):
            user_content = "INVALID_QUERY_FORMAT"
            try:
                if not isinstance(conversation_obj, dict) or "messages" not in conversation_obj or not conversation_obj["messages"]:
                    raise ValueError("Invalid conversation object format: missing 'messages' key or empty messages list.")
                
                first_user_message = None
                for message in conversation_obj["messages"]:
                    if message.get("role") == "user":
                        first_user_message = message
                        break 
                
                if not first_user_message or "content" not in first_user_message:
                     raise ValueError("No valid user message found in this conversation object.")

                user_content = first_user_message["content"]


                app_logger.info(f"--- Processing query {i+1}/{len(test_data)}: '{user_content}' ---")
                
                bot_response = orchestrator.process_query(user_content)
                
                result = {
                    "query": user_content,
                    "response": bot_response
                }
                results.append(result)
                
                print("="*80)
                print(f"Query ({i+1}/{len(test_data)}): {user_content}")
                print(f"Response: {bot_response}")
                print("="*80 + "\n")

            except (KeyError, IndexError, ValueError) as e:
                app_logger.warning(f"Skipping malformed entry #{i+1} in test.json. Error: {e}. Data: {conversation_obj}")
                print(f"\nWARNING: Skipped malformed entry #{i+1}. Check app.log for details.\n")
                continue
            
            except Exception as e:
                app_logger.error(f"An unexpected error occurred while processing query: '{user_content}'. Error: {e}", exc_info=True)
                print(f"Error processing query: '{user_content}'. See app.log for details.")

        if results:
            app_logger.info(f"Processing {len(results)} results for CSV output.")
            csv_data = []
            for idx, result in enumerate(results):
                conversation_text = (
                    f"User: {result['query']}\n"
                    f"--------------------\n"
                    f"Bot: {result['response']}"
                )
                csv_data.append({
                    "id": idx + 1,
                    "conversation": conversation_text
                })
            
            try:
                df = pd.DataFrame(csv_data)
                df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                app_logger.info(f"Successfully saved {len(df)} results to {output_csv_path}")
                print(f"\n✅ Results successfully saved to {output_csv_path}")
            except Exception as e:
                app_logger.error(f"Failed to save results to CSV. Error: {e}", exc_info=True)
                print(f"\n❌ Failed to save results to CSV. Check app.log for details.")

            app_logger.info(f"Processing {len(results)} results for TXT output.")
            try:
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    for idx, result in enumerate(results):
                        f.write("="*80 + "\n")
                        f.write(f"對話紀錄 #{idx + 1}\n")
                        f.write("="*80 + "\n")
                        f.write(f"User: {result['query']}\n\n")
                        f.write(f"Bot: {result['response']}\n")
                        f.write("\n\n") 
                app_logger.info(f"Successfully saved plain text results to {output_txt_path}")
                print(f"✅ Plain text results also saved to {output_txt_path}")
            except Exception as e:
                app_logger.error(f"Failed to save results to TXT. Error: {e}", exc_info=True)
                print(f"\n❌ Failed to save results to TXT. Check app.log for details.")
        
        else:
            app_logger.warning("No valid results were generated, skipping CSV output.")
            print("\n⚠️ No results were generated, CSV file was not created.")
    
    except Exception as e:
        app_logger.critical(f"A critical error occurred during initialization or file loading: {e}", exc_info=True)
        print(f"A critical error occurred. Please check app.log for details.")

    app_logger.info("==========================================")
    app_logger.info("=== JTCG RAG Batch Processing Job Ended ===")
    app_logger.info("==========================================")


if __name__ == "__main__":
    main()