import os
import json
import warnings
import math
import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

# We need keys 6, 7, 8
api_keys = []
for k in ["OPENROUTER_API_KEY_6", "OPENROUTER_API_KEY_7", "OPENROUTER_API_KEY_8"]:
    v = os.getenv(k)
    if v and v.strip() != "" and "sk-or-v1-..." not in v:
        api_keys.append(v)

if not api_keys:
    print("Error: No new OpenRouter API keys (6, 7, 8) found in .env!")
    exit(1)

print(f"Found {len(api_keys)} new OpenRouter API keys for retrying.")

run_config = RunConfig(max_workers=1, timeout=300, max_wait=120)

def create_metrics_for_key(api_key: str):
    base_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=4096, # Increased from 2048 to prevent LLMDidNotFinishException
        max_retries=3,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    judge_llm = LangchainLLMWrapper(base_llm)
    return [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, strictness=1),
        ContextRecall(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
    ]

def main():
    csv_path = "src/evaluation/test_metrics_40_samples.csv"
    json_path = "src/evaluation/eval_results_filtered.json"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Identify rows with NaN OR 0.0 in the metrics columns
    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    
    # Create a mask for NaN values
    mask_nan = df[metric_cols].isnull().any(axis=1)
    
    # Create a mask for exactly 0.0 values
    mask_zero = (df[metric_cols] == 0.0).any(axis=1)
    
    # Combine masks to find any row that has a NaN OR a 0.0
    failed_indices = df[mask_nan | mask_zero].index.tolist()
    
    if not failed_indices:
        print("No failed or zero-score evaluations found!")
        return
        
    print(f"Found {len(failed_indices)} evaluations to retry (contains NaN or 0.0) at indices: {failed_indices}")
    
    # Load original JSON to get the proper types (lists, etc)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    failed_data = [data[i] for i in failed_indices]
    
    # Calculate chunk size
    chunk_size = math.ceil(len(failed_data) / len(api_keys))
    
    for i in range(0, len(failed_data), chunk_size):
        chunk = failed_data[i:i + chunk_size]
        key_index = (i // chunk_size) % len(api_keys)
        current_api_key = api_keys[key_index]
        batch_num = (i // chunk_size) + 1
        
        chunk_indices = failed_indices[i:i + chunk_size]
        print(f"\nRetrying Batch {batch_num} (Original Indices {chunk_indices}) using Key {key_index + 1}...")
        
        dataset = Dataset.from_dict({
            "question":     [item["question"]    for item in chunk],
            "answer":       [item["answer"]       for item in chunk],
            "contexts":     [item["contexts"]     for item in chunk],
            "ground_truth": [item["ground_truth"] for item in chunk],
        })

        metrics = create_metrics_for_key(current_api_key)

        try:
            result = evaluate(
                dataset,
                metrics=metrics,
                embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                run_config=run_config,
            )
            
            df_chunk = result.to_pandas()
            
            # Update the original dataframe with the new values
            for local_idx, orig_idx in enumerate(chunk_indices):
                for col in metric_cols:
                    if col in df_chunk.columns:
                        df.at[orig_idx, col] = df_chunk.at[local_idx, col]
                        
            print(f"Batch {batch_num} retry completed successfully!")
            
        except Exception as e:
            print(f"Error during Batch {batch_num} retry: {e}")

    # Save the updated dataframe
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 40)
    print(" PATCHING COMPLETED! ")
    print(f"Updated CSV saved to: {csv_path}")
    print("=" * 40)

if __name__ == "__main__":
    main()
