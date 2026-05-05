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

# ── Suppress deprecation warnings ─────────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# ── 1. GATHER API KEYS ─────────────────────────────────────────────────────
# Find all API keys in the environment starting with OPENROUTER_API_KEY_
api_keys = []
for k, v in os.environ.items():
    if k.startswith("OPENROUTER_API_KEY_") and v.strip() != "" and "sk-or-v1-..." not in v:
        api_keys.append(v)

# Fallback to the default key if no numbered keys were found
if not api_keys and os.getenv("OPENROUTER_API_KEY"):
    api_keys.append(os.getenv("OPENROUTER_API_KEY"))

if not api_keys:
    print("❌ Error: No OpenRouter API keys found in .env! Please add them.")
    exit(1)

print(f"🔑 Found {len(api_keys)} OpenRouter API keys. We will rotate them to bypass rate limits!")

# ── 2. RUN CONFIG ──────────────────────────────────────────────────────────
run_config = RunConfig(
    max_workers=1,
    timeout=240,
    max_wait=120,
)

def create_metrics_for_key(api_key: str):
    """Initializes the LLM and Ragas metrics with a specific API key."""
    base_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=2048,
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
    input_path = "src/evaluation/eval_results_filtered.json"
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use all data
    test_data = data
    print(f"📊 Total samples to evaluate: {len(test_data)}")

    # Calculate chunk size to distribute evenly among keys
    chunk_size = math.ceil(len(test_data) / len(api_keys))
    
    all_results_dfs = []

    # Process in chunks
    for i in range(0, len(test_data), chunk_size):
        chunk = test_data[i:i + chunk_size]
        key_index = (i // chunk_size) % len(api_keys)
        current_api_key = api_keys[key_index]
        
        batch_num = (i // chunk_size) + 1
        print(f"\n⏳ Starting Batch {batch_num} (Samples {i+1} to {i+len(chunk)}) using Key {key_index + 1}...")

        dataset = Dataset.from_dict({
            "question":     [item["question"]    for item in chunk],
            "answer":       [item["answer"]       for item in chunk],
            "contexts":     [item["contexts"]     for item in chunk],
            "ground_truth": [item["ground_truth"] for item in chunk],
        })

        # Initialize metrics with this batch's key
        metrics = create_metrics_for_key(current_api_key)

        try:
            result = evaluate(
                dataset,
                metrics=metrics,
                embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                run_config=run_config,
            )
            
            df_chunk = result.to_pandas()
            all_results_dfs.append(df_chunk)
            print(f"✅ Batch {batch_num} completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during Batch {batch_num}: {e}")

    # Concatenate all DataFrames and save
    if all_results_dfs:
        final_df = pd.concat(all_results_dfs, ignore_index=True)
        report_path = "src/evaluation/test_metrics_40_samples.csv"
        final_df.to_csv(report_path, index=False)
        
        print("\n" + "=" * 40)
        print(" 🎉 FULL EVALUATION COMPLETED! 🎉")
        print("=" * 40)
        print(f"Total Rows Evaluated: {len(final_df)}")
        print(f"Detailed CSV saved to: {report_path}")
        print("=" * 40)
    else:
        print("\n❌ No results were generated. All batches failed.")

if __name__ == "__main__":
    main()