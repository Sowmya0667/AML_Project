import sys
import os
import pandas as pd
import json

# ── 1. PATH SETUP ────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_chain import LegalRAGChain 

def main():
    # ── 2. LOAD DATA ─────────────────────────────────────────────────────────
    csv_path = os.path.join("src", "evaluation", "eval_queries_final_2020.csv")
    if not os.path.exists(csv_path):
        print(f" Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # ── 3. INITIALIZE COMPONENTS ─────────────────────────────────────────────
    print(" Initializing Pipeline Components...")
    # Note: LegalRAGChain already initializes its own HybridRetriever in __init__
    generator = LegalRAGChain()

    eval_data = []

    # ── 4. EXECUTION LOOP ────────────────────────────────────────────────────
    for index, row in df.iterrows():
        query_id = row['query_id']
        original_query = row['paraphrased_query']
        ground_truth = row['context_holding']
        
        print(f" Processing {index+1}/{len(df)}: {query_id}")

        try:
            # STEP A & B: Run the chain
            # Your class handles retrieval and generation inside .query()
            response = generator.query(original_query)
            
            # STEP C: Extract text from the chunks returned in the response
            # We access response.sources or we can look at generator.last_result
            context_strings = []
            if hasattr(generator, 'last_result') and generator.last_result.chunks:
                for chunk in generator.last_result.chunks:
                    # Using .text based on your _format_context helper
                    context_strings.append(chunk.text if hasattr(chunk, 'text') else str(chunk))
            
            # STEP D: Bundle for Ragas
            eval_data.append({
                "question": original_query,
                "answer": response.answer, # Extracting the string from the response object
                "contexts": context_strings,
                "ground_truth": ground_truth
            })
            
        except Exception as e:
            print(f" Error processing {query_id}: {str(e)}")
            continue

    # ── 5. SAVE RESULTS ──────────────────────────────────────────────────────
    output_path = "src/evaluation/eval_results_ready_for_ragas.json"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(eval_data, f, indent=4, ensure_ascii=False)

    print(f"\n Done! {len(eval_data)} rows saved to {output_path}")

if __name__ == "__main__":
    main()