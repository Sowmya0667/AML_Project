import json

input_file = "src/evaluation/eval_results_ready_for_ragas.json"
output_file = "src/evaluation/eval_results_filtered.json"

with open(input_file, "r", encoding='utf-8') as f:
    data = json.load(f)

# Filter criteria: 
# 1. 'contexts' must not be empty
# 2. 'answer' must not contain the word "Error" (case insensitive)
filtered_data = [
    item for item in data 
    if len(item['contexts']) > 0 and "error" not in item['answer'].lower()
]

with open(output_file, "w", encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=4, ensure_ascii=False)

print(f"📊 Original count: {len(data)}")
print(f"✅ Filtered count: {len(filtered_data)}")
print(f"🚀 Filtered file saved as: {output_file}")