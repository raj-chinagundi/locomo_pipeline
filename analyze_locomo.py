import json
import csv

# Load data
print("Loading locomo_results.json...")
with open('locomo_results.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} questions")

# Extract retrievers
retrievers = ['BM25', 'TF-IDF', 'SVM', 'FAISS', 'Time-Weighted']

# Prepare CSV data
csv_rows = []

for question_data in data:
    question = question_data.get('question', '')
    answer = question_data.get('answer', '')
    evidence = set(question_data.get('evidence', []))
    category = question_data.get('category', '')
    
    # Create row with question info
    row = {
        'question': question,
        'answer': answer,
        'category': category,
        'num_evidence': len(evidence)
    }
    
    # Check each retriever
    any_found = 0
    for retriever in retrievers:
        retrieved_ids = question_data['Retriever'][retriever]['Top5_retrieved_ids']
        retrieved_set = set(retrieved_ids)
        
        # Check if any evidence ID is in retrieved set
        found = 1 if (retrieved_set & evidence) else 0
        row[retriever] = found
        
        if found:
            any_found = 1
    
    row['any_retriever_found'] = any_found
    csv_rows.append(row)

# Write to CSV
output_file = 'locomo_analysis.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['question', 'answer', 'category', 'num_evidence'] + retrievers + ['any_retriever_found']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\n✅ CSV saved to '{output_file}'")
print(f"   Total rows: {len(csv_rows)}")
print(f"   Columns: {len(fieldnames)}")
print(f"\nFirst few rows:")
for i, row in enumerate(csv_rows[:3]):
    print(f"  Row {i+1}: {row['question'][:50]}... | BM25:{row['BM25']} TF-IDF:{row['TF-IDF']} SVM:{row['SVM']} FAISS:{row['FAISS']} Time-Weighted:{row['Time-Weighted']} Any:{row['any_retriever_found']}")

