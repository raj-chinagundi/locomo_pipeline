import json
import csv

# Load data
print("Loading longmem_results.json...")
with open('longmem_results.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} questions")

# Extract retrievers
retrievers = ['BM25', 'TF-IDF', 'SVM', 'FAISS', 'Time-Weighted']

# Prepare CSV data
csv_rows = []

for question_data in data:
    question_id = question_data.get('question_id', '')
    question_type = question_data.get('question_type', '')
    question = question_data.get('question', '')
    question_date = question_data.get('question_date', '')
    answer = question_data.get('answer', '')
    answer_session_ids = set(question_data.get('answer_session_ids', []))
    
    # Create row with question info
    row = {
        'question_id': question_id,
        'question_type': question_type,
        'question': question,
        'question_date': question_date,
        'answer': answer,
        'num_answer_session_ids': len(answer_session_ids)
    }
    
    # Check each retriever
    any_found = 0
    for retriever in retrievers:
        retrieved_ids = question_data['Retriever'][retriever]['Top5_retrieved_ids']
        retrieved_set = set(retrieved_ids)
        
        # Check if any answer_session_id is in retrieved set
        found = 1 if (retrieved_set & answer_session_ids) else 0
        row[retriever] = found
        
        if found:
            any_found = 1
    
    row['any_retriever_found'] = any_found
    csv_rows.append(row)

# Write to CSV
output_file = 'longmem_analysis.csv'
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['question_id', 'question_type', 'question', 'question_date', 'answer', 'num_answer_session_ids'] + retrievers + ['any_retriever_found']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\n✅ CSV saved to '{output_file}'")
print(f"   Total rows: {len(csv_rows)}")
print(f"   Columns: {len(fieldnames)}")
print(f"\nFirst few rows:")
for i, row in enumerate(csv_rows[:3]):
    print(f"  Row {i+1}: {row['question'][:50]}... | BM25:{row['BM25']} TF-IDF:{row['TF-IDF']} SVM:{row['SVM']} FAISS:{row['FAISS']} Time-Weighted:{row['Time-Weighted']} Any:{row['any_retriever_found']}")

