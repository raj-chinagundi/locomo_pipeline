import json
import sys
import os

class CustomData:
    def __init__(self, path_locomo, path_longmem_eval):
        if path_locomo:
            self.locomo_data = json.load(open(path_locomo))
        if path_longmem_eval:
            self.longmem_eval_data = json.load(open(path_longmem_eval))

    def process_locomo(self):
        processed_data = {}
        for item in self.locomo_data:
            conv_id = item['sample_id']
            collection_key = f"vectordb_collection_{conv_id}"
            conversation = item['conversation']
            if collection_key not in processed_data:
                processed_data[collection_key] = []
            # Find all session_X_date_time keys
            for key in conversation:
                if key.startswith("session_") and key.endswith("_date_time"):
                    session_num = key[len("session_"): -len("_date_time")]
                    session_key = f"session_{session_num}"
                    timestamp = conversation[key]
                    session_dialogues = conversation.get(session_key, [])
                    for dialogue in session_dialogues:
                        meta = {
                            "speaker": dialogue.get("speaker"),
                            "dia_id": dialogue.get("dia_id"),
                            "text": dialogue.get("text"),
                            "timestamp": timestamp
                        }
                        processed_data[collection_key].append(meta)
        return processed_data
    
    def load_source_qa_locomo(self):
        questions_answers = {}
        for item in self.locomo_data:
            conv_id = item['sample_id']
            questions_answers[conv_id] = []
            qa_pair = item['qa']
            for q in qa_pair:
                question = q.get('question')
                answer = q.get('answer')
                category = q.get('category')
                if answer is None:
                    continue
                evidence_ids = q.get('evidence', [])
                questions_answers[conv_id].append({
                    'question': question,
                    'answer': answer,
                    'evidence_ids': evidence_ids,
                    'category': category
                })
        return questions_answers

    def load_source_qa_longmem_eval(self):
        questions_answers = {}
        for item in self.longmem_eval_data:
            question_id = item.get('question_id')
            question = item.get('question')
            answer = item.get('answer')
            if answer is None:
                continue
            # Use answer_session_ids as evidence_ids
            evidence_ids = item.get('answer_session_ids', [])
            questions_answers[question_id] = [{
                'question': question,
                'answer': answer,
                'evidence_ids': evidence_ids
            }]
        return questions_answers
     
    def process_longmem_eval(self):
        output = {}
        for q in self.longmem_eval_data:
            question_id = q.get('question_id')
            haystack_dates = q.get('haystack_dates', [])
            haystack_session_ids = q.get('haystack_session_ids', [])
            haystack_sessions = q.get('haystack_sessions', [])
            collection_key = f'vectordb_collection_{question_id}'
            session_entries = []
            for idx in range(min(len(haystack_dates), len(haystack_session_ids), len(haystack_sessions))):
                session_entry = {
                    'haystack_session_id': haystack_session_ids[idx],
                    'haystack_date': haystack_dates[idx],
                    'haystack_session': haystack_sessions[idx],
                    'answer_id': haystack_session_ids[idx]
                }
                session_entries.append(session_entry)
            output[collection_key] = session_entries
        return output

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    data_processor = CustomData(
        path_locomo=os.path.join("data/locomo10.json"),
        path_longmem_eval=os.path.join("data/longmemeval_s_cleaned.json")
    )
    locomo_processed = data_processor.load_source_qa_locomo()
    longmem_eval_processed = data_processor.load_source_qa_longmem_eval()
    print(locomo_processed)
    # print(longmem_eval_processed)

