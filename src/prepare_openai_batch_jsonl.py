import logging
import json
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_dataset(dataset):
    processed_data = []
    for item in dataset['train']:
        processed_data.append(item['input'])
    logging.info(f"Processed {len(processed_data)} items from the dataset")
    return processed_data

def main():
    # Load the dataset
    dataset = load_dataset("verifiers-for-code/fitlered-1k-from-546k")
    
    processed_data = process_dataset(dataset)

    # Read system prompt
    with open('data/sys.txt', 'r', encoding='utf-8') as file:
        system_prompt = file.read().strip()

    # Prepare JSONL data
    with open('data/batch_1k.jsonl', 'w', encoding='utf-8') as outfile:
        for i, content in enumerate(processed_data, 1):
            data = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-2024-08-06",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    "max_tokens": 1024,
                    "temperature": 1.0,
                    "response_format": {"type": "json_object"}
                }
            }
            json.dump(data, outfile)
            outfile.write('\n')

    logging.info(f"Processed {len(processed_data)} items and wrote to data/batch_1k.jsonl")

if __name__ == "__main__":
    main()