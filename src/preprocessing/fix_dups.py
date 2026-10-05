import json
with open('processed_data/split_metadata.json', 'r') as f:
    data = json.load(f)

for split in ['train', 'val', 'test']:
    unique_items = {}
    for item in data[split]:
        unique_items[item['video_path']] = item
    data[split] = list(unique_items.values())

data['train_count'] = len(data['train'])
data['val_count'] = len(data['val'])
data['test_count'] = len(data['test'])

with open('processed_data/split_metadata.json', 'w') as f:
    json.dump(data, f, indent=4)

print(f"Train: {data['train_count']}, Val: {data['val_count']}, Test: {data['test_count']}, Total: {data['train_count'] + data['val_count'] + data['test_count']}")
