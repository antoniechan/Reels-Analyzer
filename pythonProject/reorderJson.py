import json

ORIGINAL_FILE = 'temp.json'
SAVED_FILE = 'healthlife_yu'

# 1. Load the original Apify JSON data
# Replace 'apify_output.json' with your actual file path
with open(ORIGINAL_FILE, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 2. Sort the data in descending order of timestamp
# key=lambda x: x['timestamp'] extracts the timestamp from each post dictionary
# reverse=True ensures it sorts from newest (highest) to oldest (lowest)
data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

# 3. Save the sorted data to a new JSON file
output_path = f'reelsData\\{SAVED_FILE}.json'
with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("JSON file successfully sorted in descending order!")
