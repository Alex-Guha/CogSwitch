import os
import json
import re

MODEL_NAME = 'gpt-4o-mini'

base_dir = 'data/base'
puzzle_dir = 'data/train'
test_dir = 'data/test'
output_dir = f'data/reasonings/{MODEL_NAME}'


for grid_file in os.listdir(output_dir):
    if not grid_file.endswith(".json"):
        continue

    grid_file = os.path.join(output_dir, grid_file)

    with open(grid_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for i, puzzle in enumerate(data):
        answer = data[i]["answer"]
        answer = re.sub(r"`", '', answer)
        answer = answer.replace('\n\n', '\n')
        answer = re.sub(r'<\/[^>]+>', '', answer)
        data[i]["answer"] = answer

    with open(grid_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
