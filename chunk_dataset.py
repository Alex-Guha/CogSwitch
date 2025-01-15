import os
import json
import shutil

MODEL_NAME = 'gpt-4o-mini'

base_dir = 'data/base'
puzzle_dir = 'data/train'
test_dir = 'data/test'
output_dir = f'data/reasonings/{MODEL_NAME}'

puzzles_per_min = 30

ids = []
for grid_type in sorted(os.listdir(os.path.join(output_dir, 'chunk_old'))):
    for chunk in sorted(os.listdir(os.path.join(output_dir, 'chunk_old', grid_type))):
        chunk_filepath = os.path.join(
            output_dir, 'chunk_old', grid_type, os.path.basename(chunk))
        if os.stat(chunk_filepath).st_size != 0:
            with open(chunk_filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)

            ids.extend([puzzle['id'] for puzzle in data])

for filename in sorted(os.listdir(puzzle_dir)):
    filepath = os.path.join(puzzle_dir, filename)

    if not filename.endswith(".json"):
        continue

    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for i, puzzle in enumerate(data):
        if puzzle['id'] in ids:
            data.pop(i)

    chunked_data = []
    if len(data) < puzzles_per_min:
        chunked_data = data
    else:
        chunk_size = len(data) // puzzles_per_min

        chunked_data = [data[i:i + chunk_size]
                        for i in range(0, len(data), chunk_size)]

    shutil.rmtree(
        f'{puzzle_dir}/chunks/{filename[:3]}', ignore_errors=True)
    os.makedirs(f'{puzzle_dir}/chunks/{filename[:3]}', exist_ok=True)
    for i, chunk in enumerate(chunked_data):
        chunk_path = f'{puzzle_dir}/chunks/{filename[:3]}/{i}.json'
        with open(chunk_path, 'w', encoding='utf-8') as file:
            json.dump(chunk, file, ensure_ascii=False, indent=4)
