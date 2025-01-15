import os
import json

MODEL_NAME = 'gpt-4o-mini'

base_dir = 'data/base'
puzzle_dir = 'data/train'
test_dir = 'data/test'
output_dir = f'data/reasonings/{MODEL_NAME}'


for grid_type in os.listdir(os.path.join(puzzle_dir, 'chunks')):
    data = []
    for chunk_ in ['chunk', 'chunk_old']:
        chunk_dir = os.path.join(output_dir, chunk_, grid_type)
        if os.path.exists(chunk_dir):
            for chunk in os.listdir(chunk_dir):
                chunk_filepath = os.path.join(chunk_dir, chunk)
                if os.stat(chunk_filepath).st_size != 0:
                    with open(chunk_filepath, 'r', encoding='utf-8') as file:
                        data.extend(json.load(file))

    if data:
        with open(os.path.join(output_dir, grid_type + '.json'), 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
