import os
import json

MODEL_NAME = 'gpt-4o-mini'

base_dir = 'data/base'
puzzle_dir = 'data/train'
test_dir = 'data/test'
output_dir = f'data/reasonings/{MODEL_NAME}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(puzzle_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

clear_existing = False

if clear_existing:
    for filename in os.listdir(puzzle_dir):
        os.remove(os.path.join(puzzle_dir, filename))
    for filename in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, filename))

for filename in os.listdir(base_dir):
    filepath = os.path.join(base_dir, filename)

    if not filename.endswith(".json"):
        continue

    split = 0
    grid_size = filename[:3]
    if grid_size[0] == '3':
        split = 100
    else:
        match grid_size:
            case '441':
                split = 100
            case '442':
                split = 100
            case '443':
                split = 100
            case '451':
                split = 50
            case '452':
                split = 40
            case '453':
                split = 20
            case '461':
                split = 10
            case '462':
                split = 1
            case _:
                raise ValueError(f"Unknown grid size: {grid_size}")

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if isinstance(data, list):
            with open(os.path.join(test_dir, filename), 'w', encoding='utf-8') as puzzle_file:
                json.dump(data[-split:], puzzle_file,
                          ensure_ascii=False, indent=4)

            with open(os.path.join(puzzle_dir, filename), 'w', encoding='utf-8') as test_file:
                json.dump(data[:-split], test_file,
                          ensure_ascii=False, indent=4)

        else:
            raise ValueError(
                f"{filename} does not contain a list at the root.")

    except Exception as e:
        raise ValueError(f"Error processing {filename}: {e}")
