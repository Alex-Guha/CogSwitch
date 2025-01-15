import os
import json
import asyncio
import shutil
import time
import random
from openai import AsyncOpenAI, RateLimitError

client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

MODEL_NAME = 'gpt-4o-mini'

base_dir = 'data/base'
puzzle_dir = 'data/train'
test_dir = 'data/test'
output_dir = f'data/reasonings/{MODEL_NAME}'


def get_prompt(problem, generated_meta_token_reasoning, next_line_to_generate_for, answer):
    system_prompt = f"""Task Overview:
You are annotating a given chain of reasoning with meta-tokens. You will be provided with a description of the meta-tokens and an explanation on how they should be used. Given the previous steps as well as the immediate next step, you will generate and use the meta-tokens to arrive at that next step from the previous ones. You will also be provided the answer to ensure your generation is accurate, but you should not reference it at all. An annotation starts with the meta-token, followed by your usage of the token. Do not conclude meta-tokens with the same one and a slash, only by switching to a different meta-token.

Meta-tokens:
<recall> - This token is used to look over the context and pull out the information that will be immediately relevant to figuring out the next part.
<think> - This token is used to do internal reasoning, separately from normal discussion.
<generate> - The standard mode, used to switch back to normal response. Anything after this token is displayed to the user.

Usage Example:
```
problem statement
...
[relevant info]
...
previous line of reasoning
<recall>[relevant info] to be used in the next conclusion
<think>internal reasoning for the next conclusion
<generate>line of reasoning
```

For instance, if the text was:
```
Clues:
x. Clue

Chain of reasoning:
previous reasoning
next line of reasoning using clue x
```
Then your output would only be:
```
<recall>x. Clue
<think>How you arrive at the next line of reasoning based on the past and the recalled info
<generate>next line of reasoning
```
"""
    user_prompt = f"""
Problem:
```
{problem}

{answer}
```

Chain of reasoning:
```
{generated_meta_token_reasoning}
[Your next generation will go here]

Final Answer:
{answer}
```

Instructions:
Use the meta-tokens to arrive at the next line of reasoning and ONLY the next line of reasoning. Do not generate further lines of reasoning. Do not abbreviate. There may be steps that use multiple annotations or do not need annotations at all, that is up to you. Ensure each meta-token stays within its own domain; for instance, do not have reasoning in recall annotations, and vice versa.

Next Line of Reasoning: {next_line_to_generate_for}
"""
    return system_prompt, user_prompt


# https://platform.openai.com/docs/guides/rate-limits#retrying-with-exponential-backoff
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (RateLimitError,),
):
    """Retry a function with exponential backoff."""

    async def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return await func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
async def get_model_response(problem, generated_meta_token_reasoning, next_line_to_generate_for, answer):
    system_prompt, user_prompt = get_prompt(
        problem, generated_meta_token_reasoning, next_line_to_generate_for, answer)
    completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    return completion.choices[0].message.content


async def generate_reasoning_chain(puzzle):
    answer = puzzle['answer'].split('\nFinal Answer:\n')
    hint_list = answer[0][len('\n\nStep-by-step solution:\n')-1:].split('\n')
    solution = answer[1]

    reasoning_chain = ''
    for hint in hint_list:
        if not hint:
            continue
        params = {
            'problem': puzzle['question'],
            'generated_meta_token_reasoning': reasoning_chain,
            'next_line_to_generate_for': hint,
            'answer': solution
        }
        reasoning_chain += await get_model_response(**params)
    return reasoning_chain + '\n\nFinal Answer:\n' + solution


async def convert_file(filepath, grid_type, n=0):
    """
    Converts a JSON file by generating reasoning chains for each puzzle and appending the results to an output file.
    The output file location is determined by output_dir and has the same name as the input file.

    Args:
            filepath (str): The path to the input JSON file containing puzzles.
            n (int): The number of puzzles to convert. If n is 0, all puzzles are converted.

    The function performs the following steps:
    1. Reads the input JSON file.
    2. Checks if the output file already exists and is not empty.
    3. If the output file is not empty, it loads the existing data and skips puzzles that have already been solved.
    4. For each unsolved puzzle, generates a reasoning chain and appends the result to the output file.
    """
    if not filepath.endswith(".json"):
        return

    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    os.makedirs(os.path.join(output_dir, 'chunk', grid_type), exist_ok=True)

    output_filepath = os.path.join(
        output_dir, 'chunk', grid_type, os.path.basename(filepath))
    open(output_filepath, 'a').close()
    print(f"[+] Beginning generation for {output_filepath}")

    if os.stat(output_filepath).st_size != 0:
        with open(output_filepath, 'r', encoding='utf-8') as output_file:
            output_data = json.load(output_file)
            solved_ids = [solved_puzzle['id'] for solved_puzzle in output_data]
            for i, puzzle in enumerate(data):
                if i == len(data) - 1:
                    print(
                        f"[ ] Generation for {output_filepath} already complete, exiting.")
                    return
                if puzzle['id'] not in solved_ids:
                    data = data[i:]
                    break

    for puzzle in data[:n] if n > 0 else data:
        print(f"[ ] Converting puzzle {puzzle['id']} in {output_filepath}")
        if os.stat(output_filepath).st_size != 0:
            with open(output_filepath, 'r', encoding='utf-8') as output_file:
                output_data = json.load(output_file)
        else:
            output_data = []
        reasoning_chain = await generate_reasoning_chain(puzzle)
        converted_data = {
            'id': puzzle['id'],
            'question': puzzle['question'],
            'answer': reasoning_chain
        }
        output_data.append(converted_data)
        with open(output_filepath, 'w', encoding='utf-8') as output_file:
            json.dump(output_data, output_file, ensure_ascii=False, indent=4)
        print(f"[+] Converted puzzle {puzzle['id']} in {output_filepath}")
    print(f"[+] Finished generation for {output_filepath}")


async def main(grid_type):
    chunks = []
    for chunk in sorted(os.listdir(os.path.join(puzzle_dir, 'chunks', grid_type))):
        chunk_path = os.path.join(puzzle_dir, 'chunks', grid_type, chunk)
        chunks.append((chunk_path, grid_type))

    await asyncio.gather(*(convert_file(arg) for arg in chunks))

if __name__ == "__main__":
    for grid_type in sorted(os.listdir(puzzle_dir + '/chunks')):
        print(f"[+] Processing {grid_type} puzzles...")
        asyncio.run(main(grid_type))
