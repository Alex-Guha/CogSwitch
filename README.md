# CogSwitch
Cognitive Switching in LLMs, by Alex Guha, Peter Mousses, Ryan Skabelund, and Yonatan Rosenbloom

## Problem Statement
Inspired by how humans switch between thinking, talking, reading, and remembering, this project introduces Cognitive Switching in LLMs, where special tokens guide the generation mode to improve attention and output. A causal model is finetuned with these tokens, which act as delimiters to indicate the desired type of generation, enabling more effective guidance during inference.

To investigate this approach, we use GPT-4o-mini to convert the GridPuzzle dataset [[1]](https://arxiv.org/pdf/2407.14790) into a fine-tuning dataset that incorporates these generation-guiding hidden tokens. We identify three key areas relevant to solving reasoning problems:
1) identifying relevant information
2) reasoning over the information
3) explaining the derived conclusions.

Accordingly, we introduce three special tokens to facilitate generation in this paradigm:
- `<recall>` for extracting immediately relevant context
- `<think>` for performing reasoning and providing new insights
- `<generate>` for summarizing findings and explaining conclusions. These are converted into discrete tokens within the tokenizer.

To evaluate this approach, we compare Llama-3-8B finetuned on the converted dataset against no finetuning, and evaluate whether the inclusion of these tokens improves the model's ability to generate outputs aligned with specific requirements, thereby enhancing efficiency and control in text generation tasks.

This work builds upon recent research on guiding tokens, including Disentangling Memory And Reasoning Ability In Large Language Models [[2]](https://arxiv.org/pdf/2411.13504) and Thoughts of Words Improve Reasoning In Large Language Models [[3]](https://arxiv.org/pdf/2410.16235). However, our method has a few notable differences from these. Compared to [[2]](https://arxiv.org/pdf/2411.13504), we use <recall> to extract information from the context rather than from the model’s parameterized memory. We also introduce a third token for normal generation as well as summarization. Unlike [[3]](https://arxiv.org/pdf/2410.16235), our approach allows the model to decide where to use the special tokens, and we specifically focus on complex reasoning tasks.
