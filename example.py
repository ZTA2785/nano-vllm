import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 确定模型路径：
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # 加载 model 和 tokenizer：
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1) # LLM 就是 LLMEngine 的一层名字上的包装
    
    # 确定采样参数：
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # 准备 prompts：
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    # 推理：
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
