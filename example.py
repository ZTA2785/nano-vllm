import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 确定模型路径：
    path = os.path.expanduser("/home/tianaozhang/data2/pretrained/Qwen3-0.6B/")
    
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
    
    # Huggingface 案例：
    # tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    # chat = [
    # {"role": "user", "content": "Hello, how are you?"},
    # {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    # {"role": "user", "content": "I'd like to show off how chat templating works!"},
    # ]
    # print(tokenizer.apply_chat_template(chat, tokenize=False))
    
    # 参数 add_generation_prompt 控制是否在输入最后加上提示 assistant 开始的 token
    # print(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
    # 输出：
    # <|user|>
    # Hello, how are you?</s>
    # <|assistant|>
    # I'm doing great. How can I help you today?</s>
    # <|user|>
    # I'd like to show off how chat templating works!</s>
    # <|assistant|> 这里就是多出了最后一行 <|assistant|>，提示模型应该开始补全 assistant 的部分了
        
    # 推理：
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
