from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
model_name = "Qwen/Qwen2.5-7B-Instruct"

# 配置4位量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda",  # 强制使用GPU
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 可以尝试批量处理多个提示
prompts = [
    "Give me a short introduction to large language model.",
    "Explain the concept of machine learning in simple terms."
]

messages_list = [
    [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ] for prompt in prompts
]

texts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    ) for messages in messages_list
]

model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=False,  # 关闭采样，提高速度
    num_return_sequences=1
)

# 处理生成结果
responses = []
for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
    generated_ids_slice = output_ids[len(input_ids):]
    responses.append(tokenizer.decode(generated_ids_slice, skip_special_tokens=True))

print(responses)