import numpy as np
import openvino as ov

from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig


from simple_tokenizer import SimpleTokenizer
from misc import sampling

model_id = 'Intel/neural-chat-7b-v3'

def main():
    model_name = model_id.split('/')[1]
    model_precision = ['FP16', 'INT8', 'INT4', 'INT4_stateless'][2]

    print(f'LLM model: {model_id}, {model_precision}')

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ov_model = OVModelForCausalLM.from_pretrained(
        model_id = f'{model_name}/{model_precision}',
        config=AutoConfig.from_pretrained(model_id)
    )

    prompt_text = '請用金庸的口吻, 也就是用武俠小說的敘事, 以第一人稱、小說家的觀點說一次, 說為什麼這是一件武林軼事, 並且以「這就是江湖」結尾。\n\n'
    input_tokens = tokenizer.encode(prompt_text, return_tensors='pt')
    response = ov_model.generate(input_tokens, max_new_tokens=300)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    print(response_text)

if __name__ == "__main__":
    main()