# -*- coding: utf-8 -*-

import argparse
import json
from string import Template

from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen2.5-Coder-32B-Instruct')
parser.add_argument('--model_path', type=str, default='./Qwen2.5-Coder-32B-Instruct')
parser.add_argument('--input_data_path', default='./CodeCriticBench.jsonl', type=str)
parser.add_argument('--output_data_path', default='./output', type=str)
args = parser.parse_args()

QWEN_MODEL = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.model_path, torch_dtype='auto', device_map='auto')
QWEN_TOKENIZER = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_path)

def get_response(str_user_input):
    messages = [
        {'role': 'system', 'content': 'you are a helpful assistant'},
        {'role': 'user', 'content': str_user_input}
    ]
    text = QWEN_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = QWEN_TOKENIZER([text], return_tensors='pt').to(QWEN_MODEL.device)
    generated_ids = QWEN_MODEL.generate(**model_inputs, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    answer = QWEN_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return answer

def process_line(args):
    model_name, line, index = args
    print(index, end=' ', flush=True)
    curr = json.loads(line.strip())
    prompt = '...'  # replace full prompt here
    answer = get_response(prompt)
    curr[f'response'] = answer
    return curr

if __name__ == '__main__':
    with open(args.input_data_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    input_data_name = args.input_data_path.split('/')[-1].split('.')[0]
    output_data_path = f'{args.output_data_path}/{args.model_name}_{input_data_name}.jsonl'


    for index, line in enumerate(lines):
        curr = process_line((args.model_name, line, index))
        with open(output_data_path, 'a', encoding='utf-8') as fp:
            fp.write(json.dumps(curr, ensure_ascii=False) + '\n')