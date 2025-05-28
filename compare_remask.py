# llada/compare_remask.py

import os
import torch
from transformers import AutoTokenizer, AutoModel
import evaluate
from generate import generate

def prepare_model(model_dir, device='cuda'):
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    return model, tokenizer

def prepare_input(tokenizer, prompt_text, device='cuda'):
    chat = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(prompt)['input_ids']
    return torch.tensor(ids, device=device).unsqueeze(0)

def evaluate_metrics(references, predictions):
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    rouge.add_batch(predictions=predictions, references=references)
    meteor.add_batch(predictions=predictions, references=references)

    rouge_result = rouge.compute()
    meteor_result = meteor.compute()

    return {
        "rougeL": rouge_result["rougeL"].mid.fmeasure,
        "meteor": meteor_result["meteor"]
    }

def main():
    model_dir = '/desay2PB/ct/dev/LLaDA_visualization/LLaDA-8B-Instruct'
    device = 'cuda'
    model, tokenizer = prepare_model(model_dir, device)

    prompts = [
        {
            "id": "math1",
            "prompt": "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
            "reference": "She can run 72 kilometers in total."
        },
        {
            "id": "bio1",
            "prompt": "What is the function of mitochondria in a cell?",
            "reference": "Mitochondria generate energy for the cell."
        },
        {
            "id": "logic1",
            "prompt": "If all humans are mortal and Socrates is a human, is Socrates mortal?",
            "reference": "Yes, Socrates is mortal."
        }
    ]

    strategies = ['low_confidence', 'random']
    os.makedirs('results', exist_ok=True)

    results_file = open("results/metrics_summary.txt", "w", encoding="utf-8")
    results_file.write("prompt_id\tstrategy\trougeL\tmeteor\n")

    for prompt_item in prompts:
        input_ids = prepare_input(tokenizer, prompt_item["prompt"], device)
        for remask in strategies:
            print(f"[{prompt_item['id']}] → Running strategy = {remask}")
            out = generate(
                model=model,
                prompt=input_ids,
                steps=64,
                gen_length=128,
                block_length=32,
                temperature=0.9,
                cfg_scale=0.0,
                remasking=remask
            )
            text = tokenizer.batch_decode(
                out[:, input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]

            print(f"  → Output = {text}")

            # 保存输出
            out_path = f"results/{prompt_item['id']}_{remask}.txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # 计算评估指标
            metrics = evaluate_metrics([prompt_item["reference"]], [text])
            print(f"  → ROUGE-L = {metrics['rougeL']:.4f}, METEOR = {'N/A' if metrics['meteor'] == -1 else f'{metrics['meteor']:.4f}'}")

            # 写入结果
            results_file.write(f"{prompt_item['id']}\t{remask}\t{metrics['rougeL']:.4f}\t{metrics['meteor']:.4f}\n")

    results_file.close()
    print("\nAll done. Metrics saved to results/metrics_summary.txt.")

if __name__ == '__main__':
    main()
