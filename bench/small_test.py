import argparse
import os

from transformers import AutoTokenizer
from ssd import LLM, SamplingParams

if __name__ == '__main__':

    llama_1b_path = '/scratch/avner/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6'
    llama_70b_path = '/scratch/avner/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b'
    eagle_path = '/scratch/avner/huggingface/hub/models--lmsys--SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge/snapshots/63ebaa6585f96b89685adad8fdfa0da53be6a8fd'
    # eagle_path = '/scratch/avner/huggingface/hub/models--yuhuili--EAGLE3-LLaMA3.3-Instruct-70B'
    assert os.path.isdir(llama_1b_path)
    assert os.path.isdir(llama_70b_path)
    assert os.path.isdir(eagle_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=llama_1b_path)
    parser.add_argument("--draft", type=str, default=llama_1b_path)
    parser.add_argument("--eagle", action="store_true")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--jit-speculate", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()

    if args.eagle:
        args.draft = eagle_path
        args.model = llama_70b_path
        args.num_gpus = 5
        args.jit_speculate = True

    llm = LLM(
        model=args.model,
        draft=args.draft,
        use_eagle=args.eagle,
        speculate_k=args.k,
        speculate=True,
        draft_async=True,
        num_gpus=args.num_gpus,
        jit_speculate=args.jit_speculate,
        verbose=True,
    )
    sampling_params = [SamplingParams(temperature=0.0, max_new_tokens=64, ignore_eos=args.ignore_eos)]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokens = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital city of France?"}],
        add_generation_prompt=True,
    )
    token_str = tokenizer.decode(tokens)
    print(f"Generating response to prompt: {token_str}")

    outputs, _ = llm.generate([tokens], sampling_params)

    print(outputs[0]["text"])
