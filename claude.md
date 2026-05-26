# CLAUDE.md

## Project Overview
This codebase implements a speculative decoding algorithm for fast LLM inference called "speculative speculative decoding" which performs speculation in parallel to verification.
The way it manages to parallelize these operations is as follows:
- The draft model predicts the most likely outcomes from verifying it's most recent speculation (how many tokens will be accepted, and what the "recovery token" aka "bonus token" will be).
- The draft model then speculates a sequence of tokens for each of these outcomes, in a sequence of "lookahead" forward passes of the draft model (here, lookahead is the length of each speculation), using a special attention mask that encodes this token tree with many braches off of a shared trunk (the previous speculation).
- In the case where one of these outcomes occurs, we can immediately return the corresponding speculation from the "speculation cache".
- In the case where an outcome we hadn't prepared for occurs, we fallback to a backup speculation option. The options are "jit", in which we do regular speculation for the whole batch "just-in-time", and "fast", in which we simply return a speculation consisting of all zeros for the batch elements that we had a "cache miss" for. We also implement a "force-jit" option, which always does JIT speculation, regardless of cache hit vs miss. This is to allow direct comparison with regular (synchronous) speculative decoding for debugging purposes.

As an example, imagine the draft model just sent tokens `[a b c d]` to be verified by the target models.
We can then consider different outcomes of the verification, such as "accept `a` and `b`, reject `c`, and sample token `c'` instead (as the "recovery token").
For a list of such outcomes, we can generate the next speculated sequence (e.g., `[d e f g]` following `[a b c']`).
For each possible number of accepted tokens (any integer between 0 and the speculation length, which we call the `lookahead`), we can select the top-$F_k$ alternative tokens to follow the accepted token sequence (each corresponding to a verification outcome).
Then, for all these possible verification outcomes, we can in parallel draft a speculated token sequence of length `lookahead`, using `lookahead` forward passes of the draft model with the appropriate attention mask to encode this token tree structure.

There are two ways to run this algorithm:
- Using the LLM engine defined in this SSD code base: `git/ssd/ssd/engine/llm_engine.py` -- can create an `LLMEngine` object and called `generate` on it with a prompt.
- Using our private fork of SGLang (at `/work/avner/git/tgl`), and specifying `ASYNC_STANDALONE`, `ASYNC_EAGLE3`, or `ASYNC_PHOENIX` as the `--speculative-algorithm`, and then sending requests to this server in the standard manner.

## Speculative decoding background
Speculative decoding is an algorithm for speeding up the decoding phase of LLM inference.
It works by leveraging a "draft model" to speculate the next $L$ tokens, and then using the "target model" (the one we care to generate a response from) to "verify" these speculated tokens, using a single forward pass of the target model with all $L$ tokens.
The algorithm decides which tokens to accept and reject in such a way that the accepted tokens are generated from the same output distribution as the target model.
After deciding which tokens to accept, this algorithm also samples an additional "bonus token" or "recovery token" which follows the sequence of accepted tokens.
While the simplest form of this algorithm uses a small draft model that is completely independent of the target model, there is a method called "Eagle" where a draft model is trained to take as input its token embeddings concatenated with activations from the target model on the current prefix.
This allows the Eagle draft model to take advantage of the powerful representations from the target model to better predict the next few tokens, and thereby on average have more tokens accepted, which leads to bigger speedups. As an example (using latex notation):
- Assume the current prefix is $[t_0, ..., t_5]$, where $t_5$ is the "recovery token" from the prior round.
- The target model has only processed tokens $[t_0, ..., t_4]$, producing activations $[h_0, ..., h_4]$.
- We pass as input to the draft model $[(t_0, h_0), (t_1, h_0), (t_2, h_1), (t_3, h_2), (t_4, h_3), (t_5, h_4)]$, where $(a,b)$ is the concatenation of tensors $a$ and $b$ along their feature dimension (note: here I am overloading notation and using $t_0$ to also refer to the token embeddings for $t_0$).
- The draft model then predicts $t_6$, and generates output hidden state $h_5'$, which is passed as input to the draft model in the next speculation step in place of $h_5$ ($[(t_6, h_5')]$), because we do not have access to the target activations $h_5$ for $t_5$ yet.
- This process is repeated until $lookahead$ tokens have been sampled from the draft model.

## Build & Run

1. Activate the appropriate uv python environment.
- For SSD: `source /work/avner/git/ssd/.venv/bin/activate`
- For TGL: `source /work/avner/git/tgl/tgl-ssd-env-e/.venv/bin/activate`.
2. Launching server: See the `/work/avner/git/ssd/tests/hf/test_ssd_vs_hf_reference.py` test for examples of how to launch server for both SSD and TGL.

## Git branches
- SSD repo (`work/avner/git/ssd`): Use the `avner/sglang-fa4-phnx` branch (unless otherwise instructed).
- TGL repo (`/work/avner/git/tgl`): Use the `avner/ssd-port` branch (unless otherwise instructed).

Note that you can make changes in both of these repositories.

## Testing
- Framework: pytest
- Always run the specific test you changed, not the full suite.
- Use `-x` flag to stop on first failure.
- Example launch command:
```
source /work/avner/git/tgl/tgl-ssd-env-e/.venv/bin/activate
cd /work/avner/git/ssd/tests/hf
pytest -s
```
This test compares the outputs (both final and intermediate) of a standalone version of the SSD algorithm (written using python Huggingface code), to those of the inference engine implementation of SSD (either SSD engine or TGL engine).
Note that within the `/work/avner/git/ssd/tests/hf/test_ssd_vs_hf_reference.py` file, the primary `test_ssd_vs_hf_reference` test has a list of parameters that we can specify which values to sweep over, including the engine, speculator type, and backup strategies to use.

## Repository structure
SSD Repo (at `/work/avner/git/ssd/`):
- `ssd/`: main python package
  - `engine/`: Core components of the inference engine.
    - `llm_engine.py`: Outer loop of the SSD inference engine.
    - `draft_runner.py`: Outer loop of the draft model process. This code is used in both the TGL and SSD inference engines.
    - `helpers/runner_helpers.py`: Dataclasses that handle communication between the draft and target processes for both TGL and SSD inference engines.
  - `layers/`: Implementation of the different components of the transformer architecture (including support for tensor parallelism).
  - `models/`: Implementation of the full draft (including Eagle3 and Phoenix) and target model classes, for both Llama and Qwen families of models.
  - `utils/`: A variety of helper methods/classes, including `verify.py` that contains the logic for deciding which of the drafted tokens get accepted.
- `tests/`: All the unit and integration tests for this repository.
  - `hf/`: This directory contains `test_ssd_vs_hf_reference.py`, the most important end-to-end test of the engines' performance.

TGL repo (at `/work/avner/git/tgl/`):
- `python/sglang/`: main Python package.
  - `srt/`: SRT (SGLang Runtime) server code.
  - `srt/managers/`: scheduler, token, detokenizer managers.
  - `srt/layers/`: model layers, attention backends, quantization.
  - `srt/model_executor/`: model loading, forward execution.
  - `srt/models/`: individual model architectures.
  - `src/speculative`: Speculative decoding classes.
  - `private/`: Private versions of many of the classes in the `srt` directory.
  - `private/speculative/`: Private implementations of speculative decoding classes, including `spec_worker.py` which contains the `SpecWorker` class that the `AsyncSpecWorker` class (which implements the SSD algorithm in TGL) derives from.
  - `private/speculative/async_spec/`: Contains `async_spec_worker.py`, the TGL implementation of SSD.
- `test/srt/`: runtime tests
- `docs/`: documentation

## What NOT To Do
- Don't `pip install` anything — use `uv pip install` / `uv add` / `uv sync`.
- Don't use `subprocess.Popen` without setting `preexec_fn=os.setsid` for process group management
- Don't assume a single-GPU setup — multi-GPU tensor parallelism is common.

## What YES To Do
- I would like to work *together* with you to solve whatever challenges come up, ideally with me writing the code myself (with your supervision).
- I want you to first and foremost consider yourself my teacher, always explaining each step as you go. When the code base has complicated sections, I'd like you to be patient with me and explain them step by step.