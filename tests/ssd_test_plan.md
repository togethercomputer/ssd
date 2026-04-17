# Test plan for SSD (for both SSD and TGL repos)

##  System overview.
- We have implemented an LLM inference algorithm called SSD (speculative speculative decoding, described in this paper: https://arxiv.org/pdf/2603.03251), in two repositories:
  - SSD (/work/avner/git/ssd): This is a self-contained implementation of the algorithm.
  - TGL (/work/avner/git/tgl): This is an integration of the SSD algorithm into a private branch of the open-source inference engine SGLang. For the draft process, as well as communication between the draft and target processes, it imports code from the SSD repo.
- The high-level design of the algorithm is as follows:
  - Instead of doing speculative decoding by alternating sequentially between the draft model speculating K tokens, and the target model verifying those tokens, this algorithm does speculation and verification asynchronously, on separate GPUs.
  - It does so by letting the draft model predict what it believes to be the most likely outcomes of the ongoing verification (e.g., accept k tokens, reject the k+1 token, and sample token t instead), and then speculating in advance in parallel for each of these outcomes, while the verification is still ongoing. If the actual verification outcome is one that it had prepared for, it can immediately send the speculation for that outcome, which it had precomputed.
  - It has two strategies for handling cases where the actual verification outcome is not in the set of outcomes the draft model had prepared for: (1) "JIT": Speculate "just in time" using the draft model (the target model will wait while the draft model is running, like in regular speculative decoding), (2) "Fast": Immediately return all zeros as the speculation.  (We additionally implement "force-jit", which ALWAYS runs the draft model synchronously, to aid with debugging and sanity checking).
- We would like to create a thorough testbed for this algorithm (for now, can ).

## Test plan design criteria
- The primary repos/branches we want to test are:
  - The `avner/sglang-fa4` branch of the SSD repo (/work/avner/git/ssd)
  - The `avner/ssd-port` branch of the TGL repo (/work/avner/git/tgl)

The following are properties the SSD async speculation system should have:
- `--force-jit` performance (acceptance rates, which tokens accepted, etc) should be identical to synchronous speculative decoding performance, in both SSD repo (self-contained async spec implementation) and TGL repo (for both Eagle and standalone speculators).
- SSD behavior for a given setting (acceptance rates, which tokens accepted, cache hits vs misses, inputs/outputs, etc) should always match TGL behavior for the same setting (eagle vs standalone, and force-jit vs jit vs fast backup strategies).
- The behavior of the system (inputs/outputs, accept vs reject decisions, cache hits vs misses) should match that of a naive inefficient implementation of the algorithm (e.g., using huggingface).
- All of the above should hold true for Llama 8B with TP=1, and Llama 70B with TP=4, with both Eagle and Standalone speculators.
- The SSD performance (including speed in tokens per second) at branch `avner/sglang-fa4` should be similar to or better than the `avner/main2` branch.
- The SSD speed in the SSD repo should be similar to the SSD speed in the TGL repo.
- These tests should be as simple and efficient as possible, testing individual components whenever possible, and doing end-to-end testing whenever necessary. Perhaps there should be a fast subset of tests we can run frequently, and a slower but more thorough set of tests.
- There should be a test that simply benchmarks the algorithm, and stores the speeds of each important component in a structured format that it uses for visualization (creating plots to visualize the key results, similar to /work/avner/git/ssd/bench/extract_metrics.py), and ideally fails when there has been a regression in performance.
- The results of these tests should ideally be stored in a sub-folder of the ssd repo, and perhaps uploaded automatically to git for visualization/review. Perhaps git actions are a useful tool here, perhaps to run these tests automatically on every commit?

## Other important details:
- Current benchmarking scripts for both the SSD and TGL repositories are at /work/avner/git/ssd/bench/bench.py and /work/avner/git/ssd/bench/run_sglang_bench.py.
- The python environments for the SSD and TGL repos are uv python environments at /work/avner/git/ssd/.venv and /work/avner/git/tgl/.venv.
- I have access to research-secure-29.cloud.together.ai and research-secure-30.cloud.together.ai for testing, and my username is 'avner'.