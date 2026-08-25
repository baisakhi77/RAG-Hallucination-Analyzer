# RAG-Hallucination-Analyzer

## Description

### Llama 3 Hallucination & Parameter Optimizer
A toolkit to analyze and minimize AI hallucinations in Retrieval-Augmented Generation (RAG) pipelines using Llama 3 and Sentence Transformers.

### Overview
This project evaluates how hyperparameters (temperature, context length) and prompt structures impact factual accuracy in Llama 3 outputs. It utilizes NLI (Natural Language Inference) cross-encoder model to score generated answers against ground truth, identifying optimal configurations to reduce hallucinations. It evaluates Llama3 performance based on three NLI labels:
  * Entailment: The hypothesis is definitely true if the premise is true.
  * Contradiction: The hypothesis is definitely false if the premise is true; the two statements cannot both be true at the same time.
  * Neutral: The hypothesis might be true or false based on the premise, but there is not enough information to prove or disprove it definitively.

## Dataset

We use the [HaluEval](https://github.com/RUCAIBox/HaluEval) benchmark for testing and evaluating model hallucinations. 

<Quote bind="1.1.9">HaluEval is a large collection of generated and human-annotated hallucinated samples for evaluating the performance of LLMs in recognizing hallucination, as introduced in [HaluEval](https://github.com/RUCAIBox/HaluEval).</Quote>


## Python packages used
* langchain
* langchain_community
* sentence-transformers
* langchain_core
* jq
* numpy
* chromadb
  
## Running the Notebook

Click the **Open in Colab** badge above to launch and execute this Jupyter notebook directly in your browser via [Google Colab](https://drive.google.com/file/d/100FNn7hbvksJYSy9wd5IiOj6lYCalN_P/view?usp=sharing). No local installation is required.

### Steps:
1. Click the **Open In Colab** button.
2. Once the notebook opens in Google Colab, you can run individual code cells by clicking the **Play button** or pressing `Shift + Enter`.
3. If your notebook requires a GPU (for machine learning or AI models), go to **Runtime** > **Change runtime type** in the top menu and select **T4 GPU** (or another available accelerator), then click **Save**.
4. Before executing the notebook, mention the input JSON file path in the 'input_file_path' variable and pass the desired output file location to 'append_to_json' function in the code.
   

## Result

### Overview and Execution Performance
* Speed: Execution time scales linearly with context length (llama3 'num_ctx' hyperparameter). Halving the context length cuts execution time in half.
* Model Stability: Llama 3 experienced progressive slowdowns during continuous query execution, requiring a system restart every 10 queries to maintain stable performance.
* Token Lengths: The maximum input token count in the JSON "knowledge" field is 226 with the prompt included, and 167 without the prompt.
  * Context Length and Contradiction Trends:
    * Large Context (>256 tokens): Disabling the prompt (PROMPT OFF) results in more contradictions compared to having the prompt active (PROMPT ON), except at the 2048 token boundary.
    * Small Context (<=256 tokens): Contradictions increase as context length decreases.
    * Short Context Anomaly (128 and 256 tokens): PROMPT ON yields more contradictions than PROMPT OFF at 128 and 256 token lengths. This occurs because the added prompt pushes the total token count past the tight context limit, causing truncation and data loss. 
* Prompt Impact on Classification
    * Entailment and Neutral Shifts: Enabling the prompt (PROMPT ON) consistently drives a significant rise in entailment and a drop in neutral classifications across standard context lengths.
* Temperature Profile: Temperature-based contradiction scaling was omitted from this evaluation because Llama 3 demonstrated high baseline stability and robustness.

### Deep Dive
#### Discussion: Context Length Anomalies
* Overview of Findings
    * Natural Language Inference (NLI) contradiction trends deviate unexpectedly at the \(N=2048\) context boundary, breaking the expected linear or predictable scaling pattern where PROMPT OFF consistently exhibits higher contradiction rates than PROMPT ON.
* Technical Analysis of the 2048 Boundary
    * KV Cache Allocation Limits: Legacy inference runtimes and APIs enforce a hardcoded 2048-token ceiling. Testing precisely at this edge triggers silent memory reallocation or edge-case overflows in the Key-Value cache. 
    * Prompt Disruption: System prompt overhead consumes a disproportionate share of the 2048-token window. Minor overflows force abrupt truncation, inadvertently stripping PROMPT ON instructions and corrupting attention masks.
    * Positional Encoding Shifts: Approaching power-of-two boundaries without optimal chunk configurations disrupts Rotary Position Embedding (RoPE) scaling and native attention patterns, causing transient logical instability.
* Comparative Stability at 1024 and 4096
    * 1024 Tokens: Operates safely below truncation thresholds, ensuring uniform structural handling and clean prompt processing.
    * 4096 Tokens: Leverages fully allocated KV blocks and a well-tested legacy tier standard, eliminating positional-encoding anomalies observed at tighter boundaries.

  
### Future Scope & Improvements
This project identifies key operational constraints in Llama 3 and proposes targeted areas for future research and engineering enhancements.

* Performance & Reliability
    * Stateless Persistence: Eliminate execution overhead from recurring context degradation by implementing state-management or cache-reset strategies that avoid full model restarts.
    * Dynamic Context Scaling: Investigate behavioral changes at specific token thresholds (such as 2048 context length) where prompt conditioning introduces contradictions.
* Comparative Analysis
    * Cross-Model Benchmarking: Evaluate alternative open-source LLMs to determine if similar context degradation and prompt-sensitivity patterns exist outside the Llama architecture.
    * Temperature Robustness Mapping: Isolate the mechanisms behind Llama 3's high resilience to temperature adjustments to find models matching this stability without performance degradation or high restart latency.



### Citation

@misc{HaluEval,
  author = {Junyi Li and Xiaoxue Cheng and Wayne Xin Zhao and Jian-Yun Nie and Ji-Rong Wen },
  title = {HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models},
  year = {2023},
  journal={arXiv preprint arXiv:2305.11747},
  url={https://arxiv.org/abs/2305.11747}
}
### Acknowledgments

* AI model inference powered locally by [Ollama](https://ollama.com/).
* This project uses the following open-source libraries:
  * [LangChain](https://github.com/langchain-ai/langchain) (`langchain`, `langchain_core`, `langchain_community`)
  * [sentence-transformer](https://github.com/huggingface/sentence-transformers).
  * [ChromaDB](https://github.com/chroma-core/chroma)
  * [jq](https://github.com/jqlang/jq)
  * [numpy](https://github.com/numpy/numpy)




## Contact

Contributors names and contact info

Baisakhi Mitra (baisakhi7.mitra7@gmail.com)

