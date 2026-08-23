# RAG-Hallucination-Analyzer
 An interactive RAG-powered Q&amp;A application built with Gradio and LangChain, enabling users to query custom JSON datasets using semantic search

## Description

### Llama 3 Hallucination & Parameter Optimizer
A toolkit to analyze and minimize AI hallucinations in Retrieval-Augmented Generation (RAG) pipelines using Llama 3 and Sentence Transformers.

### Overview
This project evaluates how hyperparameters (temperature, context length) and prompt structures impact factual accuracy in Llama 3 outputs. It utilizes a CrossEncoder model to score generated answers against ground truth, identifying optimal configurations to reduce hallucinations.

## Dataset

https://github.com/RUCAIBox/HaluEval/blob/main/data/qa_data.json

## Getting Started

## Installation Steps

### The required Python libraries

* pip install gradio==4.44.0
* pip install --upgrade gradio
* pip install ibm-watsonx-ai==1.1.2
* pip install langchain==0.2.11
* pip install langchain-community==0.2.10
* pip install langchain-ibm==0.1.11
* pip install chromadb==0.4.24
* pip install pypdf==4.3.1
* pip install pydantic==2.9.1
* pip install huggingface_hub==0.23.0
* pip install gradio
* pip install huggingface_hub
* pip install --upgrade chromadb
* pip install openai
* pip install sentence-transformers
* pip install jq
* pip install "transformers<4.49.0"
* pip install datasets
* pip install "ragas==0.2.2"
* pip install nltk
* pip install deepeval litellm Ollama
* pip install --force-reinstall numpy==1.26.4

### LLM installation
!sudo apt-get update && sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh

### Executing program

* Execute Hallucination_Project.ipynb after installing the required Python library and starting Ollama.

## Result

### Overview and Execution Performance
* Speed: Execution time scales linearly with context length. Halving the context length cuts execution time in half.
* Model Stability: Llama 3 experienced progressive slowdowns during continuous query execution, requiring a system restart every 10 queries to maintain stable performance.
* Token Lengths: The maximum input token count in the JSON "knowledge" field is 226 with the prompt included, and 167 without the prompt.
* Context Length and Contradiction Trends:
** Large Context (>256 tokens): Disabling the prompt (PROMPT OFF) results in more contradictions compared to having the prompt active (PROMPT ON), except at the 2048 token boundary.
** Small Context (<=256 tokens): Contradictions increase as context length decreases.
** Short Context Anomaly (128 and 256 tokens): PROMPT ON yields more contradictions than PROMPT OFF at 128 and 256 token lengths. This occurs because the added prompt pushes the total token count past the tight context limit, causing truncation and data loss. 
* Prompt Impact on Classification
** Entailment and Neutral Shifts: Enabling the prompt (PROMPT ON) consistently drives a significant rise in entailment and a drop in neutral classifications across standard context lengths.
Temperature Profile: Temperature-based contradiction scaling was omitted from this evaluation because Llama 3 demonstrated high baseline stability and robustness.

### Deep Dive
#### Discussion: Context Length Anomalies
* Overview of Findings
** Natural Language Inference (NLI) contradiction trends deviate unexpectedly at the \(N=2048\) context boundary, breaking the expected linear or predictable scaling pattern where PROMPT OFF consistently exhibits higher contradiction rates than PROMPT ON.
Technical Analysis of the 2048 Boundary
** KV Cache Allocation Limits: Legacy inference runtimes and APIs enforce a hardcoded 2048-token ceiling. Testing precisely at this edge triggers silent memory reallocation or edge-case overflows in the Key-Value cache. [1]
** Prompt Disruption: System prompt overhead consumes a disproportionate share of the 2048-token window. Minor overflows force abrupt truncation, inadvertently stripping PROMPT ON instructions and corrupting attention masks.
** Positional Encoding Shifts: Approaching power-of-two boundaries without optimal chunk configurations disrupts Rotary Position Embedding (RoPE) scaling and native attention patterns, causing transient logical instability.
* Comparative Stability at 1024 and 4096
** 1024 Tokens: Operates safely below truncation thresholds, ensuring uniform structural handling and clean prompt processing.
** 4096 Tokens: Leverages fully allocated KV blocks and a well-tested legacy tier standard, eliminating positional-encoding anomalies observed at tighter boundaries.

  
### Future Scope & Improvements
This project identifies key operational constraints in Llama 3 and proposes targeted areas for future research and engineering enhancements.

* Performance & Reliability
** Stateless Persistence: Eliminate execution overhead from recurring context degradation by implementing state-management or cache-reset strategies that avoid full model restarts.
** Dynamic Context Scaling: Investigate behavioral changes at specific token thresholds (such as 2048 context length) where prompt conditioning introduces contradictions.
** Comparative Analysis
** Cross-Model Benchmarking: Evaluate alternative open-source LLMs to determine if similar context degradation and prompt-sensitivity patterns exist outside the Llama architecture.
** Temperature Robustness Mapping: Isolate the mechanisms behind Llama 3's high resilience to temperature adjustments to find models matching this stability without performance degradation or high restart latency.



## Authors

Contributors names and contact info

Baisakhi Mitra (baisakhi7.mitra7@gmail.com)


## Version History

* 0.1
    * Initial Release [In progress]

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments
** will be added
