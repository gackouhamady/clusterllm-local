# Ollama and Local Model Management

This project replaces external API calls with local inference through Ollama to ensure privacy and eliminate costs.

## Supported Local Models

The codebase is compatible with various open-weight families explored during research:

* 
**Llama 3/3.1/3.3**: Excellent for complex reasoning and pairwise judgments.


* 
**Qwen 2.5**: Highly efficient for triplet labeling.


* 
**DeepSeek-R1**: Strong reasoning, though slower on L4 hardware.



## Strict Parsing and Reliability

To prevent errors in noisy local environments, we enforce a strict parsing layer:

* 
**Triplets**: Must return exactly "Choice 1" or "Choice 2".


* 
**Pairs**: Must return exactly "Yes" or "No".


* 
**Logging**: All invalid or ambiguous responses are explicitly logged for auditability.



## Performance Engineering

To make local LLM supervision tractable, the system implements aggressive request-level parallelism, supporting up to 16 concurrent requests to the local Ollama server.

---