Here is a detailed `README.md` document based on the APIGen framework described in the sources, tailored to your specific architectural requirements (using decoupled vLLM servers and parallel processing). You can pass this directly to your AI agent to begin building the system.

***

# Custom APIGen: Automated Pipeline for Verifiable Function-Calling Datasets

## Quickstart

This repo contains a ready-to-run implementation for the **user–car interaction** domain.

```bash
uv sync --extra dev          # install deps
uv run pytest                # full pipeline verified offline (no GPU/network)

cp .env.example .env         # then point LLM_BASE_URL/* at your vLLM server
uv run python -m apigen --num 20 --out data/output/verified.jsonl
```

**Layout:** pipeline code in [apigen/](apigen/) (config, llm_client, samplers, prompts,
generator, [verification/](apigen/verification/) for the 3 stages, orchestrator). Temp domain
data in [data/](data/) — [car_apis.json](data/apis/car_apis.json) +
[car_functions.py](apigen/car_functions.py) (the executable mock vehicle) +
[seed_qa.json](data/seed/seed_qa.json). Replace the `data/` files and `car_functions.py`
with your real car APIs; nothing else needs to change. Add new API *types* (REST, etc.) by
implementing [backends/base.py](apigen/backends/base.py).

---

## Overview
This project aims to recreate the **APIGen** framework, an automated data generation pipeline designed to synthesize verifiable, high-quality datasets for function-calling applications. The goal is to generate synthetic query-answer pairs tailored to our specific domain, where the agent correctly translates natural language into structured API calls. 

Crucially, the system ensures data reliability by passing every generated data point through a strict **multi-stage verification process**: format checking, actual function execution, and semantic verification. 

To keep the architecture simple yet highly efficient, the pipeline will heavily utilize parallel asynchronous processing. All Large Language Models (LLMs) used for generation and semantic verification will be completely decoupled from the pipeline logic and served externally via **vLLM**.

---

## System Architecture

To achieve a simple, efficient, and parallelized system, the architecture is decoupled into two main environments:

1.  **The Generation & Verification Pipeline (Local Worker):** A lightweight, asynchronous Python application that handles data sampling, orchestrates validation, executes APIs, and coordinates HTTP calls. Parallelism will be achieved using asynchronous workers (e.g., `asyncio`) to process multiple API and LLM requests concurrently.
2.  **The LLM Serving Layer (External vLLM Servers):** All heavy lifting for text generation and semantic checking will be routed to an external vLLM server. *Note: The specific use of vLLM, `base_url`, and `api_token` configuration is an implementation detail added to fulfill your prompt's architectural request, though it aligns perfectly with APIGen's flexible LLM design.*

### Configuration 
The AI agent should build a centralized configuration system allowing easy swapping of the LLMs:
*   `LLM_BASE_URL`: The endpoint for the vLLM server.
*   `LLM_API_TOKEN`: The authentication token.
*   `GENERATOR_MODEL_NAME`: The model used to generate queries and function calls.
*   `SEMANTIC_CHECKER_MODEL_NAME`: The model used to verify if the execution results match the user's intent.

---

## Core Modules

The AI agent must implement the following distinct modules to replicate the APIGen workflow:

### 1. Data Sampling Module
This module ensures data diversity—a critical factor for robust function-calling datasets. It contains three sub-components:
*   **API Sampler:** Extracts function descriptions from our domain's API library and standardizes them into a uniform JSON format.
*   **Seed QA Data Sampler:** Samples seed examples (queries, function descriptions, answers) to serve as few-shot references for the generator.
*   **Prompt Sampler:** Selects diverse prompt templates to simulate different query styles, including **Simple**, **Multiple** (choosing one from many APIs), **Parallel** (multiple concurrent function calls in one response), and **Parallel Multiple**. 

### 2. Query-Answer Generator
This module interacts with the external vLLM server to generate data. 
*   It formats the sampled APIs, seed data, and prompts, then calls the `GENERATOR_MODEL_NAME`.
*   **Efficiency Trick:** The agent should implement a "batching" technique in the prompt, asking the LLM to output multiple query-answer pairs in a single inference to reduce token usage and cost.
*   The output must strictly be a standardized JSON format containing a `query` field and an `answers` field (which holds the function names and arguments).

### 3. Multi-Stage Verification Pipeline
Quality is paramount; previous research shows that small amounts of highly verified data substantially enhance model performance. The pipeline must pass generated data through three sequential checkers:

*   **Module 3A: Format Checker (Stage 1)**
    *   *Purpose:* Performs fast sanity checks without hitting the LLM or API.
    *   *Tasks:* Verifies that the LLM output can be parsed as valid JSON. It must check that the function names and arguments provided by the LLM actually exist in the provided API library.
    *   *Action:* Discard any outputs that hallucinate non-existent functions or lack required parameters.

*   **Module 3B: Execution Checker (Stage 2)**
    *   *Purpose:* Verifies that the generated function calls are operational.
    *   *Tasks:* Safely executes the well-formatted function calls against the actual domain backend (e.g., REST endpoints or local Python functions).
    *   *Action:* Captures execution results. Filters out data points that result in timeout, invalid parameters, or runtime errors, capturing fine-grained error messages. 

*   **Module 3C: Semantic Checker (Stage 3)**
    *   *Purpose:* Evaluates alignment between the query's objective, the function calls, and the actual execution results.
    *   *Tasks:* Formats the successful execution results, available functions, and user query, and sends them to the external vLLM server using the `SEMANTIC_CHECKER_MODEL_NAME`. 
    *   *Action:* The LLM evaluates if the results accurately reflect the user's intentions. Data points that execute successfully but produce meaningless results due to infeasible queries must be filtered out. 

### 4. Data Feedback Loop
Data points that successfully pass all three verification stages are considered high-quality. The system should automatically add these verified outputs back into the **Seed QA Datasets** to continuously enhance the diversity of future generations.

---

## Implementation Instructions for AI Agent

1.  **Tech Stack:** Use Python with `asyncio` for the main orchestrator, `aiohttp` for non-blocking HTTP requests to the vLLM server and REST APIs, and `Pydantic` for strict JSON schema enforcement in the Format Checker.
2.  **Concurrency:** Implement a producer-consumer queue system. 
    *   *Producers:* Asynchronously generate batches of function-call data via vLLM.
    *   *Consumers:* Concurrently run the generated items through Stage 1 (Format), Stage 2 (Execution), and Stage 3 (Semantic). 
3.  **LLM Abstraction:** Create a single `LLMClient` class initialized with `base_url` and `api_token` that handles all communications with the vLLM servers. The Generator and Semantic Checker modules should just pass their specific prompts and model names to this client.
4.  **Extensibility:** Design the Execution Checker (Stage 2) with a plugin architecture so that different types of domain APIs (e.g., Python scripts, REST APIs, GraphQL) can easily be added as execution backends.