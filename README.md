# LLMeter Demo

## Installation

```
uv pip install llmeter
```

## Prerequisites

- Python 3.10 or higher
- An API key for the LLM provider you want to test (e.g., OpenAI, Anthropic, etc.)

## Key Concepts

- **Endpoint**: The LLM API endpoint you want to test (e.g., OpenAI's `/v1/chat/completions`)
- **Payload**: The request body sent to the LLM API
- **Dimensions**: The metrics you want to track (e.g., input tokens, output tokens, latency, etc.)
- **LoadTest**: The class that runs the load test and collects metrics
- **Experiment**: The class that runs multiple load tests and aggregates the results

## Example Usage

```python
python main.py
```

```python
python realtime_dashboard.py
```



