import asyncio
import os
from llmeter.endpoints import OpenAICompletionEndpoint
from llmeter.experiments import CreatePromptCollection
from dotenv import load_dotenv

load_dotenv()

endpoint = OpenAICompletionEndpoint(
    model_id="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

async def main():
    prompts = [
        "Explain why latency percentiles matter in production AI systems.",
        "What is the difference between p50 and p99 latency?",
        "How do you calculate throughput in an AI system?",
        "Why is token usage important for cost optimization?",
        "Describe the relationship between concurrency and latency."
    ]
    
    collection = CreatePromptCollection(
        endpoint=endpoint,
        prompts=prompts,
        output_path="./results/prompt_collection",
        max_tokens=200
    )
    
    collection.save()
    
    print(f"Prompt collection created with {len(prompts)} prompts")
    print(f"Saved to: ./results/prompt_collection")

if __name__ == "__main__":
    asyncio.run(main())
