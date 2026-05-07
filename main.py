import asyncio
import os
from llmeter.endpoints import OpenAICompletionStreamEndpoint
from llmeter.callbacks import CostModel
from llmeter.callbacks.cost import dimensions
from dotenv import load_dotenv

load_dotenv()


endpoint = OpenAICompletionStreamEndpoint(
    model_id="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

payload = endpoint.create_payload(
    "Who is the author of QAInsights blog?",
    max_tokens=200,
)

cost_model = CostModel(
    request_dims=[
        dimensions.InputTokens(price_per_million=2.50),
        dimensions.OutputTokens(price_per_million=10.00),
    ]
)

async def main():
    from llmeter.experiments import LoadTest
    # This experiment creates a series of Runs with different levels of concurrency, defined by
    # ``sequence_of_clients``, and runs them one after the other.
    load_test = LoadTest(
        endpoint=endpoint,
        payload=payload,
        sequence_of_clients=[1, 3], 
        output_path="./results",
        run_duration=10, # seconds
        callbacks=[cost_model]
    )
    load_test_results = await load_test.run()
    load_test_results.plot_results()
    
    for clients, result in load_test_results.results.items():
        stats = result.stats
        print(f"Clients: {clients} | Total Cost: ${stats.get('cost_total', 0):.6f} | Cost/Request: ${stats.get('cost_per_request-average', 0):.6f}")

if __name__ == "__main__":
    asyncio.run(main())