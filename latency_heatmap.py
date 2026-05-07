import asyncio
import os
from llmeter.endpoints import OpenAICompletionEndpoint
from llmeter.experiments import LatencyHeatmap
from llmeter.callbacks import CostModel
from llmeter.callbacks.cost import dimensions
from dotenv import load_dotenv

load_dotenv()

endpoint = OpenAICompletionEndpoint(
    model_id="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Cost model for GPT-4o
cost_model = CostModel(
    request_dims=[
        dimensions.InputTokens(price_per_million=2.50),
        dimensions.OutputTokens(price_per_million=10.00),
    ]
)

async def main():
    heatmap = LatencyHeatmap(
        endpoint=endpoint,
        source_file=None,   
        clients=4,
        output_path="./results/heatmap",
        input_lengths=[50, 100, 200, 500],
        output_lengths=[50, 100, 200, 500],
        requests_per_combination=3,
        create_payload_fn=lambda input_len, output_len: 
            endpoint.create_payload(
                "Explain why latency percentiles matter in production AI systems.",
                max_tokens=output_len
            ),
        callbacks=[cost_model]
    )
    
    result = await heatmap.run()
    result.plot_results()
    
    print("Latency Heatmap completed!")
    print(f"Results saved to: {result.output_path}")

if __name__ == "__main__":
    asyncio.run(main())
