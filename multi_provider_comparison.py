import asyncio
import os
from llmeter.endpoints import OpenAICompletionEndpoint
from llmeter.experiments import LoadTest
from llmeter.callbacks import CostModel
from llmeter.callbacks.cost import dimensions
from dotenv import load_dotenv

load_dotenv()

# Configure multiple models for comparison
models = [
    {
        "name": "GPT-4o",
        "endpoint": OpenAICompletionEndpoint(
            model_id="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        "pricing": {"input": 2.50, "output": 10.00}
    },
    {
        "name": "GPT-4o-mini",
        "endpoint": OpenAICompletionEndpoint(
            model_id="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        "pricing": {"input": 0.15, "output": 0.60}
    }
]

# Test prompt
test_prompt = "Explain why latency percentiles matter in production AI systems."

async def test_model(model_config):
    """Test a single model and return results"""
    print(f"\nTesting {model_config['name']}...")
    
    # Create cost model for this model
    cost_model = CostModel(
        request_dims=[
            dimensions.InputTokens(price_per_million=model_config['pricing']['input']),
            dimensions.OutputTokens(price_per_million=model_config['pricing']['output']),
        ]
    )
    
    # Create payload
    payload = model_config['endpoint'].create_payload(
        test_prompt,
        max_tokens=200
    )
    
    # Run load test
    load_test = LoadTest(
        endpoint=model_config['endpoint'],
        payload=payload,
        sequence_of_clients=[1, 5],
        output_path=f"./results/{model_config['name'].lower().replace('-', '_')}",
        callbacks=[cost_model]
    )
    
    results = await load_test.run()
    
    # Extract key metrics
    stats = results.results[5].stats  # Get stats from 5-client run
    
    return {
        "model": model_config['name'],
        "total_requests": stats['total_requests'],
        "avg_latency": stats['time_to_last_token-average'],
        "p50_latency": stats['time_to_last_token-p50'],
        "p99_latency": stats['time_to_last_token-p99'],
        "requests_per_minute": stats['requests_per_minute'],
        "total_cost": stats['cost_total'],
        "cost_per_request": stats['cost_per_request-average'],
        "output_tps": stats['output_tps']
    }

async def main():
    print("=" * 60)
    print("Multi-Provider Model Comparison")
    print("=" * 60)
    
    results = []
    
    # Test each model
    for model_config in models:
        try:
            result = await test_model(model_config)
            results.append(result)
        except Exception as e:
            print(f"Error testing {model_config['name']}: {e}")
    
    # Print comparison table
    print("\n" + "=" * 100)
    print(f"{'Model':<15} {'Avg Latency':<12} {'P99 Latency':<12} {'RPM':<10} {'Cost/Req':<12} {'Output TPS':<12}")
    print("=" * 100)
    
    for result in results:
        print(f"{result['model']:<15} "
              f"{result['avg_latency']:<12.2f} "
              f"{result['p99_latency']:<12.2f} "
              f"{result['requests_per_minute']:<10.1f} "
              f"${result['cost_per_request']:<11.6f} "
              f"{result['output_tps']:<12.1f}")
    
    print("=" * 100)
    
    # Find best model for different criteria
    if results:
        fastest = min(results, key=lambda x: x['avg_latency'])
        cheapest = min(results, key=lambda x: x['cost_per_request'])
        highest_throughput = max(results, key=lambda x: x['requests_per_minute'])
        
        print(f"\n🏆 Fastest (Avg Latency): {fastest['model']} ({fastest['avg_latency']:.2f}s)")
        print(f"💰 Cheapest (Cost/Request): {cheapest['model']} (${cheapest['cost_per_request']:.6f})")
        print(f"🚀 Highest Throughput: {highest_throughput['model']} ({highest_throughput['requests_per_minute']:.1f} RPM)")

if __name__ == "__main__":
    asyncio.run(main())
