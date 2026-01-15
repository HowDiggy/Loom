import asyncio
import time
from rich.console import Console
from rich.progress import Progress
from loom.client import LoomClient

console = Console()

async def run_benchmark():
    # 1. Initialize Client (Points to Spark)
    client = LoomClient()
    
    # 2. Define the workload
    prompt = "Explain the difference between Latency and Throughput in 50 words."
    num_requests = 20  # Let's hit it with 20 parallel streams
    
    console.print(f"[bold]🚀 Starting Stress Test: {num_requests} concurrent requests...[/bold]")
    
    start_time = time.perf_counter()
    
    # 3. Create wrapper for async execution
    # Since LoomClient is synchronous (standard OpenAI), we wrap it in threads
    # to simulate concurrent load from multiple Argus agents.
    loop = asyncio.get_running_loop()
    
    tasks = []
    for _ in range(num_requests):
        # Offload the blocking IO to a thread so they all fire at once
        tasks.append(
            loop.run_in_executor(
                None, 
                client.generate, 
                "System", 
                prompt
            )
        )
    
    # 4. Fire!
    results = await asyncio.gather(*tasks)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # 5. Calculate Stats
    total_generated_tokens = sum(r.token_usage["completion_tokens"] for r in results)
    throughput = total_generated_tokens / duration
    
    console.print(f"\n[bold green]✅ Test Complete[/bold green]")
    console.print(f"Time Taken:      {duration:.2f}s")
    console.print(f"Total Tokens:    {total_generated_tokens}")
    console.print(f"Avg Latency:     {duration:.2f}s (for all 20 to finish)")
    console.print(f"Est. Throughput: [bold cyan]{throughput:.2f} tokens/sec[/]")

if __name__ == "__main__":
    asyncio.run(run_benchmark())