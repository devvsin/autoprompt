import asyncio
import os
import argparse
import time
from dotenv import load_dotenv
import pandas as pd
from src.utils import load_reviews, save_results, Review
from src.baseline import BaselinePipeline
from src.autoprompt import AutoPromptEngine
from src.evaluator import Evaluator
from src.config_loader import load_secure_config
from loguru import logger

load_dotenv()

logger.remove()
logger.add("logs/run.log", rotation="500 MB", retention="10 days", level="INFO")
logger.add(lambda msg: print(msg, end=""), level="INFO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoPrompt v2.0 Enterprise: Train/Test Pipeline")
    parser.add_argument("--test-limit", type=int, default=0, help="Max test reviews (0 = process all)")
    parser.add_argument("--train-limit", type=int, default=5, help="Number of reviews to use for prompt evolution")
    parser.add_argument("--data", default="data/reviews_30.csv", help="Dataset path")
    parser.add_argument("--truth", default="data/ground_truth_30.json", help="Ground Truth JSON path")
    parser.add_argument("--model", default=None, help="LLM to use (e.g. llama-3.1-8b-instant)")
    return parser.parse_args()


async def execute_pipeline(pipeline, reviews, output_path, semaphore):
    """Run pipeline concurrently over all reviews, bound by free tier limits."""
    logger.info(f"Firing {len(reviews)} reviews concurrently (Rate-limited to 15 chunks)...")
    start_time = time.time()
    
    async def _bounded_process(rev):
        async with semaphore:
            await asyncio.sleep(2.0)  # Presentation Pacemaker: strictly controls 30 RPM without 429 warnings
            return await pipeline.process(rev)
    
    # gather fires them all off simultaneously bound by the semaphore
    tasks = [_bounded_process(review) for review in reviews]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    final_results = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Review failed catastrophically: {r}")
        elif r is not None:
            final_results.append(r)
            
    # Save the output
    save_results(final_results, output_path)
    elapsed = time.time() - start_time
    logger.info(f"Pipeline finished {len(final_results)} items in {elapsed:.2f}s")
    return final_results


async def main_async():
    API_KEY = os.getenv("GROQ_API_KEY")
    if not API_KEY:
        raise ValueError("❌ GROQ_API_KEY not found!")
    
    config = load_secure_config()
    args = parse_args()
    
    if args.model:
        config.generator_model = args.model
        config.scoring_model = args.model

    # Load Full Data
    reviews_df = load_reviews(args.data)
    ground_truth_df = pd.read_json(args.truth, orient="records")
    # convert ID to string
    reviews_df['review_id'] = reviews_df['review_id'].astype(str)
    ground_truth_df['review_id'] = ground_truth_df['review_id'].astype(str)

    # TRAIN / TEST SPLITTING
    # To prevent Data Leakage: grab first N for training, rest for testing
    train_size = min(args.train_limit, len(reviews_df) // 4)
    train_df = reviews_df.head(train_size)
    test_df = reviews_df.iloc[train_size:]
    
    if args.test_limit > 0:
        test_df = test_df.head(args.test_limit)

    train_reviews = [Review(review_id=str(r["review_id"]), review_text=r["review_text"]) for _, r in train_df.iterrows()]
    test_reviews = [Review(review_id=str(r["review_id"]), review_text=r["review_text"]) for _, r in test_df.iterrows()]
    
    logger.info("=" * 60)
    logger.info(f"[*] AutoPrompt v2.0 Enterprise Pipeline")
    logger.info(f"Train Set: {len(train_reviews)} reviews | Test Set: {len(test_reviews)} reviews")
    logger.info("=" * 60)

    # Free tier semaphore: Prevents rate limits when dataset exceeds 30 items
    api_semaphore = asyncio.Semaphore(15)

    # 1. OPTIMIZATION PHASE (Meta-Prompting / Finding the God Prompt)
    autoprompt = AutoPromptEngine(config)
    logger.info("PHASE 1: Meta-Prompt Optimization (Generating RAG God Prompt)...")
    await autoprompt.optimize_prompt(train_reviews, ground_truth_df)
    
    # 2. RUN BASELINE AGAINST TEST
    baseline = BaselinePipeline(config)
    logger.info(f"PHASE 2a: Firing Baseline Pipeline ({len(test_reviews)} items)...")
    baseline_results = await execute_pipeline(baseline, test_reviews, "results/baseline_results.json", api_semaphore)
    
    # 3. RUN THE EVOLVED PROMPT AGAINST TEST
    logger.info(f"PHASE 2b: Firing AutoPrompt Pipeline ({len(test_reviews)} items)...")
    autoprompt_results = await execute_pipeline(autoprompt, test_reviews, "results/autoprompt_results.json", api_semaphore)
    
    # 4. EVALUATION
    logger.info("PHASE 3: Running Benchmark Evaluation...")
    evaluator = Evaluator(args.truth)
    
    # Evaluate ONLY on the Test Set ids!
    test_ids = [r.review_id for r in test_reviews]
    b_test = [r for r in baseline_results if r.review_id in test_ids]
    a_test = [r for r in autoprompt_results if r.review_id in test_ids]
    
    evaluator.generate_report(b_test, a_test)

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    asyncio.run(main_async())

if __name__ == "__main__":
    main()