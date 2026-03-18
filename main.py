import os
from dotenv import load_dotenv
from src.utils import load_reviews, save_results, Review, ExtractedData
from src.baseline import BaselinePipeline
from src.autoprompt import AutoPromptEngine
from src.evaluator import Evaluator
from loguru import logger
import time

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger.remove()
logger.add("logs/run.log", rotation="500 MB", retention="10 days", level="INFO")
logger.add(lambda msg: print(msg, end=""), level="INFO")

def main():
    # Secure API key loading
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise ValueError(
            "❌ GEMINI_API_KEY not found!\n"
            "   1. Create a .env file in project root\n"
            "   2. Add: GEMINI_API_KEY=your_key_here\n"
            "   3. Get your key from: https://aistudio.google.com/app/apikey"
        )
    
    # Load secure configuration
    from src.config_loader import load_secure_config
    config = load_secure_config()
    
    DATA_PATH = "data/reviews.csv"
    GROUND_TRUTH_PATH = "data/ground_truth.json"
    
    logger.info("Starting AutoPrompt MVP Benchmark")
    logger.info("⚠️  Free tier detected - using rate-limited processing")
    
   # Load data
    reviews_df = load_reviews(DATA_PATH)
    
    # LIMIT: Only process first 20 reviews for testing
    reviews_df = reviews_df.head(20)
    
    reviews = [Review(review_id=str(row['review_id']), 
                     review_text=row['review_text']) 
              for _, row in reviews_df.iterrows()]
    
    logger.info(f"Loaded {len(reviews)} reviews (limited to first 20 for testing)")
    
    # Initialize pipelines
    baseline = BaselinePipeline(config)
    autoprompt = AutoPromptEngine(config)
    
    # 1. Run Baseline with rate limiting
    logger.info("Running baseline pipeline...")
    baseline_results = []
    for i, review in enumerate(reviews):
        if i > 0:
            time.sleep(7)  # Free tier: 10 req/min = wait 6 seconds minimum
        baseline_results.append(baseline.process(review))
    save_results(baseline_results, "results/baseline_results.json")
    
    logger.info("⏳ Waiting 60 seconds before AutoPrompt to reset rate limit...")
    time.sleep(60)
    
    # 2. Run AutoPrompt with rate limiting
    logger.info("Running AutoPrompt pipeline...")
    logger.info(f"⏱️  Estimated time: ~{len(reviews) * 15} seconds (with rate limits)")
    autoprompt_results = []
    for i, review in enumerate(reviews):
        if i > 0:
            # Wait between reviews to avoid hitting rate limit
            logger.info(f"⏳ Waiting 15 seconds before next review ({i+1}/{len(reviews)})...")
            time.sleep(15)
        autoprompt_results.append(autoprompt.process(review))
    save_results(autoprompt_results, "results/autoprompt_results.json")
    
    # 3. Evaluate
    logger.info("Running evaluation...")
    evaluator = Evaluator(GROUND_TRUTH_PATH)
    report = evaluator.generate_report(baseline_results, autoprompt_results)
    
    logger.info("✅ Benchmark complete! Check results/benchmark_report.json")

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    main()