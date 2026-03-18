import pandas as pd
from src.utils import ExtractedData
from typing import List
import json
import os

class Evaluator:
    def __init__(self, ground_truth_path: str):
        """Load ground truth data"""
        self.ground_truth = pd.read_json(ground_truth_path, orient="records")
        # FIX: Convert review_id to string to match results
        self.ground_truth['review_id'] = self.ground_truth['review_id'].astype(str)
    
    def calculate_metrics(self, results: List[ExtractedData]) -> dict:
        """Calculate performance metrics"""
        results_df = pd.DataFrame([r.dict() for r in results])
        
        # FIX: Ensure review_id is string in results too
        results_df['review_id'] = results_df['review_id'].astype(str)
        
        merged = pd.merge(results_df, self.ground_truth, 
                         on="review_id", suffixes=("_pred", "_true"))
        
        # Accuracy metrics
        total = len(merged)
        
        # FIX: Handle case-insensitive comparison and strip whitespace
        correct_products = (
            merged['product_pred'].str.lower().str.strip() == 
            merged['product_true'].str.lower().str.strip()
        ).sum()
        
        correct_sentiments = (
            merged['sentiment_pred'].str.lower().str.strip() == 
            merged['sentiment_true'].str.lower().str.strip()
        ).sum()
        
        # Failure rate (malformed outputs)
        failure_rate = (merged['product_pred'].isin(['error', 'unknown'])).mean() * 100
        
        # Edge case performance (reviews 4, 6, 10 in our data)
        edge_case_ids = ["4", "6", "10"]
        edge_performance = merged[merged['review_id'].isin(edge_case_ids)]
        
        if len(edge_performance) > 0:
            edge_accuracy = (
                edge_performance['sentiment_pred'].str.lower().str.strip() == 
                edge_performance['sentiment_true'].str.lower().str.strip()
            ).mean() * 100
        else:
            edge_accuracy = 0.0
        
        return {
            "overall_accuracy": (correct_products + correct_sentiments) / (2 * total) * 100,
            "product_accuracy": correct_products / total * 100,
            "sentiment_accuracy": correct_sentiments / total * 100,
            "failure_rate": failure_rate,
            "edge_case_accuracy": edge_accuracy,
            "avg_confidence": merged['confidence'].mean()
        }
    
    def generate_report(self, baseline_results: List[ExtractedData], 
                       autoprompt_results: List[ExtractedData]) -> dict:
        """Generate comparison report"""
        print("\n" + "="*60)
        print("🎯 AUTOPROMPT EVALUATION REPORT")
        print("="*60)
        
        baseline_metrics = self.calculate_metrics(baseline_results)
        autoprompt_metrics = self.calculate_metrics(autoprompt_results)
        
        report = {
            "baseline": baseline_metrics,
            "autoprompt": autoprompt_metrics,
            "improvement": {
                k: autoprompt_metrics[k] - baseline_metrics[k] 
                for k in baseline_metrics
            }
        }
        
        # Pretty print
        for pipeline, metrics in report.items():
            print(f"\n{pipeline.upper()} RESULTS:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f}")
        
        # Save to file
        os.makedirs("results", exist_ok=True)
        with open("results/benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Summary
        print("\n" + "="*60)
        print("📊 KEY FINDINGS")
        print("="*60)
        print(f"✓ Accuracy Improvement: {report['improvement']['overall_accuracy']:+.1f}%")
        print(f"✓ Failure Rate Change: {report['improvement']['failure_rate']:+.1f}%")
        print(f"✓ Edge Case Boost: {report['improvement']['edge_case_accuracy']:+.1f}%")
        
        return report