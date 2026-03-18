# AutoPrompt MVP Benchmark

AutoPrompt MVP is a simple automated prompting benchmark and testing framework utilizing Google's Gemini Models. This project evaluates prompt variants by running a Baseline pipeline and an AutoPrompt optimization pipeline against a dataset of reviews for precise information extraction tasks (e.g., sentiment analysis and product extraction).

## Project Structure
- `data/`: Contains dataset files (`reviews.csv`) and `ground_truth.json` for validation.
- `src/`: The main pipeline code containing baseline evaluation, prompt generation, logging setup via Loguru, and API integration.
- `config/`: Pipeline configuration including optimization settings like temperature, limits, and explicit model choices.
- `results/`: Benchmark reports generated after pipelines are executed.
- `logs/`: Application logs with rotation and retention.

## Setup Instructions
1. Clone the repository and navigate into it.
2. Setup a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure API Keys:
   Create a `.env` file in the root directory and add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

## Usage
- You can run `python check_model.py` to ensure your API tokens work and discover which Gemini versions you have access to.
- Run the full benchmark pipeline via:
  ```bash
  python main.py
  ```
  *Note: The script includes rate-limiting sleep timers out-of-the-box to prevent exhausting the Google Gemini API free-tier quotas.*

## Report Generation
Once completed, the pipeline outputs metrics assessing overall extraction accuracy, sentiment, and failure rates in `results/benchmark_report.json`.
