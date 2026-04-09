# AutoPrompt v2.0 Enterprise Framework 🚀

AutoPrompt v2.0 is a high-performance, asynchronous LLM prompt optimization and benchmarking framework powered by **Groq** and **Llama-3.1**. It automates the search for optimal data extraction prompts using evolutionary meta-prompting and rigorous train/test validation.

---

## 🏗️ System Architecture

AutoPrompt v2.0 transforms manual prompt engineering into an automated, 3-phase optimization pipeline:

1.  **Phase 1: Meta-Prompt Optimization**: A Meta-LLM generates 4 diverse prompt strategies. These are tested against a **Train Set** using Ground Truth labels. The system locks the highest-scoring candidate as the "God Prompt" and collects perfect responses for **Few-Shot RAG injection**.
2.  **Phase 2: Asynchronous Execution**: The framework fires the Baseline and Optimized pipelines concurrently using `asyncio`. It utilizes **Semaphore pacing** to maximize throughput while staying within API rate limits.
3.  **Phase 3: Formal Evaluation**: Results are compared on a hidden **Test Set** to measure real-world performance, overall accuracy, and hallucinatory failure rates.

## ✨ Key Features

*   **⚡ Async Scalability**: Built with `AsyncGroq` and `asyncio.gather` for 10x faster concurrent processing.
*   **🧠 Meta-Prompting Engine**: Uses AI to engineer its own prompts, evolving strategies dynamically based on training success.
*   **🔒 Native JSON Mode**: Enforces `response_format={"type": "json_object"}` at the backend level to guarantee 100% parseable structured data.
*   **🧪 Train/Test Splitting**: Prevents "Data Leakage" by optimizing on a training batch and validating on an independent test batch.
*   **🛡️ Robust Fault Tolerance**: Implements `tenacity` retries with exponential backoff for a zero-crash extraction pipeline.
*   **📖 Industry-Level Documentation**: Includes a [Technical Deep Dive](technical_deep_dive.md) detailing the architecture and logic flow.

## 🛠️ Setup Instructions

1.  **Clone & Navigate**:
    ```bash
    git clone https://github.com/iussg/auto-prompting
    cd auto-prompting
    ```

2.  **Environment Setup**:
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate.ps1  # Windows
    # source venv/bin/activate   # Linux/Mac
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys**:
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_gsk_key_here
    ```

## 🚀 Usage

Check your API connectivity and available Groq models:
```bash
python check_model.py
```

Run the v2.0 Optimization Pipeline:
```bash
python main.py
```

### CLI Arguments
*   `--train-limit`: Number of rows to use for prompt evolution (default: 5).
*   `--test-limit`: Number of rows for the final benchmark (default: 30).
*   `--data`: Path to custom reviews CSV/JSON.
*   `--model`: Override the model (e.g., `llama-3.1-70b-versatile`).

## 📊 Benchmark Reporting
Once complete, the system generates a detailed audit trail in `logs/run.log` and a final performance comparison in `results/benchmark_report.json`.
