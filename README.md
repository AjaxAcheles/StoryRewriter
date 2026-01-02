## Local LLM Story Rewriter Pipeline
 
A sophisticated, offline automation pipeline designed to rewrite long-form fiction using local Large Language Models (via Ollama). 

Unlike simple "copy-paste" tools, this system handles the complexities of context windows, narrative continuity, and hardware safety. It splits manuscripts into token-aware chunks, rewrites them using a customizable style, and seamlessly stitches them back together using semantic vector analysis.

## 🌟 Key Features

* **Token-Aware Chunking:** Intelligently splits text based on exact tokenizer counts (using the specific model's tokenizer) and narrative scene boundaries (`***`, `---`) to ensure no context is lost.
* **Semantic Pivot Stitching:** Uses `sentence-transformers` to mathematically align the "tail" of one rewritten chunk with the "head" of the next, creating invisible seams between segments.
* **Thermal Protection:** Monitors system temperature (via LibreHardwareMonitor or WMI). If the CPU overheats, the system pauses execution and waits for cooldown.
* **Active CPU Throttling:** Automatically disables CPU Turbo Boost (sets max processor state to 99%) during execution to maintain thermal stability on laptops/desktops.
* **Hallucination & Summarization Guards:** Validates AI output to ensure the model isn't lazily summarizing the text or cutting content. It includes auto-retry logic with dynamic prompt hardening.
* **Resume Capability:** The pipeline saves progress chapter-by-chapter.

## 🛠️ Prerequisites

* **Operating System:** Windows 10/11 (Required for `power_utils.py` and `wmi` thermal monitoring).
* **Python:** Version 3.8 or higher.
* **Ollama:** Installed and running locally. [Download Ollama](https://ollama.com).
* **LibreHardwareMonitor (Recommended):** For accurate temperature readings. [Download Here](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor). *Note: Must be running as Admin while the script executes.*

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AjaxAcheles/StoryRewriter.git
    cd StoryRewriter
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install requests wmi pywin32 sentence-transformers nltk scikit-learn numpy
    ```

3.  **Pull your desired LLM:**
    Make sure Ollama is running, then pull the model defined in your config (default is `llama3.1`).
    ```bash
    ollama pull llama3.1
    # or
    ollama pull mistral
    # or
    ollama pull gemma2
    ```

## 🚀 Usage

1.  **Prepare your Input:**
    Place your manuscript (Markdown or Text format) in the `./input/` folder.
    * *Tip:* Use standard chapter headers (e.g., `Chapter 1`) for best detection.

2.  **Configure Style:**
    Edit `config/style_prompt.txt`. This file contains the "Ghostwriter" persona and specific instructions on how you want the prose rewritten (e.g., "Show, don't tell," "Remove filter words").

3.  **Run the Pipeline:**
    ```bash
    python rewriter_pipeline.py config/default.json
    ```

4.  **Monitor Progress:**
    * Console logs will show the current chunk, token usage, and temperature status.
    * Logs are also saved to `rewriter.log`.

5.  **Output:**
    Results are saved to `./output/`:
    * `full_manuscript_rewritten.md`: The complete stitched story.
    * `chapters/`: Individual rewritten chapter files.
    * `metadata.json`: Statistics on processing time and token usage.

## ⚙️ Configuration (`default.json`)

You can customize the entire pipeline via the JSON config file. Key settings include:

### Model Settings
* `model_name`: The Ollama model tag (e.g., `llama3.1`, `mistral`, `gemma`).
* `num_ctx`: Context window size (e.g., `4096`, `8192`). *Ensure your chunk sizes fit within this!*

### Chunking
* `target_chunk_size_tokens`: How much text to process at once (default `600`).
* `overlap_tokens`: How much context from the previous chunk to include (default `300`).

### Stitching
* `model_name`: The embedding model used to match sentences (default `all-MiniLM-L6-v2`).
* `similarity_threshold`: How similar two sentences must be to be considered a "match" (default `0.65`).

### Monitoring
* `max_temp_celsius`: Pause execution if CPU hits this temp (default `75`).
* `resume_temp_celsius`: Resume when cooled to this temp (default `68`).

## 📂 Project Structure

* `rewriter_pipeline.py`: **Main Entry Point.** Orchestrates the loading, chunking, rewriting, and stitching.
* `chunker.py`: Logic for splitting text while respecting scene boundaries (`***`) and sentence endings.
* `local_llm.py`: Handles communication with the Ollama API, including retries and token counting.
* `stitcher.py`: Implements "Windowed Semantic Pivot Stitching" to join rewritten text blocks seamlessly.
* `monitor.py`: Connects to Windows WMI sensors to watch CPU temperatures.
* `power_utils.py`: Managing Windows Power Plans to throttle CPU and prevent overheating.
* `utils.py`: Helper functions for file I/O and regex.

## 🔍 Technical Runtime Breakdown

Here is exactly what happens when you run the pipeline:

1.  **Initialization & Safety Check:**
    * The script loads the config and connects to the local Ollama server.
    * It initializes the `TemperatureGuard` to watch hardware sensors.
    * It triggers `WindowsCpuThrottler` to disable Turbo Boost (limiting CPU state to 99%) to prevent overheating during the intensive workload.

2.  **Ingestion & Analysis:**
    * The manuscript is loaded and split into chapters (using Regex detection or length fallback).
    * The `Chunker` analyzes the text using the **actual LLM tokenizer** (not just word counts) to ensure chunks fit the context window perfectly.

3.  **The Processing Loop (Per Chunk):**
    * **Context Overlap:** Each new chunk includes the last ~300 tokens of the *previous* chunk so the AI knows what just happened.
    * **Inference:** The chunk is sent to Ollama with your `style_prompt`.
    * **Validation:** The output is measured. If the output is significantly shorter than the input (a sign of summarization), the system **rejects** it and retries with a stricter prompt.
    * **Thermal Pause:** Between chunks, the system checks if the CPU is > 75°C. If so, it sleeps until the temperature drops to 68°C.

4.  **Semantic Stitching:**
    * The system takes the rewritten chunks (which now have overlapping narrative content but different wording).
    * It uses `sentence-transformers` to convert sentences into vector embeddings.
    * It scans the "Tail" of Chunk A and the "Head" of Chunk B to find the **Semantic Pivot Point**—the exact sentence where the narrative aligns mathematically.
    * It splices the text at this pivot point, removing the overlap and creating a seamless transition.

5.  **Finalization:**
    * The stitched chapters are saved.
    * CPU settings are restored to normal.
    * Performance metadata is dumped to JSON.

## ⚠️ Troubleshooting

* **"Ollama not accessible":** Ensure you have run `ollama serve` in a separate terminal window.
* **"LibreHardwareMonitor not found":** The script will fallback to Windows default sensors, which can be inaccurate. Run LibreHardwareMonitor as Administrator for best results.
* **Validation Failures:** If the AI keeps summarizing text (producing output shorter than input), try increasing `target_chunk_size_tokens` or editing the `additional_instructions_on_retry` in `default.json`.

## 📜 License
## MIT
