"""
Standard prompt definitions for benchmarking.
Three lengths designed to stress different parts of the inference pipeline:
  - SHORT:  ~30 tokens input  -> tests scheduling/TTFT overhead
  - MEDIUM: ~80 tokens input  -> typical agent task
  - LONG:   ~400 tokens input -> context-heavy orchestration task
"""

PROMPTS = {
    "short": (
        "Write a Python function that returns the nth Fibonacci number using memoization."
    ),

    "medium": (
        "Write a Python function that reads a CSV file using the standard library "
        "(no pandas), computes the mean and standard deviation for each numeric column, "
        "and returns a dictionary mapping column names to (mean, stdev) tuples. "
        "Include error handling for missing files and non-numeric values. "
        "Add a brief docstring and type hints."
    ),

    "long": (
        "You are working on a scientific computing project that processes large datasets "
        "from particle physics experiments. The data pipeline has the following stages:\n\n"
        "1. Ingestion: Read ROOT files containing event data (using uproot), "
        "validate schema, and convert to Apache Arrow format.\n"
        "2. Filtering: Apply physics cuts — select events where leading jet pT > 25 GeV, "
        "|eta| < 2.5, and missing transverse energy MET > 20 GeV.\n"
        "3. Feature extraction: Compute derived quantities including delta-R between "
        "jet pairs, invariant mass of the two leading jets, and HT (scalar sum of jet pTs).\n"
        "4. Output: Write filtered, feature-enriched events to Parquet with appropriate "
        "partitioning by run number.\n\n"
        "Please implement this pipeline as a Python module with the following:\n"
        "- A `Pipeline` class with `run(input_files, output_dir)` method\n"
        "- Each stage as a separate method with clear docstrings\n"
        "- Logging at INFO level for progress and DEBUG for per-event details\n"
        "- Type hints throughout\n"
        "- A `__main__` block using argparse for CLI usage\n"
        "- Unit tests for the filtering and feature extraction stages using pytest "
        "and synthetic data\n\n"
        "Focus on correctness and clarity. Use numpy for vectorized operations where possible."
    ),
}
