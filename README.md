# RAG Parameter Evaluation for Enterprise Documentation

Research code for evaluating Retrieval-Augmented Generation (RAG) parameters on enterprise technical documentation.

## Overview

This repository contains the evaluation framework used to systematically study the impact of chunk size and embedding model choice on RAG performance for enterprise technical documentation. The study uses SAP Business One documentation (4,544 pages across 51 PDF files) as a representative corpus for enterprise ERP documentation.

## Research Questions

1. What chunk sizes optimize RAG performance for enterprise documentation across different embedding models?
2. Which embedding model performs best with enterprise documentation?
3. How do chunk parameters and embedding models interact?
4. How much do optimized RAG configurations improve over traditional IR baselines?

## Key Findings

- **Optimal chunk sizes vary significantly by embedding model**: 750 characters (E5) to 3,000 characters (Gemini 004)
- **Dense retrieval consistently outperforms sparse baselines**: 18 of 19 dense configurations exceeded the best BM25 baseline
- **Gemini text-embedding-004 achieved best overall performance**: 0.846 nv_accuracy with 3,000-character chunks
- **Retrieval quality is the primary determinant of answer quality**: Strong correlation (ρ ≈ 0.97) between context precision and end-to-end accuracy

## Experimental Design

### Synthetic Dataset Generation

Since no public Q&A dataset exists for SAP Business One documentation, this study employs the RAGAS TestsetGenerator framework to create corpus-specific evaluation data:

**Knowledge Graph Construction**:
- Documents processed and added as graph nodes
- LLM-based headline extraction for logical sectioning
- Semantic clustering via embedding similarity
- Named entity recognition for cross-document connections

**Question-Answer Synthesis**:
- **Single-hop queries**: Direct information from one document section
- **Multi-hop queries**: Require combining information across multiple sections
- **Dual-LLM approach**: Gemini Flash 2.0 for scenario generation, Claude 3.7 Sonnet for Q&A synthesis
- **200 questions total** ensuring comprehensive corpus coverage

### Embedding Models Evaluated
- **multilingual-E5-large**: 560M parameters, contrastive learning
- **Nomic Embed V2-MoE**: Mixture-of-experts with 8 expert blocks
- **BGE-M3**: Multi-functionality model supporting dense, sparse, and token-level representations
- **Gemini text-embedding-004**: Google's production encoder (API-based)

### Baselines
- **BM25**: Okapi BM25 with default parameters
- **TF-IDF**: Traditional sparse retrieval with unigrams and bigrams
- **Random**: Random chunk selection (lower bound)

### Parameters Tested
- **Chunk sizes**: 300-16,000 characters (model-dependent based on context windows)
- **Chunk overlap**: Fixed at 20%
- **Retrieved chunks (K)**: Fixed at 5
- **Chunking strategy**: Boundary-respecting fixed-size chunks

### Evaluation Metrics (RAGAS Framework)
- **nv_accuracy**: NVIDIA Answer Accuracy for end-to-end performance
- **answer_correctness**: F1-based factual and semantic accuracy
- **factual_correctness**: Pure factual accuracy without semantic similarity
- **faithfulness**: Measures hallucination/grounding in retrieved context
- **context_precision**: Quality of retrieval ranking (Average Precision)
- **context_recall**: Coverage of retrieved contexts

## Project Structure

```
rag-evaluation/
├── config/                     # Configuration files
│   ├── base_config.yaml        # Common evaluation settings
│   └── approaches/             # Model-specific configurations
├── retrievers/                 # Retriever implementations
│   ├── base_retriever.py       # Abstract base class
│   ├── embedding_retriever.py  # Dense retrieval (E5, BGE-M3, Nomic)
│   ├── gemini_retriever.py     # Gemini API integration
│   ├── bm25_retriever.py       # BM25 baseline
│   ├── tfidf_retriever.py      # TF-IDF baseline
│   └── random_retriever.py     # Random baseline
├── core/                       # Core evaluation components
│   ├── data_manager.py         # Dataset and document handling
│   ├── evaluation.py           # RAGAS evaluation framework
│   └── experiment.py           # Experiment orchestration
├── notebooks/                  # Jupyter notebooks
│   ├── embedding_server.ipynb  # GPU-accelerated embedding service
│   ├── dataset_generation.ipynb  # Synthetic Q&A generation
│   └── results_analysis.ipynb  # Results analysis and visualization
├── scripts/                    # Utility scripts
│   ├── run_comparison.py       # Multi-approach comparison
│   └── find_latest_eval_jsons.py  # Result file management
└── main.py                     # Main experiment entry point
```

## Setup

### Requirements
- Python 3.12.2+
- CUDA-capable GPU (recommended for embedding generation)
- API keys for Gemini and DeepSeek models

### Installation

Using uv (recommended):
```bash
# Windows
.\setup-uv.ps1

# Linux/Mac
./setup-uv.sh
```

Traditional pip:
```bash
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file with necessary API keys:
```bash
# API Keys
GOOGLE_API_KEY=your_google_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# Optional: Remote embedding server
EMBEDDING_API_URL=https://your-colab-tunnel.ngrok-free.app
EMBEDDING_API_TOKEN=your_secure_token
```

## Usage

### Running Individual Configurations

Evaluate a specific approach:
```bash
python main.py --config config/approaches/gemini_c3000_o600.yaml
```

### Running the Full Evaluation Suite

Compare all approaches:
```bash
python scripts/run_comparison.py --approaches gemini_c3000_o600 bge_m3_c2000_o400 nomic_c1200_o240 e5_c750_o150 bm25_c8000_o1600 --cached-docs "path/to/cached_documents.pkl" --test-data "path/to/test_data.jsonl"
```

### Synthetic Dataset Generation

To regenerate the evaluation dataset:
1. Open `notebooks/dataset_generation.ipynb` in Google Colab
2. Configure API keys in Colab secrets
3. Upload enterprise documents to Google Drive
4. Run all cells to generate Q&A pairs with comprehensive corpus coverage

### GPU-Accelerated Embeddings

For large-scale evaluation, use the provided Colab notebook:
1. Open `notebooks/embedding_server.ipynb` in Google Colab (GPU runtime)
2. Set up ngrok tunnel and authentication token
3. Configure `EMBEDDING_API_URL` and `EMBEDDING_API_TOKEN` in your `.env`

## Results

Evaluation results are stored in `data/evaluation/` with timestamps:
- `evaluation_results_{approach}_{timestamp}.json`: Full results with per-question metrics
- `evaluation_results_{approach}_{timestamp}.csv`: Tabular format for analysis
- `comparison_{timestamp}.json`: Cross-approach comparison

## Reproducing Results

The complete evaluation matrix (27 configurations × 200 questions = 5,400 evaluations) can be reproduced by:

1. Setting up the environment as described above
2. Running all configurations in `config/approaches/`
3. Using the analysis notebook to generate figures and tables

**Note**: Full reproduction requires significant computational resources (~8 hours GPU time for embeddings, ~48 hours for evaluation) and API costs (~€100 for Gemini/DeepSeek calls).


## License

This research code is released under the MIT License. See `LICENSE` for details.

## Acknowledgments

- RAGAS framework for automated RAG evaluation
- SAP for publicly available Business One documentation