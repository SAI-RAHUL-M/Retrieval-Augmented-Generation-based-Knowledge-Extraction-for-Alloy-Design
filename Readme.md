# Refractory High-Entropy Alloys Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) chatbot specialized in refractory high-entropy alloys (RHEAs) research. The chatbot provides information about mechanical properties and characteristics of these advanced materials based on comprehensive academic research.

## Project Overview

The project follows these key steps:

1. **Data Collection:** Comprehensive data compilation on mechanical properties of refractory high-entropy alloys from primary research papers and 60 reference papers.
2. **Data Extraction:** Systematic extraction of titles, abstracts, and conclusions from each paper into a structured database.
3. **Embedding Generation:** Two embedding approaches for semantic representation:
   * MatSciBERT (Materials Science-specific BERT model)
   * Standard BERT embeddings (for comparison)
4. **RAG System Implementation:** 
   * Research: Llama 3.1 (8B parameters) for comprehensive analysis and evaluation
   * Deployment: Gemini-2.0-Flash for production due to storage constraints
5. **Deployment:** Web-based interface using Streamlit for easy access.

## Features

* **Specialized Knowledge Base:** Access insights from 60+ academic papers on refractory high-entropy alloys
* **Domain-Specific Understanding:** MatSciBERT embeddings for enhanced materials science terminology comprehension
* **Scientific Accuracy:** Rigorous evaluation metrics ensure factual consistency
* **User-Friendly Interface:** Intuitive chat interface for querying alloy properties
* **Citation Support:** Source attribution to original research papers
* **Model Comparison:** Side-by-side comparison between MatSciBERT and BERT embedding performances
* **Quality Metrics:** BERTScore and ROUGE metrics for answer evaluation
* **Interactive Visualization:** Performance metrics displayed through interactive charts

## Live Demo

Try the chatbot at: 
Streamlit: [Streamlit](https://alloy-chatbot-qwsuv56mpnpnasulqzsnph.streamlit.app/)

## Technical Architecture

```
├── data/
│   ├── raw/                  # Original research papers
│   ├── processed/            # Extracted text in structured format
│   └── embeddings/           # Generated vector embeddings
│       ├── bert/             # Standard BERT embeddings
│       └── matscibert/       # MatSciBERT domain-specific embeddings
├── models/
│   ├── matscibert/           # MatSciBERT model files
│   └── bert/                 # BERT model files
├── src/
│   ├── data_extraction.py    # Scripts for extracting data from papers
│   ├── embedding.py          # Code for generating embeddings
│   ├── rag_system.py         # RAG implementation
│   └── app.py                # Streamlit application
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## RAG System Implementation

The chatbot utilizes a Retrieval-Augmented Generation approach with these core components:

### 1. Query Encoder
- Converts user queries into high-dimensional vector representations
- Normalizes scientific terms and performs vocabulary mapping
- Uses configurable BERT or MatSciBERT models for embedding generation

### 2. Retriever
- Employs vector similarity search to fetch relevant passages
- Implements adjustable similarity thresholds (default 0.55)
- Uses Top-K selection (optimized at K=2) for best performance

### 3. Context Formatter
- Processes retrieved documents into structured prompts
- Manages token budget (3000 tokens) to optimize context window
- Highlights query-relevant segments within documents

### 4. Generator
- **Research Phase:** Powered by Llama 3.1 (8B parameters) for comprehensive analysis
- **Deployment Phase:** Integrated with Gemini-2.0-Flash for production due to storage limitations
- Features specialized scientific prompt templates
- Uses optimized parameters (temperature=0.5, top-p=0.92) for scientific accuracy

## Model Selection Note

For our research and evaluation phase, we used Llama 3.1 (8B parameters) due to its robust performance in scientific question answering. However, for the deployed version, we switched to Gemini-2.0-Flash LLM to address storage constraints while maintaining high-quality responses. All performance metrics and evaluations were conducted using the Llama 3.1 model.

## Performance Evaluation

Our system is evaluated using multiple metrics:

### BERTScore
Measures semantic similarity between generated responses and reference texts.

### ROUGE Score
Evaluates textual overlap between retrieved and generated answers.

### Cosine Similarity
Ensures retrieved texts align with query intent.

**Results Summary:**
- MatSciBERT significantly outperforms standard BERT across all metrics
- Average BERTScore F1 for MatSciBERT: 0.82 vs BERT: 0.44
- ROUGE-1 scores show 3-4x improvement with MatSciBERT

## User Interface

The Streamlit application provides an intuitive experience:

- Clean query interface with clear instructions
- Side-by-side model comparison when requested
- Interactive performance metrics visualization
- Source attribution for scientific rigor

## Usage Examples

Sample queries:
* "What are the key mechanical properties of refractory high-entropy alloys?"
* "How does temperature affect the strength of these alloys?"
* "Which composition shows the best high-temperature stability?"
* "Compare the creep resistance of different refractory HEAs."
* "What mechanisms contribute to the strengthening of these alloys?"

## Local Development

To run this project locally:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables for the Google API key (for Gemini model access)
4. Run the application: `streamlit run src/app.py`

Requirements:
- Google API key for Gemini model access
- Pre-computed embeddings in CSV format
- MatSciBERT model files

## Future Development

Planned enhancements:
* Knowledge base expansion with recent research
* Improved numerical accuracy in responses
* Enhanced citation validation
* Support for complex scientific data formats (tables, graphs, equations)
* Multi-modal capabilities for microstructure analysis
* Interactive property visualization
* User feedback integration for continuous improvement

## Citation

If you use this work in your research, please cite our system and the original research papers in the knowledge base.

## Project Contributors

- Kritesh Kumar Gupta Sir
- Ashwin Devan
- M Sai Rahul
- M Srinivasa Sai Kumar Reddy
- Kurakula Prashanth

## Contact

For questions, feedback, or collaboration opportunities, please contact project contributors.
