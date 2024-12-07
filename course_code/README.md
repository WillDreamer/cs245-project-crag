# ReadMe
| Haixin Wang, Han Zhang, Nayeli Guzman, Yuwei Xiao

Our RAG model (see `rag_ours.py`) combines multiple essential components to efficiently manage both retrieval and generation tasks. The pipeline includes preprocessing, a parent-child retriever, and a reranker.

## Pipeline Overview

The RAG model consists of several components:
- **Preprocessor**: Parses the raw HTML search results and extracts core text content into documents.
- **Parent-Child Retriever**: Splits documents into parent and child chunks, generates embeddings for each chunk, stores them in a vector database, and retrieves the most relevant chunks by measuring their similarity to the provided query.
- **Reranker**: Ranks the retrieved documents based on relevance.
- **LLM (Large Language Model)**: Generates answers using the most relevant documents retrieved.

This implementation allows for dynamic preprocessing of search results, retrieval, reranking, and query answering.

## Preprocessor

### Search Result Preprocessing
The search results are processed in the `preprocess_search_result` method. This involves:
1. **HTML Parsing**: Using BeautifulSoup to parse HTML and extract relevant content.
2. **Noise Removal**: Tags such as `script`, `style`, `aside`, and `footer` are removed to clean the text.
3. **Main Content Extraction**: Attempts to extract the main content from HTML, such as articles and paragraphs.


## Parent-Child Retriever

In the `init_retriever` method, the retrieval pipeline is set up, including:
- Splitting documents into **parent** and **child chunks** using `CharacterTextSplitter`:
    - **Parent Chunk**: Larger chunks of content that provide broader context.
    - **Child Chunk**: Smaller chunks that focus on specific details within the parent.
- Generates embeddings for each chunk using `all-MiniLM-L6-v2` model
- Stores embeddings in a vector database using `Chroma`

In the `get_retrieve_res` method, the retriever will fetch relevant documents (no more than $recall\_k$) given a query:
```python
docs = retriever.get_relevant_documents(query)
```

This hierarchical retrieval approach allows for efficient document retrieval, improving both context relevance and processing speed.

### Example Parameters Configuration:
```python
parent_chunk_size = 1000
parent_chunk_overlap = 150
child_chunk_size = 200
child_chunk_overlap = 50
recall_k = 50
```

## Reranker

After retrieving the relevant documents, the reranker is used to rank the results based on relevance. This is done in the `get_retrieve_res` method, where each document is scored against the query using the `compute_score` method of the reranker, and the top $rerank\_k$ results will be returned.

### Example Configuration
```python
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
rerank_k = 10
```

## Query Answering

After retrieving and reranking documents, the model generates responses using a **Large Language Model (LLM)**, such as **meta-llama/Llama-3.2-3B-Instruct**. The `batch_generate_answer` method formats the retrieval results into prompts for the LLM and generates answers accordingly.

## Requirements
- `torch`
- `sentence-transformers`
- `transformers`
- `vllm`
- `openai`
- `beautifulsoup4`
- `langchain`
- `langchain_community`
- `FlagEmbedding`

## Running Our Model
```sh
export CUDA_VISIBLE_DEVICES=0
vllm serve meta-llama/Llama-3.2-3B-Instruct --gpu_memory_utilization=0.85 --tensor_parallel_size=1 --dtype="half" --port=8088 --enforce_eager --max_model_len=4096
python generate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --split 1 --model_name "rag_ours" --llm_name "meta-llama/Llama-3.2-3B-Instruct" --is_server --vllm_server "http://localhost:8088/v1"
python evaluate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --model_name "rag_ours" --llm_name "meta-llama/Llama-3.2-3B-Instruct" --is_server --vllm_server "http://localhost:8088/v1" --max_retries 10
```
