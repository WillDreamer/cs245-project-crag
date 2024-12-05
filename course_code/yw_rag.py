import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from openai import OpenAI

from tqdm import tqdm
from time import time
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from FlagEmbedding import BGEM3FlagModel,FlagReranker

#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---

class ChunkExtractor:
    @ray.remote
    def _my_extract_chunks(self, interaction_id, html_source):
         # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "html.parser")
        
        # Remove noise
        NOISE_ELEMENTS = ["script", "style", "aside", "footer", "header", "hgroup", "nav", "search"]
        for tag in soup(NOISE_ELEMENTS):
            tag.decompose()
        
        main_extract_result = soup
        # Extract main content
        main_content = soup.find("main")

        if main_content:
            main_extract_result = main_content
            # Extract articles from main content
            articles = main_content.find_all("article")
            if articles:
                main_extract_result = BeautifulSoup("".join(str(article) for article in articles), 'html.parser')
                print("main->articles")
            else:
                print("main")
        else:
            # Extract articles from original html
            articles = soup.find_all("article")
            if articles:
                main_extract_result = BeautifulSoup("".join(str(article) for article in articles), 'html.parser')
                print("articles")
            else:
                print("original")
        
        text = main_extract_result.get_text(" ", strip=True)  # Use space as a separator, strip whitespaces
        if not text:
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
    
        # Extract meaningful chunks
        # max_tokens = 300
        # chunks = []
        # current_chunk = {"heading": None, "content": ""}
        # current_tokens = 0

        # for element in main_extract_result.find_all():
        #     if element.name in ['h1', 'h2', 'h3']:
        #         # Start a new chunk
        #         if current_chunk['content']:
        #             chunks.append(current_chunk)
        #         current_chunk = {"heading": element.get_text(strip=True), "content": ""}
        #         current_tokens = 0
        #     elif element.name in ['p', 'li', 'span', 'div']:
        #         # Append content to the current chunk
        #         text = element.get_text(strip=True)
        #         tokens = len(text.split())
        #         if current_tokens + tokens > max_tokens:
        #             # Save the current chunk and start a new one
        #             chunks.append(current_chunk)
        #             current_chunk = {"heading": current_chunk['heading'], "content": ""}
        #             current_tokens = 0
        #         current_chunk['content'] += text + " "
        #         current_tokens += tokens
        
        # # Add the last chunk
        # if current_chunk['content']:
        #     chunks.append(current_chunk)
        
        return interaction_id, chunks

    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._my_extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"]
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(response_ref)  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # initialize the model with vllm server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            # initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def init_retriever(self, search_results):
        # PARAMS
        hf_path="sentence-transformers/all-MiniLM-L6-v2"
        bge_large_path="models/bge-base-en-v1.5"
        m3_path="models/bge-m3"
        parent_chunk_size=700
        parent_chunk_overlap=150
        child_chunk_size=200
        child_chunk_overlap=50
        max_length = 12000
        recall_k = 50
        batch_size = 32
        top_k_search = 5
        separators=['.', "\n", " ", ""]
        token_path="meta-llama/Llama-3.2-3B-Instruct"
        # stopwords_list = 'models/processed_data/stopwords_list.npy'
        rerank_path ='BAAI/bge-reranker-v2-m3'

        hf_embeddings = HuggingFaceEmbeddings(model_name=hf_path,
                                                   encode_kwargs={'batch_size': batch_size,
                                                                  'normalize_embeddings': True})


        reranker = FlagReranker(rerank_path, use_fp16=True)
        tokenizer = AutoTokenizer.from_pretrained(token_path, clean_up_tokenization_spaces=True)
        parent_text_splitter = CharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separator=' '
        ) 
        child_text_splitter = CharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separator=' ',
        )
        # stopwords_list = np.load(stopwords_list).tolist()
        
        # init retriever
        docs = []
        hashes = set()
        for idx, html in tqdm(enumerate(search_results[:top_k_search])):
            html_content = html['page_result']
            hash_value = hash(html_content)
            if hash_value in hashes or len(html_content) == 0:
                continue
            hashes.add(hash_value)
            
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=" ", strip=True).lower()
            text = html['page_snippet'].lower() + '\n\n' + text
            inputs = tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=False)
            if len(inputs) == max_length:
                text = tokenizer.decode(inputs)
                print('exceed html max size')
                
            metadata ={}
            metadata["start_index"] =idx
            docs.append(Document(page_content=text, metadata=metadata))
         
        if len(docs) == 0:
            return False
        hf_vectorstore = Chroma(
            collection_name="hf_split_parents", embedding_function=hf_embeddings
        )
        hf_retriever = ParentDocumentRetriever(
            vectorstore=hf_vectorstore,
            docstore=InMemoryStore(),
            child_splitter=child_text_splitter,
            parent_splitter=parent_text_splitter,
            search_kwargs = { 'k': recall_k })
        hf_retriever.add_documents(docs, ids=None) 
        retriever = hf_retriever
        
        return retriever, reranker
    
    def get_retrieve_res(self, retriever, reranker, query, k=5):
        torch.torch.cuda.empty_cache()
        docs = retriever.get_relevant_documents(query)
        print('len docs',len(docs))
        if docs ==[]:
            return [""]
        if len(docs) <= k:
            return [doc.page_content for doc in docs]
        
        with torch.no_grad():
            sentence_pairs = [[query, doc.page_content]  for doc in docs]
            sim =  reranker.compute_score(sentence_pairs, normalize=True, batch_size=16)
            indexs = torch.topk(torch.tensor(sim), min(k, len(docs))).indices
            del sim
            torch.torch.cuda.empty_cache()
            docs = [docs[idx].page_content for idx in indexs]
            
        return docs 

     
    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        # chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
        #     batch_interaction_ids, batch_search_results
        # )

        # # Calculate all chunk embeddings
        # chunk_embeddings = self.calculate_embeddings(chunks)

        # # Calculate embeddings for queries
        # query_embeddings = self.calculate_embeddings(queries)
        
        # ==========================
        # ==========================

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            # query_embedding = query_embeddings[_idx]

            # # Identify chunks that belong to this interaction_id
            # relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # # Filter out the said chunks and corresponding embeddings
            # relevant_chunks = chunks[relevant_chunks_mask]
            # relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # # Calculate cosine similarity between query and chunk embeddings,
            # cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

            # # and retrieve top-N results.
            # retrieval_results = relevant_chunks[
            #     (-cosine_scores).argsort()[:NUM_CONTEXT_SENTENCES]
            # ]
            
            # ==============================
            search_results = batch_search_results[_idx]
            retriever, reranker = self.init_retriever(search_results)
            retrieval_results = self.get_retrieve_res(retriever, reranker, query)
            # ==============================
            
            # You might also choose to skip the steps above and 
            # use a vectorDB directly.
            batch_retrieval_results.append(retrieval_results)
        
        # Prepare formatted prompts from the LLM        
        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)

        # Generate responses via vllm
        # note that here self.batch_size = 1
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # randomness of the sampling
                # skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
            )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False
            )
            answers = []
            for response in responses:
                answers.append(response.outputs[0].text)

        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """        
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message 
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                # there is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts