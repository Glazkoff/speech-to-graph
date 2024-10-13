from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)
import torch
import os

from baseHandler import BaseHandler
from rich.console import Console
import logging
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document

# from nltk import sent_tokenize

logger = logging.getLogger(__name__)

console = Console()

class LangchainGraphConstructionHandler(BaseHandler):
    """
    Handles the knowledge graph construction part.
    """
    def setup(
        self,
        neo4j_uri="bolt://localhost:7999",
        neo4j_username="neo4j",
        neo4j_password="testtest",
        langchain_api_key="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",
        langchain_tracing_v2="true",
        langchain_endpoint="https://api.smith.langchain.com",
        langchain_project="test-knowledge-graph",
        ollama_llm="gemma2",
    ):
        os.environ["NEO4J_URI"] = neo4j_uri
        os.environ["NEO4J_USERNAME"] = neo4j_username
        os.environ["NEO4J_PASSWORD"] = neo4j_password
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing_v2
        os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint
        os.environ["LANGCHAIN_PROJECT"] = langchain_project
        os.environ["OLLAMA_LLM"] = ollama_llm
        self.graph = Neo4jGraph(refresh_schema=True, sanitize=True)
        self.llm = OllamaLLM(model=os.environ["OLLAMA_LLM"], temperature=0)
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        print(f'TEST INVOKE: {self.llm.invoke("Hi! It is test invoke.")}')

        dummy_input_text = """
Nikita Glazkov, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""

        dummy_documents = [Document(page_content=dummy_input_text)]

        graph_documents = self.llm_transformer.convert_to_graph_documents(dummy_documents)
        print(f"Nodes:{graph_documents[0].nodes}")
        print(f"Relationships:{graph_documents[0].relationships}")
        
    def process(self, prompt):
        logger.debug("infering language model...")

        language_code = None
        if isinstance(prompt, tuple):
            prompt, language_code = prompt

        documents = [Document(page_content=prompt)]
        graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
        console.print(f"[green]NODES:{graph_documents[0].nodes}")
        console.print(f"[green]RELATIONSHIPS:{graph_documents[0].relationships}")

        yield (f"{graph_documents[0].nodes}", language_code)
