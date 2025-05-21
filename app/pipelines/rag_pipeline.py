# app/pipelines/rag_pipeline.py
from app.config import rag_config as config
from qdrant_client import QdrantClient, models
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader  # Example loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores.qdrant import Qdrant
import os

class RAGPipelineLoader:
    def __init__(self):
        self.embeddings = self.load_embeddings()
        self.source = None
        self.retriever = None
        self.llm = self.load_model_pipeline()
        self.prompt = self.load_prompt_template()
        self.pipeline = None

    def load_embeddings(self):
        embedding_type = config.EMBEDDING_MODEL_TYPE
        if embedding_type == "local":
            return SentenceTransformerEmbeddings(model_name=config.LOCAL_EMBEDDING_MODEL_NAME)
        elif embedding_type == "huggingface":
            return HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_EMBEDDING_MODEL_NAME)
        else:
            raise ValueError(f"Unsupported embedding model type: {embedding_type}")

    def load_retriever(self, retriever_name, embeddings):
        if retriever_name == "qdrant":
            client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY
            )
            retriever = Qdrant(
                client=client,
                collection_name=config.QDRANT_COLLECTION_NAME,
                embeddings=embeddings
            ).as_retriever()
        else:
            raise ValueError("Unknown retriever source")

        return retriever

    def load_model_pipeline(self, max_new_tokens=100):
        llm_provider = config.LLM_PROVIDER.lower()  # Đọc cấu hình nhà cung cấp LLM

        if llm_provider == "ollama":
            llm_model_name = config.LLM_MODEL_NAME_OLLAMA
            ollama_base_url = config.OLLAMA_BASE_URL
            return Ollama(base_url=ollama_base_url, model=llm_model_name)
        elif llm_provider == "huggingface":
            model_id = config.LLM_MODEL
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)

            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                top_k=config.TOP_K if hasattr(config, 'TOP_K') else 30,
                top_p=config.TOP_P if hasattr(config, 'TOP_P') else 0.95,
                temperature=config.TEMPERATURE if hasattr(config, 'TEMPERATURE') else 0.7
            )
            return HuggingFacePipeline(pipeline=pipe)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Choose 'ollama' or 'huggingface'.")

    def load_prompt_template(self):
        template = (
            "Bạn là trợ lý ảo thông minh. Hãy trả lời câu hỏi dựa vào ngữ cảnh được cung cấp.\n"
            "Ngữ cảnh: {context}\n"
            "Câu hỏi: {question}\n"
            "Trả lời:"
        )
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def load_rag_pipeline(self, llm, retriever, prompt):
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

    def rag(self, source, question):
        if self.pipeline is None or self.source != source:
            self.retriever = self.load_retriever(source, self.embeddings)
            self.pipeline = self.load_rag_pipeline(self.llm, self.retriever, self.prompt)
            self.source = source

        result = self.pipeline({"query": question})
        return result