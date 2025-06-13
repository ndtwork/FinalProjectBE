# app/pipelines/rag_pipeline.py

from app.config import rag_config as config
from app.utils.settings import get_active_collection     # ← Bổ sung import
from qdrant_client import QdrantClient, models
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores.qdrant import Qdrant
from langchain.schema import Document
import torch
from sentence_transformers import CrossEncoder
import time

class RAGPipelineLoader:
    def __init__(self):
        # 1. Embedding
        self.embeddings = self.load_embeddings()
        # 2. Cache tên collection hiện tại
        self.collection_name = None
        # 3. Các thành phần khác
        self.retriever = None
        self.llm = self.load_model_pipeline()
        self.prompt = self.load_prompt_template()
        self.pipeline = None
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_embeddings(self):
        if config.EMBEDDING_MODEL_TYPE == "local":
            return SentenceTransformerEmbeddings(
                model_name=config.LOCAL_EMBEDDING_MODEL_NAME
            )
        elif config.EMBEDDING_MODEL_TYPE == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=config.HUGGINGFACE_EMBEDDING_MODEL_NAME
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {config.EMBEDDING_MODEL_TYPE}")

    def load_retriever(self, retriever_name, embeddings):
        # 1. Lấy tên collection active (do admin set) hoặc fallback về config
        name = get_active_collection() or config.QDRANT_COLLECTION_NAME
        self.collection_name = name

        # 2. Tạo client & retriever
        client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY
        )
        self.qdrant_client = client

        # 3. Chỉ support "qdrant" cho retriever_name
        if retriever_name != "qdrant":
            raise ValueError(f"Unknown retriever source: {retriever_name}")

        return Qdrant(
            client=client,
            collection_name=name,
            embeddings=embeddings
        ).as_retriever()

    def load_model_pipeline(self, max_new_tokens: int = 100):
        provider = config.LLM_PROVIDER.lower()
        if provider == "ollama":
            return Ollama(
                base_url=config.OLLAMA_BASE_URL,
                model=config.LLM_MODEL_NAME_OLLAMA,
                temperature=getattr(config, "LLM_TEMPERATURE", 0.2),
                top_p=getattr(config, "LLM_TOP_P", 0.9),
                top_k=getattr(config, "LLM_TOP_K", 40),
            )
        elif provider == "huggingface":
            model_id = config.LLM_MODEL
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                top_k=getattr(config, "HF_TOP_K", 30),
                top_p=getattr(config, "HF_TOP_P", 0.95),
                temperature=getattr(config, "HF_TEMPERATURE", 0.7),
            )
            return HuggingFacePipeline(pipeline=pipe)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def load_prompt_template(self):
        tpl = (
            "Bạn là trợ lý RAG, phải dựa 100 % vào NGỮ CẢNH bên dưới.\n"
            "• Nếu NGỮ CẢNH KHÔNG chứa đáp án, trả lời: "
            "\"Không tìm thấy thông tin trong tài liệu.\".\n"
            "• Nếu có đáp án, trả lời đủ thông tin, đưa cả nguồn vào, chính xác, tiếng Việt;  "
            " suy luận theo số liệu , không dược bịa đặt\n"
            "• Cuối cùng in dòng: \"Nguồn: {source}\".\n"
            "\n"
            "NGỮ CẢNH:\n{context}\n"
            "Câu hỏi: {question}\n"
            "Trả lời:"
        )
        return PromptTemplate(template=tpl, input_variables=["context", "question", "source"])

    def _retrieve_chunks(self, query_vector, doc_type: str, k: int = 3):
        filt = models.Filter(must=[
            models.FieldCondition(
                key="metadata.document_type",
                match=models.MatchValue(value=doc_type)
            )
        ])
        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filt,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            Document(
                page_content=h.payload.get("content", ""),
                metadata=h.payload.get("metadata", {}) | {"score": h.score},
            )
            for h in hits
        ]

    def _rerank(self, query: str, docs: list[Document], top_n: int = 3):
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_n]]

    def rag(self, source, question):
        # 0. Nếu lần đầu hoặc admin đã đổi collection, reload retriever
        active = get_active_collection() or config.QDRANT_COLLECTION_NAME
        if self.retriever is None or self.collection_name != active:
            self.retriever = self.load_retriever(source, self.embeddings)

        # 1. Embed câu hỏi
        t0 = time.time()
        qvec = self.embeddings.embed_query(question)
        print(f"[METRIC] Embed-query time: {time.time() - t0:.4f} s")

        # 2. Lấy chunks từ Qdrant
        t1 = time.time()
        regs   = self._retrieve_chunks(qvec, "Regulation",    5)
        faqs   = self._retrieve_chunks(qvec, "FAQ",           3)
        issues = self._retrieve_chunks(qvec, "RelatedIssue",  3)
        print(f"[METRIC] Retrieve time: {time.time() - t1:.4f} s")

        # 3. Rerank top 3
        raw_docs = regs + faqs + issues
        t2 = time.time()
        top_docs = self._rerank(question, raw_docs, top_n=3)
        print(f"[METRIC] Rerank time: {time.time() - t2:.4f} s")

        # 4. Tạo prompt
        context = "\n\n".join(d.page_content for d in top_docs)
        sources = ", ".join(sorted({d.metadata.get("source_file", "?") for d in top_docs}))
        prompt = self.prompt.format(context=context, question=question, source=sources)

        # 5. Inference
        t3 = time.time()
        answer = self.llm(prompt).strip()
        print(f"[METRIC] LLM inference time: {time.time() - t3:.4f} s")

        return {"result": answer, "source_documents": raw_docs}
