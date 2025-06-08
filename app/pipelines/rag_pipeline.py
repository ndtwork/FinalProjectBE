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
from langchain.schema import Document  # import tại đầu file cũng được
import torch
from sentence_transformers import CrossEncoder
import os
import time  # <- thêm import time để đo thời gian

class RAGPipelineLoader:
    def __init__(self):
        self.embeddings = self.load_embeddings()
        self.source = None
        self.retriever = None
        self.llm = self.load_model_pipeline()
        self.prompt = self.load_prompt_template()
        self.pipeline = None
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                                     device="cuda" if torch.cuda.is_available() else "cpu")

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
            self.qdrant_client = client
            retriever = Qdrant(
                client=client,
                collection_name=config.QDRANT_COLLECTION_NAME,
                embeddings=embeddings
            ).as_retriever()
        else:
            raise ValueError("Unknown retriever source")

        return retriever

    def load_model_pipeline(self, max_new_tokens: int = 100):
        """
        Khởi tạo LLM cho pipeline.
        • Nếu LLM_PROVIDER = "ollama": đọc base_url, model, temperature, top_p, top_k
          từ file .env (được nạp sẵn trong rag_config). Tham số nào không có → dùng mặc định.
        • Nếu LLM_PROVIDER = "huggingface": giữ nguyên như trước.
        """
        llm_provider = config.LLM_PROVIDER.lower()

        # ---------- OLLAMA LOCAL ----------
        if llm_provider == "ollama":
            return Ollama(
                base_url=config.OLLAMA_BASE_URL,  # ví dụ http://localhost:11434
                model=config.LLM_MODEL_NAME_OLLAMA,  # ví dụ llama3.2:3b-instruct-q8_0
                temperature=getattr(config, "LLM_TEMPERATURE", 0.2),  # mặc định 0.2
                top_p=getattr(config, "LLM_TOP_P", 0.9),  # mặc định 0.9
                top_k=getattr(config, "LLM_TOP_K", 40),  # mặc định 40
                # Bạn có thể bổ sung các tham số Ollama khác nếu cần:
                # num_ctx   = getattr(config, "LLM_NUM_CTX",   8192),
                # repeat_penalty = getattr(config, "LLM_REPEAT_PENALTY", 1.1),
            )

        # ---------- HUGGINGFACE PIPELINE ----------
        elif llm_provider == "huggingface":
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

        # ---------- KHÔNG HỖ TRỢ ----------
        else:
            raise ValueError(
                f"Unsupported LLM provider: {llm_provider}. "
                "Choose 'ollama' or 'huggingface'."
            )

    def load_prompt_template(self):
        template = (
            # ── Chỉ dẫn rõ ràng cho LLM ──────────────────────────────
            "Bạn là trợ lý RAG, phải dựa 100 % vào NGỮ CẢNH bên dưới.\n"
            "• Nếu NGỮ CẢNH KHÔNG chứa đáp án, trả lời đúng một câu: "
            "\"Không tìm thấy thông tin trong tài liệu.\".\n"
            "• Nếu có đáp án, trả lời ngắn gọn, chính xác, tiếng Việt; "
            "không thêm suy đoán.\n"
            "• Cuối cùng ghi dòng: \"Nguồn: {source}\" (danh sách file, cách nhau bằng dấu phẩy).\n"
            "\n"
            "NGỮ CẢNH:\n{context}\n"
            "Câu hỏi: {question}\n"
            "Trả lời:"
        )
        # ➊ thêm biến “source” vào input_variables
        return PromptTemplate(template=template,
                              input_variables=["context", "question", "source"])

    def load_rag_pipeline(self, llm, retriever, prompt):
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        # ——— helper mới: lấy 3 chunk cho 1 loại metadata ———

    def _retrieve_chunks(self, query_vector, doc_type: str, k: int = 3):
        filt = models.Filter(must=[models.FieldCondition(
            key="metadata.document_type",
            match=models.MatchValue(value=doc_type))])
        hits = self.qdrant_client.search(  # self.retriever là Qdrant store
            collection_name=config.QDRANT_COLLECTION_NAME,
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

    # ─── helper mới ───
    def _rerank(self, query: str, docs: list[Document], top_n: int = 3):
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_n]]

    def rag(self, source, question):
        # 0. (cache) khởi tạo retriever một lần duy nhất/nguồn
        if self.pipeline is None or self.source != source:
            self.retriever = self.load_retriever(source, self.embeddings)
            self.source = source  # ghi nhớ “qdrant” để lần sau khỏi tạo lại

        # 1. Embed câu hỏi → vector
        start_embed_query = time.time()  # BẮT ĐẦU đo thời gian embed query
        q_vec = self.embeddings.embed_query(question)
        t_embed_query = time.time() - start_embed_query  # KẾT THÚC đo
        print(f"[METRIC] Embed-query time: {t_embed_query:.4f} s")  # IN KẾT QUẢ ĐO


        # 2. Lấy 3 chunk cho mỗi loại document_type
        # 2. Lấy chunk (Regulation 3, FAQ 1, RelatedIssue 1)
        start_retrieve = time.time()  # BẮT ĐẦU đo thời gian retrieve

        regs = self._retrieve_chunks(q_vec, "Regulation", 5)
        faqs = self._retrieve_chunks(q_vec, "FAQ", 3)
        issues = self._retrieve_chunks(q_vec, "RelatedIssue", 3)

        t_retrieve = time.time() - start_retrieve  # KẾT THÚC đo
        print(f"[METRIC] Retrieve time (Qdrant search): {t_retrieve:.4f} s")

        source_docs = regs + faqs + issues  # tổng cộng 9 chunk

        # --- 3. Rerank 11 chunk xuống 3 ---
        raw_docs = regs + faqs + issues  # 11  chunk

        start_rerank = time.time()  # BẮT ĐẦU đo thời gian rerank
        top_docs = self._rerank(question, raw_docs, top_n=3)
        t_rerank = time.time() - start_rerank  # KẾT THÚC đo
        print(f"[METRIC] Rerank time: {t_rerank:.4f} s")

        # --- 4. Ghép context & nguồn ---
        context = "\n\n".join(d.page_content for d in top_docs)
        source_files = ", ".join(sorted({d.metadata.get("source_file", '?')
                                         for d in top_docs}))
        prompt = self.prompt.format(context=context,
                                    question=question,
                                    source=source_files)

        # 5. Gọi LLM (Ollama local) sinh câu trả lời
        start_inference = time.time()  # BẮT ĐẦU đo thời gian inference
        answer = self.llm(prompt).strip()
        t_inference = time.time() - start_inference  # KẾT THÚC đo
        print(f"[METRIC] LLM inference time: {t_inference:.4f} s")

        # 6. Trả về dict gồm: kết quả & danh sách chunk để bạn debug/hiển thị nguồn
        return {"result": answer, "source_documents": source_docs}
