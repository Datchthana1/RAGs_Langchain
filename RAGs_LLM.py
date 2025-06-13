from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# --- CONFIGURATION ---
PDF_PATH = r"Medical Knowledge Document.pdf"
VECTOR_DB_PATH = r"faiss_index"
MODEL_NAME = "tiiuae/falcon-rw-1b"  # หรือ "tiiuae/falcon-7b-instruct"

# --- 1. โหลด PDF ---
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# --- 2. แบ่งข้อความ ---
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# --- 3. สร้าง Embedding ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 4. โหลดหรือสร้าง Vector Store ---
try:
    db = FAISS.load_local(VECTOR_DB_PATH, embedding_model)
    print("✅ Loaded FAISS from disk.")
except:
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(VECTOR_DB_PATH)
    print("✅ FAISS created and saved.")

# --- 5. สร้าง Retriever แบบ MMR ---
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 5})

# --- 6. ใช้ HuggingFacePipeline LLM ---
hf_pipe = pipeline(
    "text2text-generation",
    model=MODEL_NAME,
    max_length=2000,
    # ถ้าต้องใช้ token ให้ใส่ตอนสร้าง pipeline เท่านั้น เช่น:
    # use_auth_token='hf_...'
)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# --- 7. สร้าง RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
)

# --- 8. ถามคำถาม ---
query = "What is CARDIOVASCULAR DISEASES?"
result = qa_chain.invoke(query)

# --- 9. แสดงผลลัพธ์ ---
print("\n📌 Answer:")
print(result["result"])

print("\n📄 Sources:")
for i, doc in enumerate(result["source_documents"]):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content)
