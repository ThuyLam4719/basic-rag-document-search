from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()
loader = DirectoryLoader(
    path="./docs",
    glob="**/*.pdf",
    loader_cls=UnstructuredFileLoader,
    show_progress=True,
    use_multithreading=True,
)

embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

vectorstore = Chroma(
    collection_name="pdf_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

if not vectorstore.get()["ids"]:
    docs = loader.load()

    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_amount=0.85
    )
    splits = text_splitter.split_documents(docs)
    vectorstore.add_documents(splits)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)


def format_docs(retrieved_docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in retrieved_docs
    )


template = """You are a helpful RAG chatbot that answers using only the retrieved context.
Instructions:
- Answer in clear, natural English.
- Do not make up facts or use outside knowledge.
- If the context is insufficient, say so clearly.
- When possible, cite the source document.
- Keep the answer concise.

Retrieved context:
{context}

User question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOllama(model="qwen3:4b", temperature=0)

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    user_input = input("Question: ").strip()
    if user_input.lower() == "q":
        print("Exiting...")
        break
    answer = rag_chain.invoke(user_input)
    print(answer)
