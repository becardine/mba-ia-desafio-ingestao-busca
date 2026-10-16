import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

REQUIRED_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
    "PDF_PATH",
)


def check_env():
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Variáveis de ambiente ausentes: {', '.join(missing)}")


def ingest_pdf():
    check_env()

    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(f"PDF não encontrado em: {PDF_PATH}")

    docs = PyPDFLoader(PDF_PATH).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    ).split_documents(docs)

    if not splits:
        raise RuntimeError("Nenhum conteúdo extraído do PDF.")

    # Remove metadados vazios para não poluir o JSONB no banco.
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in splits
    ]

    # IDs determinísticos: reexecutar a ingestão atualiza os chunks em vez de duplicá-los.
    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    store.add_documents(documents=enriched, ids=ids)

    print(f"Ingestão concluída: {len(docs)} páginas, {len(enriched)} chunks salvos.")


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as e:
        print(f"Erro na ingestão: {e}", file=sys.stderr)
        sys.exit(1)
