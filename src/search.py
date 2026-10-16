import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

REQUIRED_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_CHAT_MODEL",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
)


def get_vector_store():
    return PGVector(
        embeddings=OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL")),
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )


def search_prompt(question=None):
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        print(f"Variáveis de ambiente ausentes: {', '.join(missing)}")
        return None

    try:
        store = get_vector_store()
        llm = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL"))
    except Exception as e:
        print(f"Erro ao inicializar a busca: {e}")
        return None

    def retrieve_context(pergunta):
        # Vetoriza a pergunta e traz os 10 chunks mais próximos.
        results = store.similarity_search_with_score(pergunta, k=10)
        return "\n\n".join(doc.page_content.strip() for doc, _score in results)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {
            "contexto": RunnableLambda(retrieve_context),
            "pergunta": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    if question is not None:
        return chain.invoke(question)

    return chain