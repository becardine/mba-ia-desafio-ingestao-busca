# Ingestão e Busca Semântica com LangChain e Postgres

Software em Python que:

- **Ingestão:** lê um PDF, divide em chunks, gera embeddings e salva no PostgreSQL com pgVector.
- **Busca:** CLI de chat que responde perguntas usando **apenas** o conteúdo do PDF.

## Tecnologias

- Python 3.11+
- LangChain (`langchain-openai`, `langchain-postgres`, `langchain-community`)
- PostgreSQL 17 + pgVector (via Docker Compose)
- OpenAI
  - Embeddings: `text-embedding-3-small` (1536 dimensões)
  - LLM: `gpt-6-luna` (modelo mais leve e barato da OpenAI hoje)

## Estrutura

```
├── docker-compose.yml    # Postgres + pgVector
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Ingestão do PDF
│   ├── search.py         # Busca vetorial + prompt + LLM
│   ├── chat.py           # CLI de interação
├── document.pdf          # PDF para ingestão
└── README.md
```

## Pré-requisitos

- Python 3.11 ou superior
- Docker e Docker Compose
- API Key da OpenAI

## Como executar

### 1. Ambiente virtual e dependências

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Variáveis de ambiente

Copie o template e preencha sua chave da OpenAI:

```bash
cp .env.example .env
```

| Variável | Descrição | Padrão |
| --- | --- | --- |
| `OPENAI_API_KEY` | Sua API Key da OpenAI | — |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings | `text-embedding-3-small` |
| `OPENAI_CHAT_MODEL` | Modelo que gera a resposta | `gpt-6-luna` |
| `DATABASE_URL` | Conexão com o Postgres (driver psycopg 3) | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector | `documento_pdf` |
| `PDF_PATH` | Caminho do PDF a ser ingerido | `document.pdf` |

### 3. Subir o banco de dados

```bash
docker compose up -d
```

O serviço `bootstrap_vector_ext` cria a extensão `vector` assim que o Postgres fica saudável.

### 4. Executar a ingestão do PDF

```bash
python src/ingest.py
```

O PDF é dividido em chunks de **1000 caracteres com overlap de 150**. Cada chunk vira um embedding e é salvo no pgVector. Os IDs são determinísticos (`doc-0`, `doc-1`, ...), então rodar a ingestão de novo atualiza os registros em vez de duplicá-los.

### 5. Rodar o chat

```bash
python src/chat.py
```

Para cada pergunta, o chat:

1. Vetoriza a pergunta.
2. Busca os 10 chunks mais relevantes (`similarity_search_with_score`, `k=10`).
3. Monta o prompt com o contexto e chama a LLM.
4. Mostra a resposta.

Digite `sair` (ou `Ctrl+C`) para encerrar.

## Exemplo de uso

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: R$ 10.000.000,00

---

PERGUNTA: Em que ano foi fundada a Alfa Energia S.A.?
RESPOSTA: A Alfa Energia S.A. foi fundada em 1972.

---

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

---

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

---
```

## Porta 5432 já em uso

Se outro Postgres já estiver usando a porta 5432, o `docker compose up -d` falha com `port is already allocated`. Nesse caso, troque o mapeamento no `docker-compose.yml` (ex.: `"5433:5432"`) e ajuste a porta em `DATABASE_URL` no `.env`.

## Trocar o modelo de embeddings

A tabela de vetores é criada na primeira ingestão com a dimensão do modelo escolhido. Se você trocar `OPENAI_EMBEDDING_MODEL`, apague os dados antigos e refaça a ingestão:

```bash
docker compose down -v
docker compose up -d
python src/ingest.py
```
