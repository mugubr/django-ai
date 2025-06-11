<a id="readme-top"></a>
# AI ssistente

### Tecnologias utilizadas
<div align="center" style="display: inline_block"><br>

  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/javascript/javascript-original.svg" width="50" height="50" alt="Javascript"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/tailwindcss/tailwindcss-original-wordmark.svg" width="50" height="50" alt="TailwindCSS" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original-wordmark.svg" width="50" height="50" alt="Python"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pytest/pytest-original-wordmark.svg" width="50" height="50" alt="Pytest"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/django/django-plain.svg" width="50" height="50" alt="Django"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/sqlalchemy/sqlalchemy-original-wordmark.svg" width="50" height="50" alt="SQLAlchemy"/>
</div>

## Executando o projeto

Como executar o projeto localmente

### Pré-requisitos

* python 3.12+
* requirements.txt

```
aiohappyeyeballs==2.6.1
aiohttp==3.12.2
aiosignal==1.3.2
annotated-types==0.7.0
anyio==4.9.0
attrs==25.3.0
certifi==2025.4.26
charset-normalizer==3.4.2
contourpy==1.3.2
cycler==0.12.1
dataclasses-json==0.6.7
distro==1.9.0
faiss-cpu==1.11.0
fonttools==4.58.1
frozenlist==1.6.0
greenlet==3.2.2
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
httpx-sse==0.4.0
idna==3.10
jiter==0.10.0
joblib==1.5.1
jsonpatch==1.33
jsonpointer==3.0.0
kiwisolver==1.4.8
langchain==0.3.25
langchain-community==0.3.24
langchain-core==0.3.62
langchain-openai==0.3.18
langchain-text-splitters==0.3.8
langchain-xai==0.2.4
langsmith==0.3.42
marshmallow==3.26.1
matplotlib==3.10.3
multidict==6.4.4
mypy_extensions==1.1.0
numpy==2.2.6
openai==1.82.0
orjson==3.10.18
packaging==24.2
pillow==11.2.1
propcache==0.3.1
pydantic==2.11.5
pydantic-settings==2.9.1
pydantic_core==2.33.2
pyparsing==3.2.3
pypdf==5.5.0
python-dateutil==2.9.0.post0
python-dotenv==1.1.0
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
requests-toolbelt==1.0.0
scikit-learn==1.6.1
scipy==1.15.3
six==1.17.0
sniffio==1.3.1
SQLAlchemy==2.0.41
tenacity==9.1.2
threadpoolctl==3.6.0
tiktoken==0.9.0
tqdm==4.67.1
typing-inspect==0.9.0
typing-inspection==0.4.1
typing_extensions==4.13.2
urllib3==2.4.0
yarl==1.20.0
zstandard==0.23.0
```
### Executando

1. Clone o repositório
   ```sh
   git clone https://github.com/mugubr/django-ai
   ```
2. Na raiz do diretório , crie um arquivo ```.env``` com as seguintes variáveis de ambiente
   ```sh
    OPENAI_API_KEY=
   ```
3. Crie e entre em um ambiente virtual (venv)
   ```sh
   python -m venv venv

   venv/scripts/activate
   ```
4. Instale as dependências
   ```sh
   pip install -r requirements.txt
   ```
5. Execute as migrações
   ```sh
   python manage.py makemigrations

   python manage.py migrate
   ```
6. Para subir a aplicação (por padrão, ela se encontrará na porta 8000)
   ```sh
   python manage.py runserver
   ```
### OBS:
- Para criar um super-usuário, execute:
   ```sh
   python manage.py createsuperuser <nome>
   ```
- O painel de controle do Django pode ser acessado (por super-usuários) na rota ```/admin```
- Para rodar o cluster do ```django_q```, responsável pelo processamento dos sites, conteúdos escritos e PDFs passados para o modelo de IA, execute:
   ```sh
   python manage.py qcluster
   ```

# Teoria
## RAGs

RAG (Retrieval-Augmented Generation) é uma técnica que combina busca por informações relevantes em uma base de dados com a geração de texto por modelos como o GPT. Ao receber uma pergunta, o sistema recupera documentos relacionados (como PDFs ou artigos) e os envia como contexto para o modelo gerar uma resposta mais precisa. Isso permite respostas atualizadas e confiáveis, mesmo quando o modelo não tem esse conhecimento originalmente.

### OBS:

- O RAG não é um a técnica de treinamento de LLMs, é uma técnica de gerenciamento de contexto

## Chunks

Imagina que você precisa criar um RAG que utiliza a Constituição Federal para auxiliar advogados. Se, para uma pergunta sobre direito do consumidor, enviarmos toda a Constituição, isso fará com que o modelo de IA não consiga processar todas as informações, já que, quanto maior o prompt, menos precisa tende a ser a resposta.
Para isso, utilizamos a técnica de chunks: pegamos um arquivo geral e o quebramos em vários pequenos trechos.
Podemos usar um chunk_size para especificar quantos caracteres teremos por chunk. A Constituição Federal possui 64.488 palavras. Se definirmos um chunk_size como 100, teremos 645 mini arquivos da Constituição.

## Overlap

Mas agora enfrentamos outro problema: ao separar o texto por chunks, pode ser que eles fiquem sem sentido, já que partes importantes da informação podem ser cortadas.
Para isso, usamos o parâmetro chunk_overlap . Ele define quantos caracteres de sobreposição haverá entre um chunk e o próximo.

### OBS:

- Isso é útil para manter o contexto entre pedaços consecutivos.

## Embeddings

Agora que separamos nossos arquivos em chunks, precisamos analisar a pergunta do usuário e decidir quais chunks fazem mais sentido com sua pergunta para usá-los como contexto para a IA.
Para isso, usamos o conceito de Embeddings, que nos permite transformar um texto em um vetor de dados.

**Ex:**

```python
{
"texto": "O que é Python?",
"vetor_parcial": [
-0.007813546806573868,
0.007350319996476173,
0.01180547196418047,
-0.017262011766433716,
0.019986875355243683,
0.026335809379816055,
0.005541691556572914,
0.006291029509156942,
0.0043563758954405785,
-0.018951427191495895
],
},
```

Como temos dados vetoriais, podemos plotá-los em um gráfico. Usando 2D como exemplo:

```python
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from langchain_openai import OpenAIEmbeddings

textos = [
    "Python é uma linguagem de programação",
    "Python serve para programação web",
    "Python é multiplataforma",
    "Gatos são animais fofos",
    "Cachorros são leais",
    "Estou fazendo um risoto",
    "O que é Python?"
]

os.environ["OPENAI_API_KEY"] = ""

embedding_model = OpenAIEmbeddings()
vetores = embedding_model.embed_documents(textos)

vetores_array = np.array(vetores)
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
vetores_2d = tsne.fit_transform(vetores_array)

idx_target = textos.index("O que é Python?")
target_vector = vetores_2d[idx_target].reshape(1, -1)
distances = euclidean_distances(target_vector, vetores_2d).flatten()

closest_indices = distances.argsort()[1:4]

plt.figure(figsize=(12, 7))
for i, texto in enumerate(textos):
    x, y = vetores_2d[i]
    plt.scatter(x, y, marker='o')
    plt.text(x + 0.5, y + 0.5, f"({x:.2f}, {y:.2f}) - {texto}", fontsize=9)

x0, y0 = vetores_2d[idx_target]
plt.scatter(x0, y0, color='red', s=100, label='"O que é Python?"')

for idx in closest_indices:
    x1, y1 = vetores_2d[idx]
    plt.plot([x0, x1], [y0, y1], 'k--', alpha=0.6)

plt.title("Visualização dos Embeddings")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig("embeddings_visualizacao_ligacoes.png")
```

Com os chunks escolhidos agora podemos utiliza-los no prompt de contexto da IA:

```python
messages = [
{"role": "system", "content": f"Você é uma assistente de IA, use o contexto para
responder as perguntas. Contexto: Python é uma linguagem de progamação - Python serve para
programação web, Python é multiplataforma"},
{"role": "user", "content": "O que é Python"}
]
```

### OBS:

- Na técnica de Embedding, buscar os K elementos mais próximos utiliza o algoritmo KNN, que busca os chunks/vetores mais próximos baseado na distância euclidiana.
- Setar o “role” como “system” faz com que haja menos alucinações por parte do LLM
- Os embeddings são Multi-Layer-Perceptrons, redes neurais treinadas para gerar contextos a partir de dados textuais e transformá-los em vetores
- Um banco FAISS é um arquivo binário com padrão específico para guardar dados vetoriais com fácil e rápido acesso

## Implementação
- app.py

```python
import os
import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["OPENAI_API_KEY"] = ""

caminho_pdf = "edital-concurso-nacional-unificado-bloco-2.pdf"
loader = PyPDFLoader(caminho_pdf)
documentos = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documentos)

embeddings = OpenAIEmbeddings()
db_path = "banco_faiss"
if os.path.exists(db_path):
    vectordb = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    vectordb.add_documents(chunks)
else:
    vectordb = FAISS.from_documents(chunks, embeddings)
vectordb.save_local(db_path)  

# RAG + RETRIEVER

retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    retriever=retriever,
    return_source_documents=True
)

pergunta = "O que o PDF fala sobre inteligência artificial?"
resposta = qa.invoke(pergunta)

print("Resposta:")
print(resposta["result"])

print("Fonte:")
for doc in resposta["source_documents"]:
    print(f"- Página: {doc.metadata.get('page', '?')} - Trecho: {doc.page_content}...")

# RAG + CONTEXT

embeddings = OpenAIEmbeddings()
vectordb = FAISS.load_local("banco_faiss", embeddings, allow_dangerous_deserialization=True)

docs = vectordb.similarity_search('O que o PDF fala sobre inteligência artificial?', k=5)

contexto = "\n\n".join([
        f"Material: {doc.page_content}"
        for doc in docs
])

messages = [
        {"role": "system", "content": f"Você é um assistente virtual e deve responder com precissão as perguntas sobre uma empresa.\n\n{contexto}"},
        {"role": "user", "content": 'O que o PDF fala sobre inteligência artificial?'}
    ]

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
)

print(llm.invoke(messages))
```

- ver_faiss.py

```python
import json
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = ""

db_path = "banco_faiss"
db = FAISS.load_local(db_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

faiss_index = db.index
documentos = list(db.docstore._dict.values())

dados = []

for i, doc in enumerate(documentos):
    vetor = faiss_index.reconstruct(i)
    item = {
        "id": i,
        "conteudo": doc.page_content.replace("\n", " ").strip(),
        "vetor_parcial": vetor[:10].tolist()
    }
    dados.append(item)

with open("faiss_exportado.json", "w", encoding="utf-8") as jsonfile:
    json.dump(dados, jsonfile, ensure_ascii=False, indent=2)

print("Arquivo faiss_exportado.json criado com sucesso.")

```
Projeto baseado no projeto realizado no workshop Arcane oferecido pela https://pythonando.com.br/
