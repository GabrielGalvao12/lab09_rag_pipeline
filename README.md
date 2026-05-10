# Laboratório 09 — Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoders)

> **Instituto iCEV | Disciplina: Engenharia de IA**

Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Gabriel Galvão Cardoso.

## Visão Geral

Pipeline de **Retrieval-Augmented Generation (RAG) de nível de produção** para busca em manuais médicos privados. O sistema transforma queries coloquiais de pacientes em buscas técnicas precisas usando três tecnologias em sequência:

```
Query coloquial  →  HyDE  →  HNSW (Top-10)  →  Cross-Encoder (Top-3)  →  Contexto LLM
```

## Estrutura do Projeto

```
lab09_rag/
├── rag_pipeline.py          # Script Python principal (execução local / VS Code)
├── lab09_rag_pipeline.ipynb # Notebook para Google Colab
├── requirements.txt         # Dependências do projeto
└── README.md                # Este arquivo
```

## Como Executar

### Opção A — Google Colab (recomendado para testes)

1. Faça upload do arquivo lab09_rag_pipeline.ipynb no Google Colab
2. Execute a célula de instalação (!pip install ...)
3. Insira sua chave de API da OpenAI na célula de configuração
4. Execute todas as células em ordem (Runtime → Run all)

### Opção B — Local / VS Code

```bash
# 1. Clone o repositório
git clone https://github.com/GabrielGalvao12/lab09_rag_pipeline.git

# 2. Entre na pasta do projeto
cd LABORATORIO_09

# 3. Crie e ative o ambiente virtual
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# 4. Instale as dependências
pip install -r requirements.txt

# 5. Configure a chave de API
# Linux/macOS
export OPENAI_API_KEY="sk-..."

# Windows
set OPENAI_API_KEY=sk-...

# 6. Execute o pipeline
python rag_pipeline.py
```

## 🔬 Tarefa Analítica — Passo 1

### Como os hiperparâmetros `M` e `ef_construction` do HNSW afetam o consumo de memória RAM em relação ao KNN exato?

#### KNN Exato — Baseline de Memória

Na busca **K-Nearest Neighbors exata**, para N documentos com vetores de dimensão D (float32 = 4 bytes), o consumo de RAM é calculado pela fórmula direta:

```
RAM_KNN = N × D × 4 bytes
```

Para nosso corpus de 20 documentos com embeddings de dimensão 1536:

```
RAM_KNN = 20 × 1536 × 4 = ~120 KB
```

Parece pequeno, mas em escala de produção (ex: 10 milhões de documentos):

```
RAM_KNN = 10.000.000 × 1536 × 4 ≈ 61,4 GB
```

Além do espaço de armazenamento, o KNN exato **precisa comparar a query com TODOS os N vetores** a cada busca (complexidade O(N·D)), o que torna inviável em escala.

#### HNSW — Estrutura de Grafo Hierárquico

O HNSW adiciona uma estrutura de **grafo multicamada** sobre os vetores. Cada nó (documento) armazena referências para seus M vizinhos mais próximos. O custo extra de memória por documento é:

```
Overhead_por_nó ≈ M × camadas × tamanho_do_ponteiro
                ≈ M × log₂(N) × 8 bytes   (ponteiros de 64 bits)
```

Para M=32, N=10 milhões (≈23 camadas):

```
Overhead_total ≈ 32 × 23 × 8 × 10.000.000 ≈ 58,7 GB de overhead de grafo
RAM_HNSW_total ≈ 61,4 GB (vetores) + 58,7 GB (grafo) ≈ 120 GB
```

À primeira vista parece pior, mas o ganho está na **velocidade de busca**:

| Método   | Complexidade de Busca | RAM (10M docs, D=1536) | Recall@10 |
|----------|-----------------------|-------------------------|-----------|
| KNN Exato| O(N·D) — linear       | ~61 GB                  | 100%      |
| HNSW     | O(log N) — logarítmico| ~120 GB (com M=32)      | ~97-99%   |

#### Impacto do Hiperparâmetro `M`

`M` controla o número de **arestas bidirecionais** que cada nó mantém no grafo:

- **M baixo (ex: M=4):** Menos RAM, busca mais rápida, porém recall menor — o grafo fica "ralo" e a busca pode ficar presa em regiões subótimas.
- **M alto (ex: M=64):** Mais RAM, recall mais alto — o grafo é denso e a busca encontra vizinhos mais precisos.
- **Regra prática:** M entre 16 e 64 oferece o melhor equilíbrio. Valores acima de 64 raramente justificam o custo.

```
RAM_grafo ∝ M × N    (crescimento linear com M)
```

#### Impacto do Hiperparâmetro `ef_construction`

`ef_construction` define o tamanho da **lista de candidatos** mantida durante a inserção de cada nó. Ele **só afeta a fase de construção do índice**, não a busca em produção:

- **ef_construction baixo (ex: 40):** Índice construído rapidamente, porém com qualidade inferior — vizinhos escolhidos são subótimos.
- **ef_construction alto (ex: 400):** Construção lenta, mas o grafo resultante tem maior qualidade e recall.
- **Sobre RAM:** `ef_construction` consome RAM temporariamente apenas durante a construção (a lista de candidatos vive em memória de trabalho). **Não aumenta o tamanho permanente do índice em disco/RAM.**

```
RAM_temp_durante_construção ∝ ef_construction × D × 4 bytes
```

#### Resumo Comparativo

| Parâmetro         | Efeito na RAM (produção)|Recall Effect|Efeito na Velocidade de Busca |
|-------------------|-------------------------|-------------|------------------------------|
| `M` ↑             |↑ Aumenta linearmente    | ↑ Melhora   | ↓ Ligeiramente mais lento    |
| `ef_construction`↑|Neutro (só na construção)| ↑ Melhora   | Neutro                       |
| `ef_search` ↑     |Neutro                   | ↑ Melhora   |↓ Mais lento                  |

**Conclusão:** Para servidores com RAM limitada, o principal "vilão" é o parâmetro `M`. Recomenda-se M=32 como ponto de equilíbrio padrão para a maioria das aplicações de produção. O `ef_construction` deve ser maximizado durante a construção offline (quando há tempo disponível), pois não impacta o custo de RAM em produção.

---

## Arquitetura do Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    QUERY DO USUÁRIO                     │
│     "dor de cabeça latejante e luz incomodando"         │
└──────────────────────────┬──────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │    HyDE     │  LLM gera documento técnico hipotético
                    │  (Passo 2)  │  "cefaléia pulsátil, fotofobia..."
                    └──────┬──────┘
                           │ vetor do doc. hipotético
                    ┌──────▼──────┐
                    │    HNSW     │  Busca aproximada O(log N)
                    │  (Passo 3)  │  → Top-10 candidatos
                    └──────┬──────┘
                           │ 10 documentos
                    ┌──────▼──────┐
                    │Cross-Encoder│  Atenção profunda par-a-par
                    │  (Passo 4)  │  → Top-3 documentos finais
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  CONTEXTO   │  Injetado no LLM gerador
                    │    LLM      │
                    └─────────────┘
```


## Tecnologias Utilizadas

| Biblioteca              | Papel no Pipeline                                            |
|-------------------------|--------------------------------------------------------------|
| `openai`                | Embeddings (`text-embedding-3-small`) e HyDE (`gpt-4o-mini`) |
| `faiss-cpu`             | Índice HNSW para busca vetorial aproximada                   |
| `sentence-transformers` | Cross-Encoder para re-ranking preciso                        |
| `numpy`                 | Operações com vetores e normalização L2                      |

## Dependências

Veja o arquivo `requirements.txt`:

```
openai>=1.30.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
numpy>=1.26.0
```

### Política de Uso de IA

Partes geradas/complementadas com IA, revisadas por Gabriel Galvão Cardoso.

Ferramentas de IA generativa foram utilizadas como IA generativa Claude (Anthropic),suporte na geração de templates de código e estrutura inicial dos scripts. Todo o conteúdo foi
revisado criticamente e validado antes da submissão, em conformidade com
a Regra de Ouro do contrato pedagógico do iCEV.

