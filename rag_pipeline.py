"""
LABORATÓRIO 09: Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoders)
Pipeline completo de RAG de nível de produção com manuais médicos simulados.
"""

import os
import numpy as np
from openai import OpenAI
import faiss
from sentence_transformers import CrossEncoder

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# ==============================================================================
# BASE DE DADOS SIMULADA - 20 fragmentos de manuais médicos
# ==============================================================================
MEDICAL_CORPUS = [
    # Neurologia
    "Cefaléia pulsátil e fotofobia são sintomas clássicos de enxaqueca (migrânea). "
    "O paciente frequentemente relata dor unilateral de intensidade moderada a severa, "
    "associada a náuseas, vômitos e fonofobia. O tratamento agudo inclui triptanos e AINEs.",

    "A hipertensão intracraniana idiopática (pseudotumor cerebri) manifesta-se com cefaléia "
    "crônica difusa, visão turva, zumbido pulsátil e papiledema ao exame de fundo de olho. "
    "A punção lombar revela pressão de abertura elevada (>25 cmH2O).",

    "Cefaléia em salvas (cluster headache) caracteriza-se por dor orbitária unilateral "
    "excruciante de curta duração (15-180 min), acompanhada de lacrimejamento, "
    "rinorreia e síndrome de Horner ipsilateral.",

    "A meningite bacteriana apresenta tríade clássica: febre alta, rigidez de nuca e "
    "alteração do nível de consciência. Sinais de Kernig e Brudzinski positivos reforçam "
    "a hipótese. É emergência médica — iniciar antibioticoterapia imediatamente.",

    "Acidente Vascular Cerebral Isquêmico (AVCI): déficit neurológico focal de início súbito. "
    "Utilizar escala NIHSS para estadiamento. Janela terapêutica para trombólise com rtPA "
    "é de até 4,5 horas do início dos sintomas. TC de crânio sem contraste é mandatória.",

    # Cardiologia
    "Infarto Agudo do Miocárdio (IAM) com supradesnivelamento de ST (IAMCSST): dor "
    "precordial opressiva irradiada para membro superior esquerdo, mandíbula e/ou dorso. "
    "ECG mostra supradesnivelamento ≥1mm em ≥2 derivações contíguas. Reperfusão imediata.",

    "Insuficiência Cardíaca Congestiva (ICC): dispneia progressiva, ortopneia, edema de "
    "membros inferiores e cardiomegalia na radiografia de tórax. BNP elevado confirma "
    "diagnóstico. Tratamento: diuréticos, IECA/BRA e betabloqueadores.",

    "Fibrilação Atrial (FA): ritmo irregularmente irregular, ausência de onda P no ECG, "
    "resposta ventricular variável. Principal causa de AVC cardioembólico. Anticoagulação "
    "com warfarina (RNI 2-3) ou NOACs para CHA2DS2-VASc ≥2.",

    # Pneumologia
    "Pneumonia Adquirida na Comunidade (PAC): febre, tosse produtiva, dispneia e dor "
    "pleurítica. Radiografia mostra consolidação lobar. Escore PSI/PORT e CURB-65 "
    "orientam local de tratamento (ambulatorial vs hospitalar vs UTI).",

    "Asma Brônquica: broncoespasmo reversível com dispneia, sibilância e tosse "
    "predominantemente noturna. Espirometria com padrão obstrutivo e resposta positiva "
    "ao broncodilatador (aumento ≥12% e 200mL no VEF1). Tratamento: beta-2 agonista.",

    # Gastroenterologia
    "Úlcera Péptica Duodenal: dor epigástrica em queimação que melhora com alimentação "
    "e antiácidos. Associada a infecção por H. pylori em 90-95% dos casos. "
    "Endoscopia digestiva alta é o exame de escolha para diagnóstico.",

    "Hepatite B Crônica: HBsAg positivo por >6 meses. Monitorar com HBeAg, HBV-DNA "
    "e ALT sérica. Indicação de tratamento: HBV-DNA >2000 UI/mL e/ou ALT elevada. "
    "Antivirais: tenofovir ou entecavir são a primeira linha.",

    # Endocrinologia
    "Diabetes Mellitus Tipo 1: destruição autoimune de células beta pancreáticas. "
    "Cetoacidose diabética (CAD) é complicação grave: glicemia >250 mg/dL, pH <7,3, "
    "bicarbonato <18 mEq/L e cetonemia. Tratamento: hidratação, insulina IV e eletrólitos.",

    "Hipotireoidismo Primário: fadiga, ganho de peso, constipação, intolerância ao frio "
    "e bradicardia. TSH elevado e T4 livre baixo confirmam diagnóstico. "
    "Tratamento: levotiroxina oral em dose única diária, em jejum.",

    # Reumatologia
    "Lúpus Eritematoso Sistêmico (LES): doença autoimune multissistêmica. Critérios ACR: "
    "rash malar (asa de borboleta), fotossensibilidade, úlceras orais, artrite, serosite, "
    "nefrite lúpica. FAN positivo em >95% dos casos. ANA anti-dsDNA é específico.",

    # Infectologia
    "Sepse: disfunção orgânica ameaçadora à vida causada por resposta desregulada à infecção. "
    "Critérios SOFA ≥2 pontos. Bundle de 1 hora: hemoculturas, lactato, ATB de largo espectro "
    "e reposição volêmica. Choque séptico: vasopressor para PAM ≥65mmHg.",

    "HIV/AIDS: contagem de CD4 <200 cél/mm³ define AIDS. Profilaxia para pneumocistose "
    "(Pneumocystis jirovecii) com sulfametoxazol-trimetoprima. TARV: esquema baseado em "
    "2 ITRN + 1 INSTI (ex: TDF + 3TC + DTG) é recomendação padrão atual.",

    # Nefrologia
    "Insuficiência Renal Aguda (IRA): aumento de creatinina ≥0,3 mg/dL em 48h ou ≥1,5x "
    "o valor basal em 7 dias. Classificação KDIGO: pré-renal, intrínseca e pós-renal. "
    "Monitorar débito urinário (<0,5 mL/kg/h por >6h indica IRA).",

    # Dermatologia
    "Melanoma Maligno: lesão pigmentada com critérios ABCDE (Assimetria, Borda irregular, "
    "Cor variada, Diâmetro >6mm, Evolução). Biósia excisional com margens adequadas é "
    "obrigatória. Espessura de Breslow é o principal fator prognóstico.",

    # Ortopedia
    "Fratura de Quadril (colo femoral): mais comum em idosos com osteoporose após queda. "
    "Dor inguinal, membro inferior encurtado e em rotação externa. Radiografia confirma. "
    "Tratamento cirúrgico precoce (artroplastia ou osteossíntese) reduz mortalidade.",
]

# ==============================================================================
# PASSO 1: CONSTRUÇÃO E INDEXAÇÃO DO GRAFO HNSW
# ==============================================================================

def get_embeddings(texts: list[str]) -> np.ndarray:
    """Gera embeddings via OpenAI text-embedding-3-small."""
    print(f"[PASSO 1] Gerando embeddings para {len(texts)} documentos...")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    print(f"         Shape dos vetores: {vectors.shape}")
    return vectors


def build_hnsw_index(vectors: np.ndarray) -> faiss.IndexHNSWFlat:
    """
    Constrói índice HNSW com FAISS.

    Hiperparâmetros:
    - M=32: número de vizinhos bidirecionais por camada. Maior M → mais RAM e
      maior recall, menor M → menos RAM e menor recall.
    - ef_construction=200: tamanho da lista de candidatos durante a construção.
      Maior valor → índice de melhor qualidade, porém mais lento de construir.
    """
    print("[PASSO 1] Construindo índice HNSW (M=32, ef_construction=200)...")
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)          # M = 32
    index.hnsw.efConstruction = 200                          # ef_construction = 200
    index.hnsw.efSearch = 64                                 # ef_search em tempo de busca

    # Normalizar vetores para Similaridade de Cosseno
    faiss.normalize_L2(vectors)
    index.add(vectors)
    print(f"         Total de vetores indexados: {index.ntotal}")
    return index


# ==============================================================================
# PASSO 2: QUERY TRANSFORMATION (HyDE - Hypothetical Document Embeddings)
# ==============================================================================

def hyde_transform(query: str) -> tuple[str, np.ndarray]:
    """
    HyDE: pede ao LLM que 'aluucine' um documento técnico ideal para a query,
    depois vetoriza esse documento hipotético como âncora geométrica.
    """
    print(f"\n[PASSO 2] Query original: '{query}'")
    print("[PASSO 2] Gerando documento hipotético via LLM (HyDE)...")

    prompt = (
        "Você é um especialista em medicina clínica. "
        "Um paciente descreveu o seguinte sintoma de forma coloquial:\n\n"
        f"'{query}'\n\n"
        "Escreva um parágrafo técnico, no estilo de um manual médico, "
        "descrevendo o diagnóstico diferencial e os achados clínicos mais prováveis "
        "para esses sintomas. Use terminologia médica precisa. "
        "Responda SOMENTE com o parágrafo técnico, sem introdução ou conclusão."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )

    hypothetical_doc = response.choices[0].message.content.strip()
    print(f"[PASSO 2] Documento hipotético gerado:\n         \"{hypothetical_doc[:200]}...\"")

    # Vetorizar o documento hipotético
    emb_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[hypothetical_doc]
    )
    hyde_vector = np.array([emb_response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(hyde_vector)
    print("[PASSO 2] Documento hipotético vetorizado com sucesso.")
    return hypothetical_doc, hyde_vector


# ==============================================================================
# PASSO 3: BUSCA RÁPIDA VIA BI-ENCODER (HNSW)
# ==============================================================================

def retrieve_top_k(
    index: faiss.IndexHNSWFlat,
    corpus: list[str],
    query_vector: np.ndarray,
    k: int = 10
) -> list[tuple[str, float]]:
    """Busca os Top-K documentos via Similaridade de Cosseno no índice HNSW."""
    print(f"\n[PASSO 3] Buscando Top-{k} no índice HNSW (funil largo)...")

    distances, indices = index.search(query_vector, k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        doc = corpus[idx]
        results.append((doc, float(dist)))
        print(f"         [{rank:02d}] Score={dist:.4f} | {doc[:80]}...")

    return results


# ==============================================================================
# PASSO 4: FILTRO FINO COM CROSS-ENCODER (Re-ranking)
# ==============================================================================

def rerank_with_cross_encoder(
    original_query: str,
    candidates: list[tuple[str, float]],
    top_n: int = 3
) -> list[tuple[str, float]]:
    """
    Re-ranking com Cross-Encoder: avalia cada par (query, documento) com atenção
    profunda, produzindo um score de relevância mais preciso.
    """
    print(f"\n[PASSO 4] Re-ranking com Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(original_query, doc) for doc, _ in candidates]
    scores = cross_encoder.predict(pairs)

    # Associar scores e ordenar do maior para o menor
    ranked = sorted(
        zip([doc for doc, _ in candidates], scores),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n{'='*70}")
    print(f"  TOP-{top_n} DOCUMENTOS FINAIS (após Cross-Encoder)")
    print(f"{'='*70}")
    for rank, (doc, score) in enumerate(ranked[:top_n], start=1):
        print(f"\n  [{rank}] Score Cross-Encoder: {score:.4f}")
        print(f"  {doc}")
    print(f"{'='*70}\n")

    return ranked[:top_n]


# ==============================================================================
# PIPELINE COMPLETO
# ==============================================================================

def run_rag_pipeline(query: str):
    """Executa o pipeline RAG completo: HNSW + HyDE + Cross-Encoder."""
    print("\n" + "="*70)
    print("  INICIANDO PIPELINE RAG AVANÇADO")
    print("="*70)

    # Passo 1 — Indexação
    corpus_vectors = get_embeddings(MEDICAL_CORPUS)
    hnsw_index = build_hnsw_index(corpus_vectors)

    # Passo 2 — HyDE
    _, hyde_vector = hyde_transform(query)

    # Passo 3 — Recuperação (Bi-Encoder)
    top10 = retrieve_top_k(hnsw_index, MEDICAL_CORPUS, hyde_vector, k=10)

    # Passo 4 — Re-ranking (Cross-Encoder)
    top3 = rerank_with_cross_encoder(query, top10, top_n=3)

    print("Pipeline concluído. Os 3 documentos acima seriam injetados no contexto do LLM gerador.")
    return top3


# ==============================================================================
# PONTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    # Query coloquial de exemplo — o paciente não usa jargão médico
    USER_QUERY = "dor de cabeça latejante e luz incomodando muito, parece que vai explodir"
    run_rag_pipeline(USER_QUERY)
