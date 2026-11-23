import json
import os
import base64
import glob
import unicodedata
import urllib.request
import urllib.parse
from enum import Enum
from typing import List, Literal, Optional
import streamlit as st
from pydantic import BaseModel, Field, model_validator, ValidationError
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
from pypdf import PdfReader
from lector_informes import extraer_texto_archivo
from ficha_clinica_real import renderizar_ficha_real


def normalizar_nombre(texto: str) -> str:
    sin_acentos = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return sin_acentos.lower().replace("_", " ").replace("-", " ").strip()


def limpiar_estado_anterior():
    st.session_state.pop("reporte", None)
    st.session_state.pop("ensayos_externos", None)
    st.session_state.pop("enlaces_externos", None)


def evaluar_semaforo(riesgo_sesgo: str, nivel_evidencia: str) -> tuple[str, str, str]:
    texto = f"{riesgo_sesgo} {nivel_evidencia}".lower()
    if any(k in texto for k in ["alto riesgo", "high risk", "muy bajo", "muy baja", "insuficiente", "critico", "crítico", "jbi bajo"]):
        return "🔴", "Alerta metodológica (Alto riesgo / Certeza muy baja)", "#FEE2E2"
    if any(k in texto for k in ["bajo riesgo", "low risk", "alta", "moderada", "jbi alto", "solido", "sólido"]):
        return "🟢", "Solidez metodológica (Bajo riesgo / Certeza moderada-alta)", "#DCFCE7"
    return "🔵", "Precaución / Evidencia intermedia (Algunas preocupaciones / Piloto)", "#DBEAFE"


# 1. ESQUEMAS DE DATOS Y CORTAFUEGOS CLÍNICO
class PatologiaBase(str, Enum):
    DCA = "Daño Cerebral Adquirido"
    PARKINSON = "Enfermedad de Parkinson"
    ATAXIA = "Ataxia Hereditaria"


class TipoTrastorno(str, Enum):
    DISFAGIA = "Disfagia Neurogénica"
    DISARTRIA = "Disartria"
    AFASIA = "Afasia"
    DISPRAXIA = "Dispraxia / Apraxia del Habla / Orofacial"


class EvaluacionClinica(BaseModel):
    escalas_sugeridas: List[str] = Field(description="Escalas y pruebas instrumentales recomendadas con su justificación clínica.")
    signos_alerta_seguridad: List[str] = Field(description="Signos clínicos críticos detectados.")
    informacion_clinica_no_disponible: List[str] = Field(description="Datos ausentes en la historia del paciente que impiden confirmar diagnósticos o riesgos (ej. ausencia de FEES/VFSS, estado ON/OFF no registrado, sintaxis compleja no evaluada).")
    evidencia_cientifica_insuficiente: List[str] = Field(description="Áreas donde la literatura recuperada tiene certeza baja/muy baja, falta de ensayos clínicos o riesgo de extrapolación entre fenotipos.")
    criterios_derivacion: List[str] = Field(description="Especialidades a las que derivar (ej. Neurología, ORL, Nutrición, Farmacia).")
    prueba_instrumental_previa: bool = Field(description="True solo si el caso indica FEES o VFSS ya realizados.")


class DiagnosticoEstructurado(BaseModel):
    patologia_neuro_base: PatologiaBase
    trastorno_logopedico: TipoTrastorno
    codigo_cie11: str = Field(
        description=(
            "Código CIE-11 exacto según etiología: "
            "'8A03.10' para Ataxia Hereditaria (Ataxia de Friedreich / ataxias cerebelosas hereditarias), "
            "'8A00.0' para Enfermedad de Parkinson, "
            "'8B11 / 8B20' para Daño Cerebral Adquirido de origen vascular (ictus isquémico/hemorrágico), "
            "'NA07' para Daño Cerebral Adquirido de origen traumático (traumatismo craneoencefálico / TCE), "
            "añadiendo el código secundario del trastorno funcional si procede."
        )
    )
    analisis_fisiopatologico: str = Field(description="Explicación detallada de los subsistemas afectados (respiración, fonación, articulación, resonancia, prosodia, deglución o lenguaje).")
    diagnostico_diferencial: str = Field(description="Diferenciación clínica frente a cuadros similares y límites de certeza diagnóstica.")

    @model_validator(mode="after")
    def asegurar_cie11_exacto(self) -> "DiagnosticoEstructurado":
        if self.patologia_neuro_base == PatologiaBase.ATAXIA:
            if "8A03.10" not in self.codigo_cie11:
                self.codigo_cie11 = "8A03.10 (Ataxia hereditaria / Ataxia de Friedreich)"
        elif self.patologia_neuro_base == PatologiaBase.PARKINSON:
            if "8A00" not in self.codigo_cie11:
                self.codigo_cie11 = "8A00.0 (Enfermedad de Parkinson)"
        elif self.patologia_neuro_base == PatologiaBase.DCA:
            if not any(cod in self.codigo_cie11.upper() for cod in ["8B11", "8B20", "8B00", "NA07"]):
                self.codigo_cie11 = "8B11 / 8B20 (DCA vascular) · NA07 (DCA traumático)"
        return self


class IntervencionLogopedica(BaseModel):
    respuesta_dilema_clinico: str = Field(description="Respuesta directa, argumentada y segura frente a la propuesta o trampa planteada en la consulta.")
    objetivos_terapeuticos: List[str] = Field(description="Objetivos funcionales y comunicativos/deglutorios individualizados.")
    estrategias_compensatorias: List[str] = Field(description="Maniobras deglutorias, postura, CAA o ajustes del entorno detallados.")
    reactivacion_neuromuscular: List[str] = Field(description="Procedimientos motores o programas validados con sus parámetros de dosificación y fatiga.")
    adaptacion_dietetica_iddsi: str = Field(description="Pauta detallada sobre viscosidades/texturas IDDSI y condiciones obligatorias previas.")


class CitaRAG(BaseModel):
    id_nodo: str = Field(description="ID exacto del claim (ej. DCA-C001, EP-CLM-002, ATX-CL-007) o RS_ID de la capa de evidencia.")
    tipo_fuente: Literal["Guía Clínica / Manual", "Revisión Sistemática / Estudio Primario"] = Field(
        description="Origen jerárquico de la evidencia."
    )
    fuente_documental: str = Field(description="ID del documento, año y localizador físico PDF o sección de la revisión/estudio.")
    hecho_clinico_factual: str = Field(description="Dato clínico objetivo extraído sin narrativa autoral.")


class EnsayoComparado(BaseModel):
    referencia_estudio: str = Field(description="Autor, año o identificador NCT y tipo de diseño (ECA, Revisión Sistemática).")
    intervenciones_comparadas: str = Field(description="Intervención experimental frente a grupo control o terapia estándar.")
    hallazgos_principales: str = Field(description="Resultados cuantitativos/funcionales principales y nivel de significación o certeza.")


class SintesisEnsayosExternos(BaseModel):
    ensayos: List[EnsayoComparado]


class ReporteClinicoLogopedico(BaseModel):
    diagnostico: DiagnosticoEstructurado
    evaluacion: EvaluacionClinica
    intervencion: IntervencionLogopedica
    evidencia_rag: List[CitaRAG] = Field(min_length=1)

    @model_validator(mode="after")
    def cortafuegos_clinico(self) -> "ReporteClinicoLogopedico":
        plan = " ".join(
            self.intervencion.estrategias_compensatorias +
            self.intervencion.reactivacion_neuromuscular
        ).lower()
        negaciones = ["desaconseja", "no se recomienda", "contraindicado", "evitar", "no utilizar", "no aplicar"]

        if self.diagnostico.patologia_neuro_base == PatologiaBase.ATAXIA and "nmes" in plan:
            if not any(n in plan for n in negaciones):
                raise ValueError("VIOLACIÓN GRADE: Desaconsejado el uso de NMES en Ataxia de Friedreich.")

        if self.diagnostico.patologia_neuro_base == PatologiaBase.DCA and "tdcs" in plan:
            if not any(n in plan for n in negaciones) and "ensayo" not in plan:
                raise ValueError("VIOLACIÓN ESO 2025: La tDCS no debe prescribirse fuera de ensayos clínicos.")

        return self


# ESQUEMA DE EXTRACCIÓN PARA EL MÓDULO DE EVIDENCIA (CAPA 2)
class ChunkEvidenciaCientifica(BaseModel):
    rs_id: str = Field(description="Identificador único del bloque (ej. COCHRANE-2014-ATX-RESUMEN, TRIAL-SCA3-2023-01).")
    seccion_rs: Literal["resumen_conclusiones", "tabla_estudio_primario", "discusion_practica_clinica"]
    prioridad_recuperacion: Literal["alta", "media"]
    dominio_clinico: Literal["Evaluación", "Disartria", "Disfagia Neurogénica", "Afasia", "Dispraxia", "General"]
    subtipo_ataxia: str = Field(default="hereditaria_inespecifica", description="Subtipo específico: friedreich, sca3, sca2, sca6, sca_general o hereditaria_inespecifica.")
    evidence_tier: Literal["Tier_1_SR", "Tier_2_Trial"] = Field(
        default="Tier_1_SR",
        description="Tier_1_SR para Revisiones Sistemáticas/Metaanálisis; Tier_2_Trial para Ensayos Clínicos o Estudios Piloto primarios."
    )
    evidence_status: Literal["insufficient_rct", "contraindicated_or_harmful", "supported"] = Field(
        default="insufficient_rct",
        description="insufficient_rct si faltan ECA pero es plausible/seguro; contraindicated_or_harmful si provoca fatiga extrema o riesgo de aspiración; supported si hay respaldo favorable."
    )
    phenotypes: List[str] = Field(
        default_factory=list,
        description="Fenotipos clínicos ejecutables: disartria_ataxica, disfagia_orofaringea, disfonia, incoordinacion_fonorrespiratoria."
    )
    intervention_intent: List[str] = Field(
        default_factory=list,
        description="Intención terapéutica: compensatoria, rehabilitadora_motora, higienico_dietetica, instrumental."
    )
    intervencion_evaluada: str = Field(description="Descripción detallada de la intervención o exposición (dosis, intensidad, frecuencia, duración).")
    comparadores: str = Field(description="Grupo control, placebo, terapia estándar o ausencia de comparador.")
    desenlaces_primarios_secundarios: str = Field(description="Desenlaces primarios y secundarios medidos en el estudio o revisión.")
    herramientas_evaluacion: str = Field(description="Instrumentos, escalas clínicas y pruebas instrumentales empleadas.")
    momentos_medicion: str = Field(description="Tiempos de evaluación (basal, post-intervención, seguimiento a X semanas/meses).")
    variables_continuas: str = Field(description="Tamaño muestral (n) por grupos, media y desviación estándar (DE), mediana y rango intercuartílico (RIC), error estándar (EE) si aplica.")
    variables_categoricas: str = Field(description="Número de eventos, proporciones/porcentajes, Odds Ratio (OR), Riesgo Relativo (RR) con intervalos de confianza (IC 95%) y valores p.")
    hallazgos_cualitativos_citas: str = Field(description="Categorías temáticas, citas textuales o hallazgos cualitativos relevantes.")
    riesgo_sesgo_calidad: str = Field(description="Valoración del riesgo de sesgo o calidad metodológica según Cochrane RoB 2, ROBINS-I, JBI o GRADE (indicar explícitamente si es Bajo riesgo, Algunas preocupaciones/Moderado o Alto riesgo).")
    nivel_evidencia: str = Field(description="Nivel de certeza global o diseño (ej. Alta, Moderada, Baja, Muy baja, ECA, Serie de casos).")
    limitaciones: str = Field(description="Limitaciones metodológicas principales reportadas.")
    contenido_chunk: str = Field(description="Síntesis clínica completa y autocontenida del bloque para recuperación RAG.")


class ExtraccionDocumentoCientifico(BaseModel):
    titulo_documento: str
    autores_principales: str
    doi: str = Field(description="DOI del documento si aparece en el texto, o cadena vacía ''.")
    chunks: List[ChunkEvidenciaCientifica]


# 2. CARGA DEL ÍNDICE VECTORIAL Y CAPA 2 DE EVIDENCIA
@st.cache_resource
def inicializar_indice():
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    with open("corpus_factual_unificado_rag.json", "r", encoding="utf-8") as f:
        registros = json.load(f)
    docs = [
        Document(
            text=f"Patología: {r['patologia']} | Dominio: {r['dominio_clinico']}\nHecho: {r['dato_factual_objetivo']}\nAlerta: {r['alerta_seguridad']}",
            metadata={"claim_id": r["claim_id"], "doc_id": r["doc_id"], "localizador": r["localizador_pdf"]}
        )
        for r in registros
    ]
    return VectorStoreIndex.from_documents(docs)


@st.cache_data
def detectar_subtipo_consulta(texto: str) -> str:
    t = normalizar_nombre(texto)
    if "friedreich" in t or "frda" in t:
        return "friedreich"
    if "sca3" in t or "machado" in t or "tipo 3" in t:
        return "sca3"
    if "sca2" in t or "tipo 2" in t:
        return "sca2"
    if "sca6" in t or "tipo 6" in t:
        return "sca6"
    return "hereditaria_inespecifica"

def recuperar_evidencia_rs(patologia: str, trastorno: str, notas_clinicas: str = "") -> str:
    registros_rs = cargar_capa_rs()
    if not registros_rs:
        return "No hay revisiones sistemáticas cargadas en la capa de evidencia."

    filtrados = [r for r in registros_rs if r.get("patologia", "").lower() == patologia.lower()]
    subtipo_paciente = detectar_subtipo_consulta(notas_clinicas)

    resumenes = [r for r in filtrados if r.get("seccion_rs") == "resumen_conclusiones"]
    especificos = [
        r for r in filtrados
        if r.get("seccion_rs") != "resumen_conclusiones"
        and (r.get("dominio_clinico", "").lower() in [trastorno.lower(), "evaluación", "general"])
    ]

    # Orden cronológico (más reciente primero para Temporal Decay Override)
    resumenes.sort(key=lambda x: x.get("anio_publicacion", 0), reverse=True)
    especificos.sort(key=lambda x: x.get("anio_publicacion", 0), reverse=True)

    seleccionados = resumenes[:2] + especificos[:6]
    if not seleccionados:
        return "Sin registros específicos en la capa de Revisiones Sistemáticas para esta combinación."

    bloques = []
    alertas_subtipo = []
    bloqueos_seguridad = []

    for r in seleccionados:
        tier = r.get("evidence_tier") or ("Tier_1_SR" if "revision" in str(r.get("tipo_documento", "")).lower() else "Tier_2_Trial")
        status = r.get("evidence_status", "insufficient_rct")
        subtipo_est = normalizar_nombre(str(r.get("subtipo_ataxia", "hereditaria_inespecifica")))
        fenotipos = ", ".join(r.get("phenotypes", [])) or r.get("dominio_clinico", "")
        intencion = ", ".join(r.get("intervention_intent", [])) or "No especificada"

        # Detectar riesgo de sobre-generalización entre subtipos (ej. protocolo probado en SCA3 aplicado a Friedreich)
        if subtipo_paciente == "friedreich" and ("sca3" in subtipo_est or "machado" in subtipo_est) and "friedreich" not in subtipo_est:
            alertas_subtipo.append(
                f"⚠️ ALERTA ANTI-SOBREGENERALIZACIÓN ({r['rs_id']}): Evidencia obtenida en {r.get('subtipo_ataxia')}, NO en Ataxia de Friedreich. No extrapolar dosis intensivas sin adaptar por fatiga neuromuscular y riesgo cardiológico."
            )

        if status == "contraindicated_or_harmful":
            bloqueos_seguridad.append(
                f"🛑 BLOQUEO DE SEGURIDAD ({r['rs_id']}): {r.get('intervencion_evaluada')} marcada como contraindicada o dañina ({r.get('limitaciones')})."
            )

        bloques.append(
            f"[RS_ID: {r['rs_id']} | Tier: {tier} | Status: {status} | Año: {r['anio_publicacion']} | "
            f"Subtipo estudio: {r.get('subtipo_ataxia', 'General')} | Fenotipos: {fenotipos} | Intención: {intencion}]\n"
            f"- Intervención vs Comparador: {r.get('intervencion_evaluada')} vs. {r.get('comparador', 'Control')}\n"
            f"- Desenlaces e Instrumentos: {r.get('desenlaces_primarios_secundarios', '')} ({r.get('instrumentos_y_tiempos', r.get('herramientas_evaluacion', ''))})\n"
            f"- Datos Estadísticos: Continuas: {r.get('variables_continuas', 'N/D')} | Categóricas: {r.get('variables_categoricas', 'N/D')}\n"
            f"- Nivel de evidencia y Sesgo: {r.get('nivel_evidencia')} | Sesgo: {r.get('riesgo_sesgo_calidad', 'N/D')} | Limitaciones: {r.get('limitaciones')}\n"
            f"- Síntesis: {r.get('contenido_chunk')}"
        )

    cabecera_reglas = ""
    if bloqueos_seguridad:
        cabecera_reglas += "\n".join(bloqueos_seguridad) + "\n\n"
    if alertas_subtipo:
        cabecera_reglas += "\n".join(alertas_subtipo) + "\n\n"

    return cabecera_reglas + "\n\n".join(bloques)

def buscar_evidencia_externa(patologia: str, trastorno: str) -> dict:
    mapa_en = {
        "Daño Cerebral Adquirido": "stroke",
        "Enfermedad de Parkinson": "Parkinson disease",
        "Ataxia Hereditaria": "ataxia",
        "Disfagia Neurogénica": "dysphagia",
        "Disartria": "dysarthria",
        "Afasia": "aphasia",
        "Dispraxia / Apraxia del Habla / Orofacial": "apraxia of speech"
    }
    q_pat = mapa_en.get(patologia, "neurological disorder")
    q_trast = mapa_en.get(trastorno, "speech therapy")
    query_simple = f"{q_pat} {q_trast}"
    query_enc = urllib.parse.quote_plus(query_simple)
    headers = {"User-Agent": "Mozilla/5.0"}

    texto_recuperado = ""

    url_pm = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query_enc}&retmode=json&retmax=5&sort=relevance"
    try:
        req_pm = urllib.request.Request(url_pm, headers=headers)
        ids = json.loads(urllib.request.urlopen(req_pm, timeout=8).read().decode()).get("esearchresult", {}).get("idlist", [])
        if ids:
            url_fetch = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml"
            req_fetch = urllib.request.Request(url_fetch, headers=headers)
            texto_recuperado += "=== PUBMED ===\n" + urllib.request.urlopen(req_fetch, timeout=8).read().decode("utf-8", errors="ignore")[:9000]
    except Exception as err_pm:
        texto_recuperado += f"\n[Aviso PubMed: {err_pm}]\n"

    url_ct = f"https://clinicaltrials.gov/api/v2/studies?query.term={query_enc}&pageSize=3"
    try:
        req_ct = urllib.request.Request(url_ct, headers=headers)
        datos_ct = json.loads(urllib.request.urlopen(req_ct, timeout=8).read().decode())
        for est in datos_ct.get("studies", []):
            prot = est.get("protocolSection", {})
            nct = prot.get("identificationModule", {}).get("nctId", "")
            tit = prot.get("identificationModule", {}).get("briefTitle", "")
            estado = prot.get("statusModule", {}).get("overallStatus", "")
            texto_recuperado += f"\nCLINICALTRIALS | ID: {nct} | URL: https://clinicaltrials.gov/study/{nct} | Estado: {estado} | Título: {tit}"
    except Exception as err_ct:
        texto_recuperado += f"\n[Aviso ClinicalTrials: {err_ct}]\n"

    enlaces = {
        "PubMed": f"https://pubmed.ncbi.nlm.nih.gov/?term={query_enc}",
        "PubMed Central (PMC)": f"https://pmc.ncbi.nlm.nih.gov/?term={query_enc}",
        "Cochrane Library": f"https://www.cochranelibrary.com/search?q={query_enc}",
        "speechBITE": "https://speechbite.com/",
        "ASHAWire": f"https://pubs.asha.org/action/doSearch?AllField={query_enc}",
        "ASHA Evidence Maps": "https://www.asha.org/evidence-maps/"
    }

    return {"datos_api": texto_recuperado, "enlaces": enlaces}


# 3. INTERFAZ VISUAL Y ENCABEZADO DEGRADADO
st.set_page_config(page_title="LogoPilot | CDSS Logopedia Neurológica", page_icon="🧭", layout="wide")

archivos_img = glob.glob("**/*.png", recursive=True) + glob.glob("**/*.jpg", recursive=True) + glob.glob("**/*.jpeg", recursive=True)
img_b64 = base64.b64encode(open(archivos_img[0], "rb").read()).decode() if archivos_img else ""

st.markdown(f"""
<style>
    .block-container {{ padding-top: 3.0rem; padding-bottom: 2.5rem; }}
    .header-cdss {{
        box-sizing: border-box;
        width: 100%;
        background: linear-gradient(90deg, #FFFFFF 0%, #FFFFFF 10%, #2563EB 23%, #1E3A8A 62%, #0F172A 100%);
        padding: 1.4rem 2.2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border: 1px solid #E2E8F0;
        border-left: 6px solid #1E3A8A;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }}
    .logo-fundido {{
        width: 84px;
        flex-shrink: 0;
        mix-blend-mode: multiply;
        filter: contrast(1.06) brightness(1.03);
    }}
    .header-contenido {{
        flex: 1;
        min-width: 0;
    }}
    .header-cdss h2 {{
        margin: 0;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: 0.3px;
    }}
    .titulo-logo {{
        color: #0F172A;
    }}
    .titulo-pilot {{
        color: #FFFFFF;
        text-shadow: 0 1px 3px rgba(15, 23, 42, 0.45);
    }}
    .header-inferior {{
        text-align: right;
        margin-top: 0.55rem;
    }}
    .header-cdss p {{
        margin: 0;
        font-size: 0.93rem;
        color: #F8FAFC;
        font-weight: 500;
        text-shadow: 0 1px 3px rgba(15, 23, 42, 0.55);
    }}
    .subtexto-rag {{
        display: block;
        color: #CBD5E1;
        font-size: 0.78rem;
        margin-top: 0.3rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }}
    div[data-testid="stExpander"] {{ border: 1px solid #CBD5E1; border-radius: 6px; background-color: #F8FAFC; }}
</style>
<div class="header-cdss">
    <img src="data:image/png;base64,{img_b64}" class="logo-fundido">
    <div class="header-contenido">
        <h2><span class="titulo-logo">LOGO</span><span class="titulo-pilot">-Pilot · Soporte Clínico para Logopedia Neurológica</span></h2>
        <div class="header-inferior">
            <p>Sistema de navegación y apoyo a la toma de decisiones en DCA, Parkinson y Ataxias Hereditarias</p>
            <span class="subtexto-rag">Corpus Auditado · Trazabilidad CitaRAG · Supervisión Humana (Art. 14 EU AI Act)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not os.getenv("OPENAI_API_KEY"):
    st.error("Falta configurar OPENAI_API_KEY en Secrets de Streamlit Cloud o en el entorno local.")
    st.stop()

tab_cdss, tab_extractor = st.tabs([
    "🧭 Soporte Clínico (CDSS)",
    "🧬 Extractor y Auditor de Evidencia (RS · Trials · Artículos)"
])

# ==============================================================================
# PESTAÑA 1: SOPORTE CLÍNICO (CDSS)
# ==============================================================================
with tab_cdss:
    indice = inicializar_indice()
    col_izq, col_centro, col_der = st.columns([1, 1.4, 1])

    with col_izq:
        c1, c1_i = st.columns([0.85, 0.15])
        c1.subheader("1. Contexto Clínico")
        with c1_i.popover("ℹ️"):
            st.caption("1. Elija fuente de datos.\n2. Verifique anonimización (sin nombres ni NHC).\n3. Pulse Analizar.")

            # 1. Selector de las 3 vías de entrada
    modo_ingreso = st.radio(
        "Modo de entrada de datos:",
        ["Banco Maestro (31 Casos)", "Subir Informe (PDF/DOCX/TXT/XLSX)", "Ficha Caso Real (Manual)"],
        on_change=limpiar_estado_anterior
    )

    opciones_pat = [p.value for p in PatologiaBase]
    idx_pat = 0
    texto_auto = ""

    # VÍA 1: Desplegable del Banco Maestro
    if modo_ingreso == "Banco Maestro (31 Casos)":
        casos_dict = {"-- Selecciona un caso --": None}
        for archivo_maestro in [
            "LOGO_LLM_25_Casos_Clinicos_Sinteticos_Banco_Maestro_v1_0.json",
            "casos_desafio_ataxia_rs_26_31.json"
        ]:
            if os.path.exists(archivo_maestro):
                with open(archivo_maestro, "r", encoding="utf-8") as f:
                    datos_json = json.load(f)
                    lista_casos = datos_json.get("casos", []) if isinstance(datos_json, dict) else datos_json
                    for c in lista_casos:
                        if isinstance(c, dict):
                            cid = c.get("id_caso") or c.get("id") or "CASO"
                            tit = c.get("titulo") or c.get("perfil") or ""
                            casos_dict[f"{cid} · {tit}"] = c

        seleccion = st.selectbox(
            "Selecciona caso sintético:",
            list(casos_dict.keys()),
            on_change=limpiar_estado_anterior
        )
           caso_activo = casos_dict[seleccion]

        if caso_activo:
            pat_caso = str(caso_activo.get("patologia_base") or caso_activo.get("patologia") or "")
            idx_pat = next((i for i, p in enumerate(opciones_pat) if p.lower() in pat_caso.lower()), 0)
            lineas_caso = [
                f"{k.upper().replace('_', ' ')}: {v}"
                for k, v in caso_activo.items()
                if k not in ["id_caso", "id", "numero", "rubrica", "criterio_experto"] and isinstance(v, str)
            ]
            texto_auto = "\n".join(lineas_caso)

        elif modo_ingreso == "Subir Informe (PDF/DOCX/TXT/XLSX)":
            archivo_informe = st.file_uploader("Sube el informe clínico:", type=["pdf", "docx", "txt", "xlsx"])
            if archivo_informe:
                texto_auto = extraer_texto_archivo(archivo_informe)
                st.success("Informe cargado y anonimizado localmente.")

        else:
            texto_auto = renderizar_ficha_real()

        pat_sel = st.selectbox("Patología Neurológica Base", opciones_pat, index=idx_pat)
        trast_sel = st.selectbox("Dominio Logopédico", [t.value for t in TipoTrastorno])
        instrumental = st.checkbox("Consta prueba instrumental previa (FEES / VFSS)")
        lineas_estimadas = texto_auto.count("\n") + (len(texto_auto) // 48) + 3
        altura_dinamica = max(150, min(lineas_estimadas * 22, 650))
        notas = st.text_area("Hallazgos clínicos consolidados (Revisar antes de enviar):", value=texto_auto, height=altura_dinamica)

        analizar = st.button("Analizar Caso Clínico", type="primary", use_container_width=True)

    if analizar and notas:
        consulta = f"Patología: {pat_sel}. Trastorno: {trast_sel}. Instrumental previo: {'Sí' if instrumental else 'No'}. Hallazgos: {notas}"
        nodos = indice.as_retriever(similarity_top_k=8).retrieve(consulta)
        contexto = "\n\n".join([f"[{n.metadata['claim_id']} | {n.metadata['doc_id']} | {n.metadata['localizador']}] {n.text}" for n in nodos])
        contexto_rs = recuperar_evidencia_rs(pat_sel, trast_sel, notas)

prompt = (
    "Eres un Sistema de Soporte a la Decisión Clínica (CDSS) experto en Logopedia Neurológica. "
    "Redacta un informe técnico riguroso basándote ESTRICTAMENTE en dos capas jerárquicas de conocimiento:\n\n"
    "=== CAPA 1: PROTOCOLOS Y GUÍAS CLÍNICAS (Tier_3_Protocol · CÓMO HACER / PRÁCTICA SEGURA) ===\n"
    f"{contexto}\n\n"
    "=== CAPA 2: REVISIONES SISTEMÁTICAS (Tier_1_SR) Y ENSAYOS PRIMARIOS (Tier_2_Trial) ===\n"
    f"{contexto_rs}\n\n"
    f"DATOS DEL CASO CLÍNICO:\n{consulta}\n\n"
    "REGLAS DETERMINISTAS DE RAZONAMIENTO Y SÍNTESIS (OBLIGATORIAS):\n"
    "1. DISTINCIÓN ENTRE 'insufficient_rct' Y 'contraindicated_or_harmful':\n"
    "   - Si un registro tiene Status 'insufficient_rct' (ej. Cochrane indica ausencia de ECA), NO paralices ni rechaces el tratamiento: extrae los protocolos seguros de Tier_3_Protocol y Tier_2_Trial para ofrecer un plan de acción real, aclarando el nivel de certeza bajo.\n"
    "   - Si un registro o regla clínica marca 'contraindicated_or_harmful' (ej. NMES en Ataxia de Friedreich, tDCS fuera de ensayo o cargas extremas con riesgo de fatiga/aspiración), activa el bloqueo y desaconséjalo explícitamente.\n"
    "2. RESOLUCIÓN TEMPORAL (TEMPORAL DECAY OVERRIDE): Si un ensayo clínico reciente (Tier_2_Trial) es posterior a una Revisión Sistemática (Tier_1_SR) y evalúa el fenotipo del paciente, prioriza los parámetros del ensayo reciente aclarando que aún no ha sido incorporado a una RS.\n"
    "3. ANTI-SOBREGENERALIZACIÓN DE SUBTIPO: Nunca extrapoles automáticamente dosificaciones intensivas probadas solo en SCA3 hacia Ataxia de Friedreich o fenotipos con fatiga severa/miocardiopatía; adapta la intensidad y exige control de fatiga.\n"
    "4. ESTRUCTURA EN 3 BLOQUES DENTRO DE 'respuesta_dilema_clinico':\n"
    "   - [Bloque 1 · Recomendación Operativa]: Acción terapéutica concreta (frecuencia, intensidad, foco y maniobras) extraída de Tier_2_Trial y Tier_3_Protocol.\n"
    "   - [Bloque 2 · Nivel de Certeza y Contexto RS]: Grado de respaldo según Tier_1_SR (citando datos estadísticos o limitaciones).\n"
    "   - [Bloque 3 · Banderas Rojas y Contraindicaciones]: Criterios de seguridad, control de fatiga y derivaciones.\n"
    "5. TRAZABILIDAD Y SEGURIDAD IDDSI: Diferencia en 'evidencia_rag' entre [Guía Clínica / Manual] y [Revisión Sistemática / Estudio Primario]. Si hay sospecha de disfagia, indica en 'adaptacion_dietetica_iddsi' que no se prescriben viscosidades IDDSI sin valoración instrumental previa (FEES/VFSS); si no hay disfagia, déjalo en ''."
    )


for intento in range(1, 4):
    try:
        llm_activo = LlamaOpenAI(model="gpt-4o", temperature=0.1 if intento == 1 else 0.0)
        st.session_state["reporte"] = llm_activo.as_structured_llm(ReporteClinicoLogopedico).complete(prompt).raw
        st.session_state.pop("ensayos_externos", None)
        st.session_state.pop("enlaces_externos", None)
        break
    except ValidationError as e:
        prompt += f"\nCORRECCIÓN (Intento {intento}): {e}"

    if "reporte" in st.session_state:
        reporte = st.session_state["reporte"]
        with col_centro:
            st.subheader("2. Reporte Clínico")

            c_cie, c_cie_i = st.columns([0.88, 0.12])
            c_cie.info(f"**CIE-11:** {reporte.diagnostico.codigo_cie11} | **Patología:** {reporte.diagnostico.patologia_neuro_base.value}")
            pop_cie = c_cie_i.popover("ℹ️")
            pop_cie.markdown(f"**Guía de Codificación CIE-11 ({reporte.diagnostico.codigo_cie11})**")
            pop_cie.caption("Valida la correspondencia entre el diagnóstico neurológico base y el trastorno funcional:")
            pop_cie.markdown(
                "- **8A00.0:** Enfermedad de Parkinson.\n"
                "- **8A03.10:** Ataxias hereditarias (Ataxia de Friedreich).\n"
                "- **8B11 / 8B20:** DCA vascular (Ictus isquémico / hemorrágico).\n"
                "- **NA07:** DCA traumático (Traumatismo intracraneal / TCE)."
            )
            pop_cie.markdown("[Consultar navegador OMS CIE-11](https://icd.who.int/browse/2024-01/mms/es)")

            c_an, c_an_i = st.columns([0.88, 0.12])
            c_an.markdown("#### Análisis Fisiopatológico y Diagnóstico Diferencial")
            pop_an = c_an_i.popover("ℹ️")
            pop_an.markdown("**Auditoría del Razonamiento Fisiopatológico**")
            pop_an.caption(
                "• **Subsistemas motores:** Verifica respiración, fonación, resonancia, articulación y prosodia.\n"
                "• **Fases deglutorias:** Distingue alteración de eficacia (residuo) frente a seguridad (aspiración).\n"
                "• **Diferencial:** Comprueba que no se confundan déficits motores (disartria/apraxia) con trastornos lingüísticos o cognitivos."
            )

            st.write(reporte.diagnostico.analisis_fisiopatologico)
            st.markdown(
                f"""<div style="background-color: #F5F3FF; border-left: 4px solid #4C1D95; padding: 0.95rem 1.15rem; border-radius: 6px; margin: 0.8rem 0 1.2rem 0; color: #1E293B; font-size: 1.03rem; line-height: 1.55;">
                <strong style="color: #1E1B4B;">Diagnóstico Diferencial:</strong> {reporte.diagnostico.diagnostico_diferencial}
                </div>""",
                unsafe_allow_html=True
            )

            st.error(f"**Resolución del Dilema Clínico:** {reporte.intervencion.respuesta_dilema_clinico}")

            if reporte.evaluacion.signos_alerta_seguridad:
                st.error("**Alertas de Seguridad:**\n" + "\n".join([f"- {a}" for a in reporte.evaluacion.signos_alerta_seguridad]))

            if reporte.evaluacion.informacion_clinica_no_disponible:
                st.info("**Información Clínica No Disponible (Datos Ausentes):**\n" + "\n".join([f"- {d}" for d in reporte.evaluacion.informacion_clinica_no_disponible]))

            if reporte.evaluacion.evidencia_cientifica_insuficiente:
                st.warning("**Declaración de Evidencia Científica Insuficiente:**\n" + "\n".join([f"- {ev}" for ev in reporte.evaluacion.evidencia_cientifica_insuficiente]))
                if st.button("🔎 Consultar Bases Científicas y Ensayos (PubMed, Cochrane, ClinicalTrials, ASHA)", key="btn_pubmed"):
                    with st.spinner("Consultando PubMed/MEDLINE, Cochrane y ClinicalTrials.gov..."):
                        res_ext = buscar_evidencia_externa(pat_sel, trast_sel)
                        st.session_state["enlaces_externos"] = res_ext["enlaces"]
                        if res_ext["datos_api"]:
                            prompt_ev = (
                                f"Extrae los estudios y ensayos clínicos más relevantes de estos registros sobre {pat_sel} y {trast_sel}, "
                                f"enfocándote en las dudas del caso ({'; '.join(reporte.evaluacion.evidencia_cientifica_insuficiente)}). "
                                f"Indica la base de origen (PubMed, Cochrane o ClinicalTrials NCT). No inventes datos ni códigos:\n\n{res_ext['datos_api']}\n\n"
                                "REGLA ESTRICTA DE ENLACES en 'referencia_estudio': Incluye autor, año y el enlace Markdown usando EXCLUSIVAMENTE el PMID o código NCT literal que aparece en los registros anteriores:\n"
                                "- Si tiene PMID: [PubMed PMID](https://pubmed.ncbi.nlm.nih.gov/CODIGO_PMID/)\n"
                                "- Si tiene NCT: [ClinicalTrials NCT](https://clinicaltrials.gov/study/CODIGO_NCT)\n"
                                "Nunca uses https://nih.gov ni /ct2/show/. Si un registro no tiene código PMID ni NCT, pon solo autor y año sin enlace."
                            )
                            llm_ev = LlamaOpenAI(model="gpt-4o", temperature=0.0)
                            res_ev = llm_ev.as_structured_llm(SintesisEnsayosExternos).complete(prompt_ev).raw
                            st.session_state["ensayos_externos"] = res_ev.ensayos
                        else:
                            st.session_state["ensayos_externos"] = []

                if "ensayos_externos" in st.session_state:
                    if st.session_state["ensayos_externos"]:
                        st.markdown("**Síntesis de Ensayos y Revisiones (PubMed · Cochrane · ClinicalTrials.gov):**")
                        for e in st.session_state["ensayos_externos"]:
                            st.markdown(f"- **{e.referencia_estudio}** | *Comparación:* {e.intervenciones_comparadas} | *Resultados:* {e.hallazgos_principales}")

                    if "enlaces_externos" in st.session_state:
                        links_md = " · ".join([f"[{nombre}]({url})" for nombre, url in st.session_state["enlaces_externos"].items()])
                        st.info(f"**Búsqueda directa pre-filtrada en repositorios especializados:**\n\n{links_md}")

            c_ev, c_ev_i = st.columns([0.88, 0.12])
            c_ev.markdown("**Evaluación Sugerida:**")
            pop_ev = c_ev_i.popover("ℹ️")
            pop_ev.markdown("**Criterio de Selección de Pruebas y Escalas**")
            pop_ev.caption(
                "• **Pertinencia clínica:** Las escalas se seleccionan según los subsistemas afectados y la etiología base.\n"
                "• **Cribado vs. Instrumental:** Diferencia herramientas clínicas (MECV-V, EAT-10, FDA-2, CAPE-V) de pruebas instrumentales confirmatorias (FEES / VFSS).\n"
                "• **Supervisión:** Marca únicamente las pruebas que consideres indicadas para el caso."
            )

            for esc in reporte.evaluacion.escalas_sugeridas:
                st.checkbox(esc, value=True, key=f"e_{esc}")

            if reporte.evaluacion.criterios_derivacion:
                st.markdown("**Derivación Multidisciplinar:** " + ", ".join(reporte.evaluacion.criterios_derivacion))

            st.markdown("**Objetivos Terapéuticos:**")
            for obj in reporte.intervencion.objetivos_terapeuticos:
                st.markdown(f"- {obj}")

            c_int, c_int_i = st.columns([0.88, 0.12])
            c_int.markdown("**Intervención Propuesta:**")
            pop_int = c_int_i.popover("ℹ️")
            pop_int.markdown("**Criterios de Selección y Dosificación Terapéutica**")
            pop_int.caption(
                "• **Enfoques:** Contrasta estrategias compensatorias (posturales/maniobras) frente a reactivación neuromuscular.\n"
                "• **Control de fatiga:** En patología neurodegenerativa, prioriza sesiones cortas y valora el estado motor ON/OFF.\n"
                "• **Human-in-the-Loop:** Marca únicamente las técnicas que valides bajo tu criterio profesional antes de exportar."
            )

            for c in reporte.intervencion.estrategias_compensatorias:
                st.checkbox(f"[Compensatoria] {c}", key=f"c_{c}")
            for r in reporte.intervencion.reactivacion_neuromuscular:
                st.checkbox(f"[Reactivación] {r}", key=f"r_{r}")

            hay_disfagia = (
                trast_sel == TipoTrastorno.DISFAGIA.value
                or "disfagia" in notas.lower()
                or "deglu" in notas.lower()
                or "atragantamiento" in notas.lower()
                or "aspiraci" in notas.lower()
            )
            if hay_disfagia and reporte.intervencion.adaptacion_dietetica_iddsi:
                st.markdown(f"**Pauta IDDSI:** {reporte.intervencion.adaptacion_dietetica_iddsi}")

            st.divider()
            resumen_txt = (
                f"INFORME CLÍNICO GENERADO CON LOGO-PILOT\n"
                f"CIE-11: {reporte.diagnostico.codigo_cie11} | Patología: {reporte.diagnostico.patologia_neuro_base.value}\n\n"
                f"ANÁLISIS FISIOPATOLÓGICO:\n{reporte.diagnostico.analisis_fisiopatologico}\n\n"
                f"DIAGNÓSTICO DIFERENCIAL:\n{reporte.diagnostico.diagnostico_diferencial}\n\n"
                f"RESOLUCIÓN DEL DILEMA:\n{reporte.intervencion.respuesta_dilema_clinico}\n\n"
                f"PAUTA IDDSI:\n{reporte.intervencion.adaptacion_dietetica_iddsi}"
            )
            st.download_button("Validar y Descargar para Historia Clínica", data=resumen_txt, file_name="informe_LOGO_Pilot.txt", use_container_width=True)

        with col_der:
            c_rag, c_rag_i = st.columns([0.85, 0.15])
            c_rag.subheader("3. Inspector CitaRAG")
            pop_rag = c_rag_i.popover("ℹ️")
            pop_rag.markdown("**¿Cómo funciona la trazabilidad CitaRAG?**")
            pop_rag.caption(
                "• **Recuperación determinista:** Extrae únicamente los hechos factuales más cercanos semánticamente de los nodos auditados.\n"
                "• **Semáforo metodológico:** 🟢 Bajo riesgo / Certeza moderada-alta · 🔵 Precaución / Piloto · 🔴 Alto riesgo / Certeza muy baja.\n"
                "• **Auditoría directa:** Usa el enlace **DOI** o descarga el **PDF original** para verificar la fuente primaria."
            )

            mapa_fuentes = {}
            if os.path.exists("mapa_fuentes.json"):
                with open("mapa_fuentes.json", "r", encoding="utf-8") as mf:
                    mapa_fuentes = json.load(mf)

            mapa_rs_pdfs = {}
            mapa_rs_meta = {}
            for r_item in cargar_capa_rs():
                rs_id_val = str(r_item.get("rs_id", "")).strip().upper()
                arch_val = str(r_item.get("archivo_origen", "")).strip()
                if rs_id_val:
                    mapa_rs_meta[rs_id_val] = r_item
                    if arch_val:
                        mapa_rs_pdfs[rs_id_val] = arch_val

            subcarpeta_pat = {
                "Daño Cerebral Adquirido": "dca",
                "Enfermedad de Parkinson": "parkinson",
                "Ataxia Hereditaria": "ataxia"
            }.get(pat_sel, "")

            carpeta_referencias = "referencias" if os.path.exists("referencias") else r"F:\VIU\IA SLP\Proyecto LOGO-LLM\Referencias"
            pdfs_disponibles = {}
            pdfs_patologia = []

            if os.path.exists(carpeta_referencias):
                for raiz, _, archivos in os.walk(carpeta_referencias):
                    es_carpeta_pat = subcarpeta_pat and (subcarpeta_pat in normalizar_nombre(raiz))
                    for arch in archivos:
                        if arch.lower().endswith(".pdf"):
                            ruta_completa = os.path.join(raiz, arch)
                            pdfs_disponibles[normalizar_nombre(arch)] = (arch, ruta_completa)
                            if es_carpeta_pat:
                                pdfs_patologia.append((normalizar_nombre(arch), arch, ruta_completa))

            for idx, cita in enumerate(reporte.evidencia_rag):
                id_limpio = cita.id_nodo.upper().replace("RS_ID:", "").strip()
                texto_busqueda = f"{id_limpio} {cita.fuente_documental}".upper()
                doc_id_limpio = next((k for k in mapa_fuentes.keys() if k.upper() in texto_busqueda), cita.fuente_documental)
                info_doc = mapa_fuentes.get(doc_id_limpio, {})
                titulo_doc = info_doc.get("titulo", cita.fuente_documental)

                meta_rs = mapa_rs_meta.get(id_limpio)
                if not meta_rs:
                    prefijo_rs = "-".join(id_limpio.split("-")[:2])
                    for rs_k, r_obj in mapa_rs_meta.items():
                        if rs_k == id_limpio or (prefijo_rs and rs_k.startswith(prefijo_rs)):
                            meta_rs = r_obj
                            break

                if cita.tipo_fuente == "Guía Clínica / Manual":
                    icono_fuente = "📘 [Guía/Manual]"
                else:
                    if meta_rs:
                        emoji_sem, _, _ = evaluar_semaforo(
                            meta_rs.get("riesgo_sesgo_calidad", ""),
                            meta_rs.get("nivel_evidencia", "")
                        )
                    else:
                        emoji_sem = "🔵"
                    icono_fuente = f"{emoji_sem} [Rev. Sistemática / Estudio]"

                with st.expander(f"{icono_fuente} {id_limpio} · {doc_id_limpio}", expanded=True):
                    st.markdown(f"**{titulo_doc}**")
                    st.caption(f"Localizador: {cita.fuente_documental}")

                    if meta_rs and cita.tipo_fuente != "Guía Clínica / Manual":
                        emoji_sem, desc_sem, color_bg = evaluar_semaforo(
                            meta_rs.get("riesgo_sesgo_calidad", ""),
                            meta_rs.get("nivel_evidencia", "")
                        )
                        st.markdown(
                            f"<div style='background-color:{color_bg}; padding:6px 10px; border-radius:6px; font-size:0.82rem; color:#0F172A; margin-bottom:8px;'>"
                            f"<strong>{emoji_sem} {desc_sem}</strong><br>"
                            f"<span>Sesgo/Calidad: {meta_rs.get('riesgo_sesgo_calidad', 'N/D')} · Certeza: {meta_rs.get('nivel_evidencia', 'N/D')}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                    st.write(cita.hecho_clinico_factual)

                    doi = str(info_doc.get("doi", "") or (meta_rs.get("doi", "") if meta_rs else "")).strip()
                    if doi.startswith("10."):
                        st.markdown(f"[Abrir publicación oficial (DOI: {doi})](https://doi.org/{doi})")

                    nombre_pdf = info_doc.get("archivo_pdf", "")
                    if not nombre_pdf:
                        nombre_pdf = mapa_rs_pdfs.get(id_limpio, "")
                        if not nombre_pdf and meta_rs:
                            nombre_pdf = meta_rs.get("archivo_origen", "")

                    info_pdf = pdfs_disponibles.get(normalizar_nombre(nombre_pdf)) if nombre_pdf else None

                    if not info_pdf and cita.tipo_fuente != "Guía Clínica / Manual" and pdfs_patologia:
                        pistas = normalizar_nombre(f"{nombre_pdf} {id_limpio} {cita.hecho_clinico_factual}").split()
                        pistas_utiles = [p for p in pistas if len(p) > 3 and p not in ["resumen", "conclusiones", "tabla", "estudio", "discusion", "ataxia", "revision", "sistematica"]]
                        mejor_score = 0
                        for norm_arch, arch_real, ruta_real in pdfs_patologia:
                            score = sum(1 for p in set(pistas_utiles) if p in norm_arch)
                            if score > mejor_score:
                                mejor_score = score
                                info_pdf = (arch_real, ruta_real)

                    if info_pdf:
                        nombre_real, ruta_encontrada = info_pdf
                        with open(ruta_encontrada, "rb") as pdf_file:
                            st.download_button(
                                label=f"📄 Abrir / Descargar PDF ({nombre_real})",
                                data=pdf_file.read(),
                                file_name=nombre_real,
                                mime="application/pdf",
                                key=f"pdf_{id_limpio}_{idx}"
                            )


# ==============================================================================
# PESTAÑA 2: EXTRACTOR Y AUDITOR DE EVIDENCIA (RS · TRIALS · ARTÍCULOS)
# ==============================================================================
with tab_extractor:
    st.subheader("🧬 Extractor y Auditor de Evidencia Científica (C2)")
    st.caption(
        "Procesa Revisiones Sistemáticas, Ensayos Clínicos (Clinical Trials) y Artículos Primarios para alimentar "
        "`capa_evidencia_rs.json` en tiempo real sin contaminar los protocolos clínicos de la Capa 1."
    )

    if "capa_rs_acumulada" not in st.session_state:
        st.session_state["capa_rs_acumulada"] = cargar_capa_rs()

    col_ext_izq, col_ext_der = st.columns([1, 1.35])

    with col_ext_izq:
        st.markdown("#### 1. Carga y Metadatos del Documento")
        pdf_subido = st.file_uploader("Sube el PDF científico:", type=["pdf"], key="uploader_capa2")

        tipo_doc = st.selectbox(
            "Tipo de documento científico:",
            [
                "Revisión Sistemática / Metaanálisis",
                "Ensayo Clínico Aleatorizado / Clinical Trial",
                "Artículo Científico Primario / Observacional"
            ]
        )

        patologia_ext = st.selectbox(
            "Patología Neurológica Base:",
            [p.value for p in PatologiaBase],
            index=2,
            key="pat_extractor"
        )

        c_id, c_anio = st.columns([0.65, 0.35])
        id_base = c_id.text_input("Identificador corto (sin espacios):", value="COCHRANE-2014-ATX")
        anio_pub = c_anio.number_input("Año:", min_value=1990, max_value=2030, value=2024, step=1)

        st.markdown(
            "**Leyenda del Semáforo Metodológico:**\n"
            "- 🟢 **Verde:** Bajo riesgo de sesgo (RoB 2 / JBI alto) o certeza moderada-alta.\n"
            "- 🔵 **Azul:** Algunas preocupaciones / Riesgo moderado o estudio piloto.\n"
            "- 🔴 **Rojo:** Alto riesgo de sesgo o certeza muy baja / insuficiente."
        )

        procesar_doc = st.button("⚙️ Extraer Metadatos, Estadística y Sesgo", type="primary", use_container_width=True)

    if procesar_doc:
        if not pdf_subido:
            st.warning("Por favor, sube primero un archivo PDF.")
        elif not id_base.strip():
            st.warning("Escribe un identificador corto (ej. COCHRANE-2014-ATX).")
        else:
            with st.spinner("Analizando texto, variables continuas/categóricas y riesgo de sesgo con GPT-4o..."):
                lector = PdfReader(pdf_subido)
                paginas = []
                for i, pag in enumerate(lector.pages[:35]):
                    txt = pag.extract_text() or ""
                    if txt.strip():
                        paginas.append(f"--- PÁGINA {i + 1} ---\n{txt}")
                texto_pdf = "\n\n".join(paginas)[:48000]

                prompt_extraccion = (
                    f"Eres un metodólogo clínico experto en Medicina Basada en la Evidencia y Logopedia Neurológica. "
                    f"Analiza el siguiente documento de tipo '{tipo_doc}' sobre '{patologia_ext}' (ID base: {id_base.strip()}, Año: {anio_pub}) "
                    f"y extrae toda la información estructurada con máximo rigor.\n\n"
                    "REGLAS DE FRAGMENTACIÓN:\n"
                    "1. 'resumen_conclusiones' (prioridad 'alta'): 1 bloque único con las conclusiones globales de los autores.\n"
                    "2. 'tabla_estudio_primario' (prioridad 'media'): Si es una Revisión Sistemática, convierte cada estudio incluido en un bloque independiente; si es un Ensayo Clínico o Artículo Primario, extrae sus brazos o subgrupos principales.\n"
                    "3. 'discusion_practica_clinica' (prioridad 'media'): Fragmenta las implicaciones clínicas y de seguridad por dominio (Evaluación, Disartria, Disfagia Neurogénica, etc.).\n\n"
                    "REGLAS DE EXTRACCIÓN DE DATOS:\n"
                    "- Extrae descripciones detalladas de intervenciones/exposición y comparadores.\n"
                    "- Detalla desenlaces primarios y secundarios, escalas/instrumentos empleados y momentos de medición.\n"
                    "- Variables continuas: tamaño muestral (n) por grupos, media y desviación estándar (DE), mediana y rango intercuartílico (RIC), error estándar (EE) si aplica.\n"
                    "- Variables categóricas: número de eventos, proporciones, Odds Ratio (OR), Riesgo Relativo (RR) con IC 95% y valores p.\n"
                    "- Hallazgos cualitativos: categorías, citas o hallazgos relevantes.\n"
                    "- Riesgo de sesgo / calidad metodológica: especifica la valoración (Cochrane RoB 2, ROBINS-I, JBI o GRADE) indicando claramente 'Bajo riesgo', 'Algunas preocupaciones / Moderado' o 'Alto riesgo'.\n\n"
                    f"TEXTO DEL DOCUMENTO ({pdf_subido.name}):\n{texto_pdf}"
                )

                client = OpenAI()
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o",
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": "Extrae datos científicos estructurados sin inventar cifras no presentes en el texto."},
                        {"role": "user", "content": prompt_extraccion}
                    ],
                    response_format=ExtraccionDocumentoCientifico,
                )
                resultado_ext = completion.choices[0].message.parsed

                id_prefijo = id_base.strip().upper()
                nuevos_registros = []
                for idx_c, ch in enumerate(resultado_ext.chunks, start=1):
                    emoji_s, etiqueta_s, _ = evaluar_semaforo(ch.riesgo_sesgo_calidad, ch.nivel_evidencia)
                    reg = ch.model_dump()
                    if not reg["rs_id"].upper().startswith(id_prefijo):
                        reg["rs_id"] = f"{id_prefijo}-{idx_c:02d}"
                    reg["tipo_fuente"] = "revision_sistematica"
                    reg["tipo_documento"] = tipo_doc
                    reg["anio_publicacion"] = int(anio_pub)
                    reg["patologia"] = patologia_ext
                    reg["titulo_documento"] = resultado_ext.titulo_documento
                    reg["autores_principales"] = resultado_ext.autores_principales
                    reg["doi"] = resultado_ext.doi
                    reg["archivo_origen"] = pdf_subido.name
                    reg["semaforo"] = f"{emoji_s} {etiqueta_s}"
                    nuevos_registros.append(reg)

                # Reemplazar bloques previos del mismo id_prefijo y conservar el resto
                existentes = [
                    r for r in st.session_state["capa_rs_acumulada"]
                    if not str(r.get("rs_id", "")).upper().startswith(id_prefijo)
                ]
                st.session_state["capa_rs_acumulada"] = existentes + nuevos_registros
                st.session_state["ultimos_extraidos"] = nuevos_registros

                # Guardar en disco local y refrescar caché de la Pestaña 1 al instante
                with open("capa_evidencia_rs.json", "w", encoding="utf-8") as f_out:
                    json.dump(st.session_state["capa_rs_acumulada"], f_out, ensure_ascii=False, indent=2)
                cargar_capa_rs.clear()

                st.success(f"Extraídos {len(nuevos_registros)} bloques de '{pdf_subido.name}' e integrados en memoria activa.")

    with col_ext_der:
        st.markdown("#### 2. Auditoría Visual y Exportación JSON")

        bloques_mostrar = st.session_state.get("ultimos_extraidos") or st.session_state["capa_rs_acumulada"][-6:]
        if bloques_mostrar:
            for b in bloques_mostrar:
                emoji_s, desc_s, color_bg = evaluar_semaforo(
                    b.get("riesgo_sesgo_calidad", ""),
                    b.get("nivel_evidencia", "")
                )
                with st.expander(f"{emoji_s} {b.get('rs_id')} · {b.get('seccion_rs')} ({b.get('dominio_clinico')})", expanded=True):
                    st.markdown(
                        f"<div style='background-color:{color_bg}; padding:6px 10px; border-radius:6px; font-size:0.83rem; color:#0F172A; margin-bottom:8px;'>"
                        f"<strong>{emoji_s} {desc_s}</strong><br>"
                        f"<span>Riesgo de sesgo / Calidad: {b.get('riesgo_sesgo_calidad')} · Nivel: {b.get('nivel_evidencia')}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**Intervención:** {b.get('intervencion_evaluada')} | **Comparadores:** {b.get('comparadores')}")
                    st.markdown(f"**Desenlaces e instrumentos:** {b.get('desenlaces_primarios_secundarios')} ({b.get('herramientas_evaluacion')}) · *Tiempos:* {b.get('momentos_medicion')}")
                    st.markdown(f"**Variables continuas:** {b.get('variables_continuas')} | **Variables categóricas:** {b.get('variables_categoricas')}")
                    if b.get("hallazgos_cualitativos_citas"):
                        st.markdown(f"**Hallazgos cualitativos / Citas:** {b.get('hallazgos_cualitativos_citas')}")
                    st.write(b.get("contenido_chunk"))
        else:
            st.info("Aún no hay bloques extraídos en esta sesión. Sube un PDF a la izquierda para comenzar.")

        st.divider()
        total_bloques = len(st.session_state["capa_rs_acumulada"])
        st.caption(f"Total acumulado en `capa_evidencia_rs.json`: **{total_bloques} bloques**.")
        json_descarga = json.dumps(st.session_state["capa_rs_acumulada"], ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Descargar capa_evidencia_rs.json actualizado (para respaldar en GitHub)",
            data=json_descarga,
            file_name="capa_evidencia_rs.json",
            mime="application/json",
            use_container_width=True
        )
