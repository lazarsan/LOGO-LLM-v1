import json
import os
import base64
import glob
import unicodedata
import urllib.request
import urllib.parse
from enum import Enum
from typing import List, Literal
import streamlit as st
from pydantic import BaseModel, Field, model_validator, ValidationError
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from lector_informes import extraer_texto_archivo
from ficha_clinica_real import renderizar_ficha_real


def normalizar_nombre(texto: str) -> str:
    sin_acentos = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return sin_acentos.lower().replace("_", " ").replace("-", " ").strip()


def limpiar_estado_anterior():
    st.session_state.pop("reporte", None)
    st.session_state.pop("ensayos_externos", None)
    st.session_state.pop("enlaces_externos", None)


def calcular_semaforo_evidencia(riesgo_sesgo: str, nivel_evidencia: str) -> tuple:
    txt = f"{riesgo_sesgo} {nivel_evidencia}".lower()
    if any(k in txt for k in ["alto riesgo", "high risk", "muy baja", "muy bajo", "insuficiente", "jbi bajo", "critico", "crítico"]):
        return ("🔴", "Alerta metodológica / Certeza baja", "#FEE2E2", "#991B1B")
    if any(k in txt for k in ["bajo riesgo", "low risk", "alta certeza", "moderada", "jbi alto"]):
        return ("🟢", "Solidez metodológica / Buena certeza", "#DCFCE7", "#166534")
    return ("🔵", "Precaución / Evidencia intermedia o piloto", "#DBEAFE", "#1E40AF")


# 1. ESQUEMA DE DATOS Y CORTAFUEGOS CLÍNICO
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
    id_nodo: str = Field(description="ID exacto del claim (ej. ATX-CL-007) o RS_ID (ej. COCHRANE-2014-ATX-RESUMEN).")
    tipo_fuente: Literal["Guía Clínica / Manual", "Revisión Sistemática / Estudio Primario"] = Field(
        description="Origen jerárquico de la evidencia."
    )
    fuente_documental: str = Field(description="ID del documento, año y localizador físico o sección de RS.")
    hecho_clinico_factual: str = Field(description="Dato clínico objetivo y estadístico extraído sin narrativa autoral.")


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


# 2. CARGA DEL ÍNDICE VECTORIAL Y CAPA DE REVISIONES SISTEMÁTICAS
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
def cargar_capa_rs() -> list:
    if os.path.exists("capa_evidencia_rs.json"):
        with open("capa_evidencia_rs.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def recuperar_evidencia_rs(patologia: str, trastorno: str) -> str:
    registros_rs = cargar_capa_rs()
    if not registros_rs:
        return "No hay revisiones sistemáticas cargadas en la capa de evidencia."

    filtrados = [r for r in registros_rs if r.get("patologia", "").lower() == patologia.lower()]

    resumenes = [r for r in filtrados if r.get("seccion_rs") == "resumen_conclusiones"]
    especificos = [
        r for r in filtrados
        if r.get("seccion_rs") != "resumen_conclusiones"
        and (r.get("dominio_clinico", "").lower() in [trastorno.lower(), "evaluación", "general"])
    ]

    resumenes.sort(key=lambda x: x.get("anio_publicacion", 0), reverse=True)
    especificos.sort(key=lambda x: x.get("anio_publicacion", 0), reverse=True)

    seleccionados = resumenes[:2] + especificos[:6]
    if not seleccionados:
        return "Sin registros específicos en la capa de Revisiones Sistemáticas para esta combinación."

    bloques = []
    for r in seleccionados:
        extras = []
        if r.get("comparadores"):
            extras.append(f"Comparador: {r['comparadores']}")
        if r.get("desenlaces_primarios") or r.get("desenlaces_secundarios"):
            extras.append(f"Desenlaces: {r.get('desenlaces_primarios', '')} / {r.get('desenlaces_secundarios', '')}")
        if r.get("momentos_medicion"):
            extras.append(f"Momentos medición: {r['momentos_medicion']}")
        if r.get("variables_continuas"):
            extras.append(f"Var. continuas: {r['variables_continuas']}")
        if r.get("variables_categoricas"):
            extras.append(f"Var. categóricas: {r['variables_categoricas']}")
        if r.get("riesgo_sesgo"):
            extras.append(f"Riesgo de sesgo (Cochrane/JBI): {r['riesgo_sesgo']}")
        if r.get("hallazgos_cualitativos"):
            extras.append(f"Hallazgos/citas: {r['hallazgos_cualitativos']}")

        detalle_extra = "\n- ".join(extras)
        bloques.append(
            f"[RS_ID: {r['rs_id']} | Año: {r['anio_publicacion']} | Tipo: {r.get('tipo_documento', 'RS')} | Sección: {r['seccion_rs']} | "
            f"Subtipo: {r['subtipo_ataxia']} | Dominio: {r['dominio_clinico']}]\n"
            f"- Intervención/Herramienta: {r['intervencion_evaluada']} ({r['herramientas_evaluacion']})\n"
            f"- Nivel de evidencia: {r['nivel_evidencia']} | Limitaciones: {r['limitaciones']}\n"
            + (f"- {detalle_extra}\n" if detalle_extra else "")
            + f"- Síntesis: {r['contenido_chunk']}"
        )
    return "\n\n".join(bloques)


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


# 3. INTERFAZ VISUAL DE TRES PANELES
st.set_page_config(page_title="LogoPilot | CDSS Logopedia Neurológica", page_icon="🧭", layout="wide")

archivos_img = glob.glob("**/*.png", recursive=True) + glob.glob("**/*.jpg", recursive=True) + glob.glob("**/*.jpeg", recursive=True)
img_b64 = base64.b64encode(open(archivos_img[0], "rb").read()).decode() if archivos_img else ""

st.markdown(f"""
<style>
    .block-container {{ padding-top: 3.2rem; padding-bottom: 2.5rem; }}
    .header-cdss {{
        background: linear-gradient(90deg, #FFFFFF 0%, #FFFFFF 21%, #2563EB 32%, #1E3A8A 65%, #0F172A 100%);
        padding: 1.4rem 2.2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
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
    }}
    .header-contenido {{
        flex: 1;
        display: flex;
        flex-direction: column;
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
        align-self: flex-end;
        text-align: right;
        margin-top: 0.45rem;
    }}
    .header-cdss p {{
        margin: 0;
        font-size: 0.92rem;
        color: #F8FAFC;
        font-weight: 500;
        text-shadow: 0 1px 3px rgba(15, 23, 42, 0.55);
    }}
    .badge-rag {{
        display: inline-block;
        background-color: rgba(15, 23, 42, 0.45);
        color: #F8FAFC;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin-top: 0.45rem;
        font-weight: 600;
    }}
    div[data-testid="stExpander"] {{ border: 1px solid #CBD5E1; border-radius: 6px; background-color: #F8FAFC; }}
</style>
<div class="header-cdss">
    <img src="data:image/png;base64,{img_b64}" class="logo-fundido">
    <div class="header-contenido">
        <h2><span class="titulo-logo">LOGO</span><span class="titulo-pilot">-Pilot · Soporte Clínico para Logopedia Neurológica</span></h2>
        <div class="header-inferior">
            <p>Sistema de navegación y apoyo a la toma de decisiones en DCA, Parkinson y Ataxias Hereditarias</p>
            <span class="badge-rag">Corpus Auditado · Trazabilidad CitaRAG · Supervisión Humana (Art. 14 EU AI Act)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not os.getenv("OPENAI_API_KEY"):
    st.error("Falta configurar OPENAI_API_KEY en la terminal antes de iniciar.")
    st.stop()

indice = inicializar_indice()
col_izq, col_centro, col_der = st.columns([1, 1.4, 1])

with col_izq:
    c1, c1_i = st.columns([0.85, 0.15])
    c1.subheader("1. Contexto Clínico")
    with c1_i.popover("ℹ️"):
        st.caption("1. Elija fuente de datos.\n2. Verifique anonimización (sin nombres ni NHC).\n3. Pulse Analizar.")

    modo_ingreso = st.radio(
        "Modo de entrada de datos:",
        ["Banco Maestro (25 Casos)", "Subir Informe (PDF/DOCX/TXT/XLSX)", "Ficha Caso Real (Manual)"],
        on_change=limpiar_estado_anterior
    )

    opciones_pat = [p.value for p in PatologiaBase]
    idx_pat = 0
    texto_auto = ""

    if modo_ingreso == "Banco Maestro (25 Casos)":
        casos_dict = {"-- Selecciona un caso --": None}
        archivo_maestro = "LOGO_LLM_25_Casos_Clinicos_Sinteticos_Banco_Maestro_v1_0.json"
        if os.path.exists(archivo_maestro):
            with open(archivo_maestro, "r", encoding="utf-8") as f:
                for c in json.load(f).get("casos", []):
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
    contexto_rs = recuperar_evidencia_rs(pat_sel, trast_sel)

    prompt = (
        "Eres un Sistema de Soporte a la Decisión Clínica (CDSS) experto en Logopedia Neurológica. "
        "Redacta un informe técnico amplio, detallado y riguroso para un profesional logopeda, "
        "basándote ESTRICTAMENTE en dos capas jerárquicas de conocimiento:\n\n"
        "=== CAPA 1: PROTOCOLOS Y GUÍAS CLÍNICAS (CÓMO HACER / PRÁCTICA SEGURA) ===\n"
        f"{contexto}\n\n"
        "=== CAPA 2: REVISIONES SISTEMÁTICAS Y ESTUDIOS PRIMARIOS (NIVEL DE EVIDENCIA Y ESTADÍSTICA) ===\n"
        f"{contexto_rs}\n\n"
        f"DATOS DEL CASO CLÍNICO:\n{consulta}\n\n"
        "REGLAS DE RAZONAMIENTO CLÍNICO Y TRAZABILIDAD (OBLIGATORIAS):\n"
        "1. SEPARAR EFICACIA ESTADÍSTICA DE PRÁCTICA CLÍNICA: Nunca descartes una intervención útil de la Capa 1 solo porque una Revisión Sistemática (ej. Cochrane) indique certeza baja o ausencia de ECA. Primero declara explícitamente el grado de respaldo científico, tamaño muestral/estadísticos y riesgo de sesgo según la Capa 2, y a continuación desarrolla el abordaje clínico recomendado y seguro según los manuales y protocolos de la Capa 1.\n"
        "2. RESOLUCIÓN DE CONFLICTOS TEMPORALES: Si un estudio primario o ensayo reciente (ej. terapia intensiva LSVT o miofuncional en SCA3) muestra resultados prometedores que matizan o contradicen una RS más antigua, prioriza el hallazgo cronológicamente posterior aclarando expresamente que se trata de evidencia preliminar aún no incorporada a una revisión sistemática.\n"
        "3. TRAZABILIDAD EXPLÍCITA: En el texto y en 'evidencia_rag', diferencia siempre qué recomendación proviene de [Guía Clínica / Manual] y qué dato proviene de [Revisión Sistemática / Estudio Primario], citando su ID exacto (sin prefijos como 'RS_ID:').\n"
        "4. SEGURIDAD Y DATOS AUSENTES: Si hay sospecha de disfagia, indica en 'adaptacion_dietetica_iddsi' que no se prescriben viscosidades IDDSI sin valoración instrumental (FEES/VFSS); si no hay disfagia, déjalo en ''. Declara en 'informacion_clinica_no_disponible' los datos ausentes y en 'evidencia_cientifica_insuficiente' las limitaciones metodológicas y riesgos de sesgo detectados en la Capa 2."
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
        pop_rag.markdown("**¿Cómo funciona la trazabilidad CitaRAG + Capa RS?**")
        pop_rag.caption(
            "• **📘 Guía/Manual:** Protocolos clínicos auditados (cómo hacer).\n"
            "• **📊 Rev. Sistemática:** Grado de certeza, estadísticos y semáforo de riesgo de sesgo (🟢 Bajo / 🔵 Moderado / 🔴 Alto).\n"
            "• **Auditoría directa:** Descarga el PDF original de la carpeta de la patología."
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

            # Buscar metadatos de RS si aplica
            meta_rs = mapa_rs_meta.get(id_limpio)
            if not meta_rs and cita.tipo_fuente != "Guía Clínica / Manual":
                prefijo_rs = "-".join(id_limpio.split("-")[:2])
                for rs_k, m_val in mapa_rs_meta.items():
                    if rs_k == id_limpio or (prefijo_rs and rs_k.startswith(prefijo_rs)):
                        meta_rs = m_val
                        break

            if cita.tipo_fuente == "Guía Clínica / Manual":
                icono_fuente = "📘 [Guía/Manual]"
                sem_emoji = ""
            else:
                icono_fuente = "📊 [Rev. Sistemática]"
                r_sesgo = meta_rs.get("riesgo_sesgo", "") if meta_rs else ""
                n_evid = meta_rs.get("nivel_evidencia", "") if meta_rs else ""
                sem_emoji, sem_texto, sem_bg, sem_fg = calcular_semaforo_evidencia(r_sesgo, n_evid)
                icono_fuente = f"{sem_emoji} 📊 [Rev. Sistemática]"

            with st.expander(f"{icono_fuente} {id_limpio} · {doc_id_limpio}", expanded=True):
                st.markdown(f"**{titulo_doc}**")
                if cita.tipo_fuente != "Guía Clínica / Manual" and meta_rs:
                    r_sesgo = meta_rs.get("riesgo_sesgo", "No especificado")
                    n_evid = meta_rs.get("nivel_evidencia", "No especificado")
                    sem_emoji, sem_texto, sem_bg, sem_fg = calcular_semaforo_evidencia(r_sesgo, n_evid)
                    st.markdown(
                        f'<div style="background-color:{sem_bg}; color:{sem_fg}; padding:4px 10px; border-radius:6px; font-size:0.82rem; font-weight:600; margin-bottom:6px;">'
                        f'{sem_emoji} {sem_texto} · Sesgo: {r_sesgo} | Certeza: {n_evid}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.caption(f"Localizador: {cita.fuente_documental}")
                st.write(cita.hecho_clinico_factual)

                doi = str(info_doc.get("doi", "")).strip()
                if doi.startswith("10."):
                    st.markdown(f"[Abrir publicación oficial (DOI: {doi})](https://doi.org/{doi})")

                nombre_pdf = info_doc.get("archivo_pdf", "")
                if not nombre_pdf:
                    nombre_pdf = mapa_rs_pdfs.get(id_limpio, "")
                    if not nombre_pdf:
                        prefijo_rs = "-".join(id_limpio.split("-")[:2])
                        for rs_k, arch_v in mapa_rs_pdfs.items():
                            if rs_k == id_limpio or (prefijo_rs and rs_k.startswith(prefijo_rs)):
                                nombre_pdf = arch_v
                                break

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
