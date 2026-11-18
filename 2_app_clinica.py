import json
import os
from enum import Enum
from typing import List, Optional
import streamlit as st
from lector_informes import extraer_texto_archivo
from ficha_clinica_real import renderizar_ficha_real
import unicodedata
def normalizar_nombre(texto: str) -> str:
    # Elimina acentos, espacios extra y convierte a minúsculas para comparar sin fallos
    sin_acentos = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return sin_acentos.lower().replace("_", " ").replace("-", " ").strip()
from pydantic import BaseModel, Field, model_validator, ValidationError
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

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
    informacion_clinica_no_disponible: List[str] = Field(description="Datos ausentes en la historia del paciente que impiden confirmar diagnósticos o riesgos (ej. ausencia de FEES/VFSS, estado ON/OFF no registrado, sintaxis compleja no                   evaluada).")
    evidencia_cientifica_insuficiente: List[str] = Field(description="Áreas donde la literatura recuperada tiene certeza baja/muy baja, falta de ensayos clínicos o riesgo de extrapolación entre fenotipos.")
    criterios_derivacion: List[str] = Field(description="Especialidades a las que derivar (ej. Neurología, ORL, Nutrición, Farmacia).")
    prueba_instrumental_previa: bool = Field(description="True solo si el caso indica FEES o VFSS ya realizados.")

class DiagnosticoEstructurado(BaseModel):
    patologia_neuro_base: PatologiaBase
    trastorno_logopedico: TipoTrastorno
    codigo_cie11: str = Field(description="Código CIE-11 principal y secundarios si hay comorbilidad.")
    analisis_fisiopatologico: str = Field(description="Explicación detallada de los subsistemas afectados (respiración, fonación, articulación, resonancia, prosodia, deglución o lenguaje).")
    diagnostico_diferencial: str = Field(description="Diferenciación clínica frente a cuadros similares y límites de certeza diagnóstica.")

class IntervencionLogopedica(BaseModel):
    respuesta_dilema_clinico: str = Field(description="Respuesta directa, argumentada y segura frente a la propuesta o trampa planteada en la consulta.")
    objetivos_terapeuticos: List[str] = Field(description="Objetivos funcionales y comunicativos/deglutorios individualizados.")
    estrategias_compensatorias: List[str] = Field(description="Maniobras deglutorias, postura, CAA o ajustes del entorno detallados.")
    reactivacion_neuromuscular: List[str] = Field(description="Procedimientos motores o programas validados con sus parámetros de dosificación y fatiga.")
    adaptacion_dietetica_iddsi: str = Field(description="Pauta detallada sobre viscosidades/texturas IDDSI y condiciones obligatorias previas.")

class CitaRAG(BaseModel):
    id_nodo: str = Field(description="ID exacto del claim (ej. DCA-C001, EP-CLM-002, ATX-CL-007).")
    fuente_documental: str = Field(description="ID del documento y localizador físico PDF.")
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

# 2. CARGA DEL ÍNDICE VECTORIAL
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

import urllib.request
import urllib.parse

import urllib.request
import urllib.parse

def buscar_evidencia_externa(patologia: str, trastorno: str) -> dict:
    mapa_en = {
        "Daño Cerebral Adquirido": "stroke OR brain injury",
        "Enfermedad de Parkinson": "Parkinson disease",
        "Ataxia Hereditaria": "ataxia OR Friedreich",
        "Disfagia Neurogénica": "dysphagia OR swallowing",
        "Disartria": "dysarthria OR speech therapy",
        "Afasia": "aphasia OR language therapy",
        "Dispraxia / Apraxia del Habla / Orofacial": "apraxia of speech"
    }
    q_pat = mapa_en.get(patologia, patologia)
    q_trast = mapa_en.get(trastorno, trastorno)
    query_bool = f"({q_pat}) AND ({q_trast})"
    query_enc = urllib.parse.quote(query_bool)
    headers = {"User-Agent": "Mozilla/5.0 (LOGO-Pilot Clinical Research)"}

    texto_recuperado = ""

    # 1. Consulta PubMed / MEDLINE / Cochrane (ampliada a ECA, Revisiones, Guías y Ensayos)
    filtro_tipo = "(Randomized Controlled Trial[pt] OR Systematic Review[pt] OR Clinical Trial[pt] OR Guideline[pt] OR \"Cochrane Database Syst Rev\"[Journal])"
    query_pm = urllib.parse.quote(f"{query_bool} AND {filtro_tipo}")
    url_pm = f"https://nih.gov{query_pm}&retmode=json&retmax=5&sort=relevance"
    try:
        req_pm = urllib.request.Request(url_pm, headers=headers)
        ids = json.loads(urllib.request.urlopen(req_pm, timeout=8).read().decode()).get("esearchresult", {}).get("idlist", [])
        if ids:
            url_fetch = f"https://nih.gov{','.join(ids)}&retmode=xml"
            req_fetch = urllib.request.Request(url_fetch, headers=headers)
            texto_recuperado += "=== PUBMED / MEDLINE / COCHRANE ===\n" + urllib.request.urlopen(req_fetch, timeout=8).read().decode("utf-8", errors="ignore")[:10000]
    except Exception:
        pass

    # 2. Consulta API ClinicalTrials.gov v2 con User-Agent
    url_ct = f"https://clinicaltrials.gov{urllib.parse.quote(q_pat)}&query.term={urllib.parse.quote(q_trast)}&pageSize=3"
    try:
        req_ct = urllib.request.Request(url_ct, headers=headers)
        datos_ct = json.loads(urllib.request.urlopen(req_ct, timeout=8).read().decode())
        estudios = datos_ct.get("studies", [])
        if estudios:
            texto_recuperado += "\n=== CLINICALTRIALS.GOV ===\n"
            for est in estudios:
                prot = est.get("protocolSection", {})
                nct = prot.get("identificationModule", {}).get("nctId", "")
                tit = prot.get("identificationModule", {}).get("briefTitle", "")
                estado = prot.get("statusModule", {}).get("overallStatus", "")
                texto_recuperado += f"ID: {nct} | URL: https://clinicaltrials.gov{nct} | Estado: {estado} | Título: {tit}\n"
    except Exception:
        pass

    enlaces = {
        "PubMed Central (PMC · Texto Completo)": f"https://nih.gov{query_enc}",
        "Cochrane Library": f"https://cochranelibrary.com{query_enc}",
        "speechBITE (Logopedia)": f"https://speechbite.com{urllib.parse.quote(q_trast)}",
        "ASHAWire (Revistas ASHA)": f"https://asha.org{query_enc}",
        "ASHA Evidence Maps": "https://asha.org"
    }

    return {"datos_api": texto_recuperado, "enlaces": enlaces}

    ...

# 3. INTERFAZ VISUAL DE TRES PANELES
st.set_page_config(page_title="LogoPilot | CDSS Logopedia Neurológica", page_icon="🧭", layout="wide")

import base64

import glob
archivos_img = glob.glob("**/*.png", recursive=True) + glob.glob("**/*.jpg", recursive=True) + glob.glob("**/*.jpeg", recursive=True)
img_b64 = base64.b64encode(open(archivos_img[0], "rb").read()).decode() if archivos_img else ""

st.markdown(f"""
<style>
    .block-container {{ padding-top: 3.2rem; padding-bottom: 2.5rem; }}
    .header-cdss {{
        background: linear-gradient(90deg, #1E1B4B 0%, #4C1D95 100%);
        color: white;
        padding: 1.8rem 2.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #A855F7;
        box-shadow: 0 4px 12px rgba(30, 27, 75, 0.2);
    }}
    .header-cdss h2 {{ margin: 0; font-size: 1.5rem; font-weight: 700; letter-spacing: 0.3px; }}
    .header-cdss p {{ margin: 0.3rem 0 0 0; font-size: 0.9rem; color: #E2E8F0; }}
    .badge-rag {{
        display: inline-block;
        background-color: rgba(255,255,255,0.15);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin-top: 0.5rem;
        font-weight: 600;
    }}
    div[data-testid="stExpander"] {{ border: 1px solid #CBD5E1; border-radius: 6px; background-color: #F8FAFC; }}
</style>
<div class="header-cdss" style="display: flex; align-items: center; gap: 1.5rem;">
    <img src="data:image/png;base64,{img_b64}" width="76" style="background: white; padding: 6px; border-radius: 12px; flex-shrink: 0;">
    <div>
        <h2>LOGO-Pilot · Soporte Clínico para Logopedia Neurológica</h2>
        <p>Sistema de navegación y apoyo a la toma de decisiones en DCA, Parkinson y Ataxias Hereditarias</p>
        <span class="badge-rag">Corpus Auditado · Trazabilidad CitaRAG · Supervisión Humana (Art. 14 EU AI Act)</span>
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

    # 1. Selector de las 3 vías de entrada
    modo_ingreso = st.radio(
        "Modo de entrada de datos:",
        ["Banco Maestro (25 Casos)", "Subir Informe (PDF/DOCX/TXT/XLSX)", "Ficha Caso Real (Manual)"]
    )

    opciones_pat = [p.value for p in PatologiaBase]
    idx_pat = 0
    texto_auto = ""

    # VÍA 1: Desplegable del Banco Maestro
    if modo_ingreso == "Banco Maestro (25 Casos)":
        casos_dict = {"-- Selecciona un caso --": None}
        archivo_maestro = "LOGO_LLM_25_Casos_Clinicos_Sinteticos_Banco_Maestro_v1_0.json"
        if os.path.exists(archivo_maestro):
            with open(archivo_maestro, "r", encoding="utf-8") as f:
                for c in json.load(f).get("casos", []):
                    cid = c.get("id_caso") or c.get("id") or "CASO"
                    tit = c.get("titulo") or c.get("perfil") or ""
                    casos_dict[f"{cid} · {tit}"] = c

        seleccion = st.selectbox("Selecciona caso sintético:", list(casos_dict.keys()))
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

    # VÍA 2: Subida de archivos locales
    elif modo_ingreso == "Subir Informe (PDF/DOCX/TXT/XLSX)":
        archivo_informe = st.file_uploader("Sube el informe clínico:", type=["pdf", "docx", "txt", "xlsx"])
        if archivo_informe:
            texto_auto = extraer_texto_archivo(archivo_informe)
            st.success("Informe cargado y anonimizado localmente.")

    # VÍA 3: Ficha estructurada manual de 8 bloques
    else:
        texto_auto = renderizar_ficha_real()

    # 2. Parámetros fijos y botón de ejecución (siempre visibles abajo)
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

    prompt = (
        "Eres un Sistema de Soporte a la Decisión Clínica (CDSS) experto en Logopedia Neurológica. "
        "Redacta un informe técnico amplio, detallado y riguroso para un profesional logopeda, "
        "basándote ESTRICTAMENTE en los hechos clínicos del siguiente contexto:\n\n"
        f"{contexto}\n\n"
        f"DATOS DEL CASO CLÍNICO:\n{consulta}\n\n"
        "INSTRUCCIONES DE DETALLE Y SEGURIDAD:\n"
        "1. Desarrolla de forma exhaustiva el análisis fisiopatológico por subsistemas y el diagnóstico diferencial.\n"
        "2. Responde explícitamente en 'respuesta_dilema_clinico' por qué la propuesta planteada en la consulta es adecuada o contraindicada.\n"
        "3. Detalla la dosificación, frecuencia y control de fatiga en cada procedimiento de intervención.\n"
        "4. Si el caso involucra disfagia o sospecha de alteración deglutoria, indica en 'adaptacion_dietetica_iddsi' que no se prescriben viscosidades IDDSI sin valoración instrumental previa (FEES/VFSS); si no hay sospecha de disfagia, deja 'adaptacion_dietetica_iddsi' como cadena vacía ''.\n"
        "5. Declara explícitamente en 'informacion_clinica_no_disponible' todo dato ausente en el caso y en 'evidencia_cientifica_insuficiente' cualquier limitación metodológica o grado de certeza bajo de las fuentes."
    )

    for intento in range(1, 4):
        try:
            llm_activo = LlamaOpenAI(model="gpt-4o", temperature=0.1 if intento == 1 else 0.0)
            st.session_state["reporte"] = llm_activo.as_structured_llm(ReporteClinicoLogopedico).complete(prompt).raw
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
            pop_cie.markdown("- **8A00.0:** Enfermedad de Parkinson.\n- **8A02.0:** Ataxias hereditarias.\n- **8B11 / 8B20:** Ictus isquémico/hemorrágico.")
            pop_cie.markdown("[Consultar navegador OMS CIE-11](https://who.int)")

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
                                f"Indica la base de origen (PubMed, Cochrane o ClinicalTrials NCT). No inventes datos:\n\n{res_ext['datos_api']}"
                                "Incluye en 'referencia_estudio' el autor, año y el enlace markdown clicable al PMID (https://nih.gov) o NCT (https://clinicaltrials.gov...)."
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
                "• **Recuperación determinista:** Extrae únicamente los hechos factuales más cercanos semánticamente de los 184 nodos auditados.\n"
                "• **Cero alucinaciones:** Cada afirmación está anclada a su `claim_id` y página exacta.\n"
                "• **Auditoría directa:** Usa el enlace **DOI** para verificar la fuente primaria original."
            )

            mapa_fuentes = {}
            if os.path.exists("mapa_fuentes.json"):
                with open("mapa_fuentes.json", "r", encoding="utf-8") as mf:
                    mapa_fuentes = json.load(mf)

            carpeta_referencias = r"F:\VIU\IA SLP\Proyecto LOGO-LLM\Referencias"

            for idx, cita in enumerate(reporte.evidencia_rag):
                texto_busqueda = f"{cita.id_nodo} {cita.fuente_documental}".upper()
                doc_id_limpio = next((k for k in mapa_fuentes.keys() if k.upper() in texto_busqueda), cita.fuente_documental)
                info_doc = mapa_fuentes.get(doc_id_limpio, {})
                titulo_doc = info_doc.get("titulo", cita.fuente_documental)

                with st.expander(f"{cita.id_nodo} · {doc_id_limpio}", expanded=True):
                    st.markdown(f"**{titulo_doc}**")
                    st.caption(f"Localizador: {cita.fuente_documental}")
                    st.write(cita.hecho_clinico_factual)

                    doi = str(info_doc.get("doi", "")).strip()
                    if doi.startswith("10."):
                        st.markdown(f"[Abrir publicación oficial (DOI: {doi})](https://doi.org{doi})")

                    nombre_pdf = info_doc.get("archivo_pdf", "")
                    ruta_encontrada = None
                    if nombre_pdf and os.path.exists(carpeta_referencias):
                        for raiz, _, archivos in os.walk(carpeta_referencias):
                            if nombre_pdf in archivos:
                                ruta_encontrada = os.path.join(raiz, nombre_pdf)
                                break

                    if ruta_encontrada:
                        with open(ruta_encontrada, "rb") as pdf_file:
                            st.download_button(
                                label="Abrir / Descargar PDF original",
                                data=pdf_file.read(),
                                file_name=nombre_pdf,
                                mime="application/pdf",
                                key=f"pdf_{cita.id_nodo}_{idx}"
                            )


