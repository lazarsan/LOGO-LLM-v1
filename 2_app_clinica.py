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

# 3. INTERFAZ VISUAL DE TRES PANELES
st.set_page_config(page_title="LOGO-LLM | CDSS Logopedia Neurológica", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .header-cdss {
        background: linear-gradient(90deg, #0F2C59 0%, #1E4E8C 100%);
        color: white;
        padding: 1.1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        border-left: 6px solid #38BDF8;
    }
    .header-cdss h2 { margin: 0; font-size: 1.45rem; font-weight: 700; letter-spacing: 0.3px; }
    .header-cdss p { margin: 0.3rem 0 0 0; font-size: 0.88rem; color: #E2E8F0; }
    .badge-rag {
        display: inline-block;
        background-color: rgba(255,255,255,0.15);
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        background-color: #F8FAFC;
    }
</style>
<div class="header-cdss">
    <h2>LOGO-LLM · Sistema de Soporte a la Decisión Clínica (CDSS)</h2>
    <p>Logopedia Neurológica Basada en Evidencia · Daño Cerebral Adquirido, Parkinson y Ataxias Hereditarias</p>
    <span class="badge-rag">Corpus Auditado: 184 Hechos Factuales · Trazabilidad CitaRAG · Supervisión Humana (Art. 14 EU AI Act)</span>
</div>
""", unsafe_allow_html=True)

if not os.getenv("OPENAI_API_KEY"):
    st.error("Falta configurar OPENAI_API_KEY en la terminal antes de iniciar.")
    st.stop()

indice = inicializar_indice()
col_izq, col_centro, col_der = st.columns([1, 1.4, 1])

with col_izq:
    st.subheader("1. Contexto Clínico")

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

    reporte = None
    for intento in range(1, 4):
        try:
            llm_activo = LlamaOpenAI(model="gpt-4o", temperature=0.1 if intento == 1 else 0.0)
            reporte = llm_activo.as_structured_llm(ReporteClinicoLogopedico).complete(prompt).raw
            break
        except ValidationError as e:
            prompt += f"\nCORRECCIÓN (Intento {intento}): {e}"

    if reporte:
        with col_centro:
            st.subheader("2. Reporte Clínico")
            st.info(f"**CIE-11:** `{reporte.diagnostico.codigo_cie11}` | **{reporte.diagnostico.patologia_neuro_base.value}**")
            
            st.markdown("**Análisis Fisiopatológico:**")
            st.write(reporte.diagnostico.analisis_fisiopatologico)
            st.caption(f"**Diagnóstico Diferencial:** {reporte.diagnostico.diagnostico_diferencial}")

            st.error(f"**Resolución del Dilema Clínico:** {reporte.intervencion.respuesta_dilema_clinico}")

            if reporte.evaluacion.signos_alerta_seguridad:
                st.warning("**Alertas de Seguridad:**\n" + "\n".join([f"- {a}" for a in reporte.evaluacion.signos_alerta_seguridad]))
            
            if reporte.evaluacion.informacion_clinica_no_disponible:
                st.info("**Información Clínica No Disponible (Datos Ausentes):**\n" + "\n".join([f"- {d}" for d in reporte.evaluacion.informacion_clinica_no_disponible]))

            if reporte.evaluacion.evidencia_cientifica_insuficiente:
                st.warning("**Declaración de Evidencia Científica Insuficiente:**\n" + "\n".join([f"- {ev}" for ev in reporte.evaluacion.evidencia_cientifica_insuficiente]))

            st.markdown("**Evaluación Sugerida:**")
            for esc in reporte.evaluacion.escalas_sugeridas:
                st.checkbox(esc, value=True, key=f"e_{esc}")

            if reporte.evaluacion.criterios_derivacion:
                st.markdown("**Derivación Multidisciplinar:** " + ", ".join(reporte.evaluacion.criterios_derivacion))

            st.markdown("**Objetivos Terapéuticos:**")
            for obj in reporte.intervencion.objetivos_terapeuticos:
                st.markdown(f"- {obj}")

            st.markdown("**Intervención Propuesta:**")
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
                f"REPORTE CLÍNICO LOGO-LLM\n"
                f"CIE-11: {reporte.diagnostico.codigo_cie11} | Patología: {reporte.diagnostico.patologia_neuro_base.value}\n\n"
                f"ANÁLISIS FISIOPATOLÓGICO:\n{reporte.diagnostico.analisis_fisiopatologico}\n\n"
                f"DIAGNÓSTICO DIFERENCIAL:\n{reporte.diagnostico.diagnostico_diferencial}\n\n"
                f"RESOLUCIÓN DEL DILEMA:\n{reporte.intervencion.respuesta_dilema_clinico}\n\n"
                f"PAUTA IDDSI:\n{reporte.intervencion.adaptacion_dietetica_iddsi}"
            )
            st.download_button("Validar y Descargar para Historia Clínica", data=resumen_txt, file_name="reporte_logopedia.txt", use_container_width=True)

        with col_der:
            st.subheader("3. Inspector CitaRAG")
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


