import json
import os
from typing import List, Literal
import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI
from pypdf import PdfReader


class ChunkEvidencia(BaseModel):
    rs_id: str = Field(description="Identificador único del fragmento (ej. COCHRANE-2014-ATX-RESUMEN o NCT0123456-RES).")
    tipo_documento: Literal[
        "Revisión Sistemática / Metaanálisis",
        "Ensayo Clínico (ECA / Clinical Trial)",
        "Artículo Científico Primario / Observacional"
    ]
    seccion_rs: Literal["resumen_conclusiones", "tabla_estudio_primario", "discusion_practica_clinica"]
    prioridad_recuperacion: Literal["alta", "media"]
    anio_publicacion: int
    patologia: str
    subtipo_ataxia: str = Field(description="Subtipo específico (ej. SCA1, SCA3, Ataxia de Friedreich, Mixta o No aplica).")
    dominio_clinico: Literal["Evaluación", "Disartria", "Disfagia Neurogénica", "Afasia", "Dispraxia", "General"]
    intervencion_evaluada: str = Field(description="Descripción detallada de la intervención, exposición o protocolo activo (dosis, frecuencia, duración).")
    comparadores: str = Field(description="Grupo control, placebo, terapia estándar o comparador activo empleado; indicar 'No aplica' si no existe.")
    desenlaces_primarios_secundarios: str = Field(description="Desenlaces primarios y secundarios evaluados en el estudio o revisión.")
    herramientas_evaluacion: str = Field(description="Instrumentos, escalas clínicas o pruebas instrumentales empleadas y momentos de medición (basal, post-intervención, seguimiento).")
    variables_continuas: str = Field(description="Tamaño muestral (n) por grupos, media y desviación estándar (DE), mediana y rango intercuartílico (RIC), error estándar (EE) y diferencias de medias, si aplica.")
    variables_categoricas: str = Field(description="Número de eventos, proporciones/porcentajes, Odds Ratio (OR), Riesgo Relativo (RR) con intervalos de confianza (IC 95%) y valores p, si aplica.")
    hallazgos_cualitativos_citas: str = Field(description="Categorías temáticas, citas textuales relevantes o hallazgos cualitativos/narrativos clave.")
    riesgo_sesgo_calidad: str = Field(description="Valoración del riesgo de sesgo o calidad metodológica de cada estudio o global (herramientas Cochrane RoB 2, ROBINS-I, JBI o GRADE).")
    nivel_evidencia: str = Field(description="Nivel de certeza global o diseño metodológico (ej. Nivel I ECA, Nivel IV Serie de casos, Certeza GRADE muy baja).")
    limitaciones: str = Field(description="Limitaciones metodológicas reportadas (tamaño muestral, heterogeneidad, falta de cegamiento, seguimiento corto).")
    contenido_chunk: str = Field(description="Síntesis clínica factual completa del chunk integrando los datos PICO, estadísticos y metodológicos.")


class ExtraccionEvidencia(BaseModel):
    chunks: List[ChunkEvidencia]


st.set_page_config(page_title="Extractor de Evidencia | LOGO-Pilot", page_icon="📚", layout="wide")

st.markdown("""
<div style="background: linear-gradient(90deg, #1E1B4B 0%, #4C1D95 100%); color: white; padding: 1.4rem 1.8rem; border-radius: 10px; margin-bottom: 1.2rem; border-left: 6px solid #A855F7;">
    <h2 style="margin:0; font-size:1.35rem;">📚 Extractor Estructurado de Evidencia Científica (RS · Clinical Trials · Artículos)</h2>
    <p style="margin:0.3rem 0 0 0; font-size:0.88rem; color:#E2E8F0;">Genera y actualiza <code>capa_evidencia_rs.json</code> con extracción PICO, estadística completa y riesgo de sesgo (Cochrane / JBI)</p>
</div>
""", unsafe_allow_html=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Falta configurar OPENAI_API_KEY en Secrets de Streamlit Cloud o en el entorno local.")
    st.stop()

if "registros_rs" not in st.session_state:
    if os.path.exists("capa_evidencia_rs.json"):
        with open("capa_evidencia_rs.json", "r", encoding="utf-8") as f:
            st.session_state["registros_rs"] = json.load(f)
    else:
        st.session_state["registros_rs"] = []

col1, col2 = st.columns([1, 1.35])

with col1:
    st.subheader("1. Cargar Documento Científico")
    archivo_pdf = st.file_uploader("Selecciona el PDF (Revisión Sistemática, Clinical Trial o Artículo):", type=["pdf"])

    tipo_doc = st.selectbox(
        "Tipo de publicación:",
        [
            "Revisión Sistemática / Metaanálisis",
            "Ensayo Clínico (ECA / Clinical Trial)",
            "Artículo Científico Primario / Observacional"
        ]
    )

    id_base = st.text_input("Identificador corto (sin espacios):", placeholder="Ej. COCHRANE-2014-ATX o NCT0451234-SCA3")
    anio_pub = st.number_input("Año de publicación:", min_value=1980, max_value=2030, value=2024, step=1)
    patologia_sel = st.selectbox(
        "Patología asociada:",
        ["Ataxia Hereditaria", "Enfermedad de Parkinson", "Daño Cerebral Adquirido"]
    )

    procesar = st.button("⚙️ Extraer Metadatos y Fragmentar Documento", type="primary", use_container_width=True)

    if procesar:
        if not archivo_pdf or not id_base.strip():
            st.warning("Por favor, sube un archivo PDF y escribe un identificador corto.")
        else:
            with st.spinner("Leyendo PDF y extrayendo variables PICO, estadísticas y riesgo de sesgo..."):
                lector = PdfReader(archivo_pdf)
                paginas = []
                for i, pag in enumerate(lector.pages):
                    txt = pag.extract_text() or ""
                    if txt.strip():
                        paginas.append(f"--- PÁGINA {i + 1} ---\n{txt}")
                texto_completo = "\n\n".join(paginas)[:85000]

                if tipo_doc == "Revisión Sistemática / Metaanálisis":
                    instruccion_fragmentacion = (
                        "1. 'resumen_conclusiones' (prioridad 'alta'): Un único chunk con la síntesis global y conclusiones de los autores.\n"
                        "2. 'tabla_estudio_primario' (prioridad 'media'): Un chunk independiente por CADA estudio incluido en las tablas de características o riesgo de sesgo, manteniendo el título de la RS como contexto.\n"
                        "3. 'discusion_practica_clinica' (prioridad 'media'): Fragmentos separados por subdominio clínico (Evaluación, Disartria, Disfagia Neurogénica, etc.) conectando la evidencia con la práctica clínica."
                    )
                else:
                    instruccion_fragmentacion = (
                        "1. 'resumen_conclusiones' (prioridad 'alta'): Un único chunk global con el diseño del estudio, objetivo, población, resumen estadístico y conclusión principal.\n"
                        "2. 'tabla_estudio_primario' (prioridad 'media'): Un chunk detallado con los resultados cuantitativos por grupo/desenlace (tamaños muestrales, medias/DE, RIC, EE, eventos, OR/RR, IC 95%, valores p) y valoración de riesgo de sesgo/calidad.\n"
                        "3. 'discusion_practica_clinica' (prioridad 'media'): Fragmentos por subdominio clínico con aplicabilidad, dosificación e implicaciones clínicas."
                    )

                prompt_sys = (
                    "Eres un metodólogo experto en síntesis de evidencia clínica y logopedia neurológica. "
                    "Extrae los datos del documento de forma exhaustiva, objetiva y sin inventar cifras. "
                    "Si un dato estadístico (ej. RIC, EE, OR o RR) no aparece en el texto, indica explícitamente 'No reportado'.\n\n"
                    f"ESTRATEGIA DE FRAGMENTACIÓN OBLIGATORIA:\n{instruccion_fragmentacion}"
                )

                prompt_usr = (
                    f"ID BASE: {id_base.strip().upper()}\n"
                    f"TIPO DE DOCUMENTO: {tipo_doc}\n"
                    f"AÑO: {anio_pub}\n"
                    f"PATOLOGÍA: {patologia_sel}\n"
                    f"ARCHIVO ORIGEN: {archivo_pdf.name}\n\n"
                    f"TEXTO DEL DOCUMENTO:\n{texto_completo}"
                )

                client = OpenAI(api_key=api_key)
                resp = client.beta.chat.completions.parse(
                    model="gpt-4o",
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": prompt_sys},
                        {"role": "user", "content": prompt_usr}
                    ],
                    response_format=ExtraccionEvidencia
                )

                nuevos_chunks = []
                for idx_c, ch in enumerate(resp.choices[0].message.parsed.chunks, start=1):
                    item = ch.model_dump()
                    item["tipo_fuente"] = "revision_sistematica"
                    item["archivo_origen"] = archivo_pdf.name
                    if not item["rs_id"].startswith(id_base.strip().upper()):
                        item["rs_id"] = f"{id_base.strip().upper()}-{idx_c:02d}"
                    nuevos_chunks.append(item)

                prefijo = id_base.strip().upper()
                existentes = [r for r in st.session_state["registros_rs"] if not str(r.get("rs_id", "")).startswith(prefijo)]
                st.session_state["registros_rs"] = existentes + nuevos_chunks
                st.success(f"Se extrajeron {len(nuevos_chunks)} bloques estructurados de '{archivo_pdf.name}'.")

with col2:
    st.subheader(f"2. Base de Evidencia Acumulada ({len(st.session_state['registros_rs'])} fragmentos)")

    if st.session_state["registros_rs"]:
        json_str = json.dumps(st.session_state["registros_rs"], ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Descargar capa_evidencia_rs.json actualizado",
            data=json_str.encode("utf-8"),
            file_name="capa_evidencia_rs.json",
            mime="application/json",
            type="primary",
            use_container_width=True
        )

        st.divider()
        for reg in reversed(st.session_state["registros_rs"][-15:]):
            etiqueta = f"[{reg.get('tipo_documento', 'RS')}] {reg.get('rs_id')} · {reg.get('dominio_clinico')} ({reg.get('anio_publicacion')})"
            with st.expander(etiqueta, expanded=False):
                st.markdown(f"**Subtipo / Población:** {reg.get('subtipo_ataxia', '')}")
                st.markdown(f"**Intervención:** {reg.get('intervencion_evaluada', '')}")
                st.markdown(f"**Comparadores:** {reg.get('comparadores', '')}")
                st.markdown(f"**Desenlaces (1º y 2º):** {reg.get('desenlaces_primarios_secundarios', '')}")
                st.markdown(f"**Instrumentos y tiempos:** {reg.get('herramientas_evaluacion', '')}")
                st.markdown(f"**Variables continuas:** {reg.get('variables_continuas', '')}")
                st.markdown(f"**Variables categóricas:** {reg.get('variables_categoricas', '')}")
                st.markdown(f"**Hallazgos cualitativos / citas:** {reg.get('hallazgos_cualitativos_citas', '')}")
                st.markdown(f"**Riesgo de sesgo / Calidad (Cochrane/JBI):** {reg.get('riesgo_sesgo_calidad', '')}")
                st.info(reg.get("contenido_chunk", ""))
    else:
        st.info("Aún no hay fragmentos procesados. Sube un PDF en el panel izquierdo para comenzar.")
