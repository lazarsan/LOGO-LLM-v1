import json
import os
from typing import List, Literal
import streamlit as st
from pypdf import PdfReader
from pydantic import BaseModel, Field
from openai import OpenAI


def evaluar_semaforo(riesgo_sesgo: str, nivel_evidencia: str) -> tuple[str, str, str]:
    texto = f"{riesgo_sesgo} {nivel_evidencia}".lower()
    if any(k in texto for k in ["alto", "high", "muy baja", "muy bajo", "insuficiente", "crítico", "critico", "grave"]):
        return "🔴", "Alerta metodológica (Alto sesgo / Certeza muy baja)", "#FEE2E2"
    if any(k in texto for k in ["bajo riesgo", "low risk", "alta", "moderada", "moderado", "favorable", "solidez"]):
        return "🟢", "Solidez metodológica (Bajo sesgo / Certeza moderada-alta)", "#DCFCE7"
    return "🔵", "Precaución metodológica (Riesgo moderado / Certeza baja)", "#DBEAFE"


class ChunkEvidencia(BaseModel):
    rs_id: str = Field(description="Identificador único del fragmento (ej. COCHRANE-2014-ATX-RESUMEN, ECA-2023-SCA3-01).")
    seccion_rs: Literal["resumen_conclusiones", "tabla_estudio_primario", "discusion_practica_clinica"] = Field(
        description="Tipo de fragmento según estructura documental."
    )
    prioridad_recuperacion: Literal["alta", "media"] = Field(
        description="Alta para conclusiones globales/resúmenes; media para estudios individuales o discusión."
    )
    anio_publicacion: int = Field(description="Año de publicación del documento o estudio primario.")
    patologia: str = Field(description="Patología principal (Ataxia Hereditaria, Enfermedad de Parkinson, Daño Cerebral Adquirido).")
    subtipo_ataxia: str = Field(description="Subtipo clínico o genético exacto (ej. SCA3, Friedreich, Mixta, Ictus isquémico).")
    dominio_clinico: Literal["Evaluación", "Disartria", "Disfagia Neurogénica", "Afasia", "Dispraxia", "General"] = Field(
        description="Subdominio logopédico principal evaluado."
    )
    intervencion_evaluada: str = Field(description="Descripción detallada de la intervención o exposición (técnica, dosis, intensidad, frecuencia).")
    comparador: str = Field(description="Grupo control, placebo, terapia estándar o ausencia de comparador.")
    desenlaces_primarios_secundarios: str = Field(description="Desenlaces primarios y secundarios medidos en el estudio.")
    herramientas_evaluacion: str = Field(description="Instrumentos, escalas empleadas y momentos de medición (basal, post, seguimiento).")
    variables_continuas: str = Field(description="Tamaño muestral (n) por grupos, media y DE, mediana y RIC, error estándar (EE) si aplica.")
    variables_categoricas: str = Field(description="Número de eventos, proporciones, OR, RR con IC 95% y valores p si aplica.")
    hallazgos_cualitativos_citas: str = Field(description="Categorías temáticas, citas textuales o hallazgos cualitativos relevantes.")
    riesgo_sesgo_calidad: str = Field(description="Valoración del riesgo de sesgo o calidad metodológica (Cochrane RoB 2, ROBINS-I, JBI, GRADE).")
    nivel_evidencia: str = Field(description="Nivel de certeza global o diseño (ECA, Revisión Sistemática, Serie de casos, etc.).")
    limitaciones: str = Field(description="Limitaciones metodológicas principales declaradas o detectadas.")
    contenido_chunk: str = Field(description="Síntesis clínica factual clara y autocontenida del bloque.")


class ExtraccionDocumentoCientifico(BaseModel):
    chunks: List[ChunkEvidencia]


st.set_page_config(page_title="Extractor de Evidencia Científica | LOGO-Pilot", page_icon="📊", layout="wide")

st.title("📊 Extractor de Evidencia Científica y Meta-Capa (LOGO-Pilot)")
st.caption("Procesa Revisiones Sistemáticas, Ensayos Clínicos (Clinical Trials) y Artículos Primarios hacia capa_evidencia_rs.json")

api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.error("Falta configurar OPENAI_API_KEY en Secrets de Streamlit Cloud o en el entorno local.")
    st.stop()

if "acumulado_rs" not in st.session_state:
    if os.path.exists("capa_evidencia_rs.json"):
        with open("capa_evidencia_rs.json", "r", encoding="utf-8") as f:
            st.session_state["acumulado_rs"] = json.load(f)
    else:
        st.session_state["acumulado_rs"] = []

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("1. Cargar Documento Científico")
    archivo_pdf = st.file_uploader("Selecciona el PDF (RS, Clinical Trial o Artículo):", type=["pdf"])
    tipo_doc = st.selectbox(
        "Tipo de publicación:",
        [
            "Revisión Sistemática / Metaanálisis",
            "Ensayo Clínico (ECA / Clinical Trial)",
            "Artículo Científico Primario / Observacional"
        ]
    )
    id_base = st.text_input("Identificador corto sin espacios (ej. COCHRANE-2014-ATX):", value="")
    anio_doc = st.number_input("Año de publicación:", min_value=1990, max_value=2030, value=2024, step=1)
    patologia_doc = st.selectbox(
        "Patología neurológica asociada:",
        ["Ataxia Hereditaria", "Enfermedad de Parkinson", "Daño Cerebral Adquirido"]
    )
    reemplazar_existente = st.checkbox("Reemplazar bloques previos si ya existe este identificador", value=True)

    procesar = st.button("⚙️ Extraer Metadatos y Fragmentar Evidencia", type="primary", use_container_width=True)

if procesar and archivo_pdf and id_base.strip():
    with st.spinner("Leyendo PDF y extrayendo variables PICO, estadística y riesgo de sesgo..."):
        lector = PdfReader(archivo_pdf)
        paginas = []
        for i, pag in enumerate(lector.pages):
            txt = pag.extract_text() or ""
            paginas.append(f"--- PÁGINA {i + 1} ---\n{txt}")
        texto_completo = "\n\n".join(paginas)[:45000]

        id_limpio = id_base.strip().upper().replace(" ", "-")
        instruccion_tipo = (
            "Fragmenta siguiendo la matriz de Revisión Sistemática: 1) un chunk 'resumen_conclusiones' de alta prioridad; "
            "2) un chunk 'tabla_estudio_primario' por cada estudio incluido en tablas; "
            "3) chunks 'discusion_practica_clinica' separados por subdominio."
            if "Revisión" in tipo_doc
            else "Fragmenta el estudio primario / ensayo clínico extrayendo: 1) un chunk 'resumen_conclusiones' con diseño, objetivo y conclusión; "
                 "2) chunks 'tabla_estudio_primario' por cada grupo de intervención/desenlace principal; "
                 "3) chunks 'discusion_practica_clinica' con aplicabilidad y limitaciones."
        )

        prompt = (
            f"Eres un metodólogo experto en Medicina Basada en la Evidencia y Logopedia Neurológica. "
            f"Analiza este documento ({tipo_doc}, año {anio_doc}, patología: {patologia_doc}, ID base: {id_limpio}).\n\n"
            f"{instruccion_tipo}\n\n"
            "EXTRAE CON MÁXIMA PRECISIÓN:\n"
            "- Descripciones detalladas de intervenciones/exposición y comparadores.\n"
            "- Desenlaces primarios y secundarios, escalas/instrumentos empleados y momentos de medición.\n"
            "- Variables continuas: tamaño muestral (n) por grupos, media y desviación estándar (DE), mediana y rango intercuartílico (RIC), error estándar (EE).\n"
            "- Variables categóricas: número de eventos, proporciones, Odds Ratio (OR), Riesgo Relativo (RR) con IC 95% y valores p.\n"
            "- Categorías temáticas, citas o hallazgos cualitativos si existen.\n"
            "- Valoración explícita del riesgo de sesgo o calidad metodológica (herramientas Cochrane RoB 2, ROBINS-I, JBI o GRADE).\n"
            "Si algún parámetro estadístico no aparece en el texto, indica 'No reportado'.\n\n"
            f"TEXTO DEL DOCUMENTO:\n{texto_completo}"
        )

        cliente = OpenAI(api_key=api_key)
        respuesta = cliente.beta.chat.completions.parse(
            model="gpt-4o",
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtraccionDocumentoCientifico
        )
        resultado = respuesta.choices[0].message.parsed

        nuevos_registros = []
        for idx, ch in enumerate(resultado.chunks, start=1):
            item = ch.model_dump()
            if not item["rs_id"].startswith(id_limpio):
                item["rs_id"] = f"{id_limpio}-{idx:02d}"
            item["tipo_fuente"] = "revision_sistematica"
            item["tipo_documento_origen"] = tipo_doc
            item["archivo_origen"] = archivo_pdf.name
            icono, etiqueta_sem, _ = evaluar_semaforo(item["riesgo_sesgo_calidad"], item["nivel_evidencia"])
            item["semaforo_icono"] = icono
            item["semaforo_etiqueta"] = etiqueta_sem
            nuevos_registros.append(item)

        if reemplazar_existente:
            st.session_state["acumulado_rs"] = [
                r for r in st.session_state["acumulado_rs"]
                if not str(r.get("rs_id", "")).upper().startswith(id_limpio)
            ]
        st.session_state["acumulado_rs"].extend(nuevos_registros)
        st.success(f"Se extrajeron {len(nuevos_registros)} bloques estructurados de '{archivo_pdf.name}'.")

with col2:
    st.subheader(f"2. Vista Previa Evidencia ({len(st.session_state['acumulado_rs'])} bloques)")

    if st.session_state["acumulado_rs"]:
        json_str = json.dumps(st.session_state["acumulado_rs"], ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Descargar capa_evidencia_rs.json actualizado",
            data=json_str.encode("utf-8"),
            file_name="capa_evidencia_rs.json",
            mime="application/json",
            type="primary",
            use_container_width=True
        )

        for reg in reversed(st.session_state["acumulado_rs"][-15:]):
            icono, etiqueta_sem, color_bg = evaluar_semaforo(
                reg.get("riesgo_sesgo_calidad", ""),
                reg.get("nivel_evidencia", "")
            )
            with st.expander(f"{icono} {reg.get('rs_id')} · {reg.get('dominio_clinico')} ({reg.get('anio_publicacion')})", expanded=True):
                st.markdown(
                    f"<div style='background-color:{color_bg}; padding:6px 10px; border-radius:6px; font-weight:600; margin-bottom:8px;'>"
                    f"{icono} {etiqueta_sem}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Archivo origen:** `{reg.get('archivo_origen', 'N/A')}` | **Sección:** `{reg.get('seccion_rs')}`")
                st.markdown(f"**Intervención vs. Comparador:** {reg.get('intervencion_evaluada')} **vs.** {reg.get('comparador', 'No reportado')}")
                st.markdown(f"**Desenlaces e Instrumentos:** {reg.get('desenlaces_primarios_secundarios', 'No reportado')} — *{reg.get('herramientas_evaluacion')}*")
                st.markdown(f"**Variables Continuas:** {reg.get('variables_continuas', 'No reportado')}")
                st.markdown(f"**Variables Categóricas / Efecto:** {reg.get('variables_categoricas', 'No reportado')}")
                st.markdown(f"**Riesgo de Sesgo / Calidad:** {reg.get('riesgo_sesgo_calidad', 'No reportado')} | **Nivel:** {reg.get('nivel_evidencia')}")
                if reg.get("hallazgos_cualitativos_citas") and reg.get("hallazgos_cualitativos_citas") != "No reportado":
                    st.markdown(f"**Hallazgos / Citas:** {reg.get('hallazgos_cualitativos_citas')}")
                st.info(reg.get("contenido_chunk"))
    else:
        st.info("Aún no hay bloques cargados. Sube un PDF en el panel izquierdo para comenzar.")
