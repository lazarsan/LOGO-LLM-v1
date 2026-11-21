import json
import os
from typing import List, Literal
import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI
from pypdf import PdfReader

# 1. ESQUEMA DE EXTRACCIÓN ESTRUCTURADA DE REVISIONES SISTEMÁTICAS
class ChunkRS(BaseModel):
    rs_id: str = Field(
        description="Identificador único del chunk (ej. COCHRANE-2014-RESUMEN, RS-2021-FILA-01, RS-2021-DISC-DISARTRIA)."
    )
    tipo_fuente: Literal["revision_sistematica"] = "revision_sistematica"
    seccion_rs: Literal["resumen_conclusiones", "tabla_estudio_primario", "discusion_practica_clinica"] = Field(
        description="Sección de origen según la estrategia de fragmentación."
    )
    prioridad_recuperacion: Literal["alta", "media"] = Field(
        description="'alta' exclusivamente para resumen_conclusiones; 'media' para el resto."
    )
    anio_publicacion: int = Field(description="Año de publicación de la Revisión Sistemática.")
    patologia: str = Field(description="Patología neurológica base (ej. Ataxia Hereditaria).")
    subtipo_ataxia: str = Field(
        description="Subtipo específico evaluado (ej. Ataxia de Friedreich, SCA1, SCA2, SCA3, SCA6, Mixta/No especificada)."
    )
    dominio_clinico: Literal["Evaluación", "Disartria", "Disfagia Neurogénica", "General"] = Field(
        description="Dominio clínico principal del chunk."
    )
    intervencion_evaluada: str = Field(
        description="Terapia, técnica o programa evaluado (ej. LSVT LOUD, terapia miofuncional, manejo postural)."
    )
    herramientas_evaluacion: str = Field(
        description="Escalas, parámetros acústicos o pruebas instrumentales empleadas (ej. SARA, VFSS, FEES, inteligibilidad)."
    )
    nivel_evidencia: str = Field(
        description="Grado de certeza o diseño del estudio (ej. GRADE Muy Bajo, Ausencia de ECA, Serie de casos Nivel IV)."
    )
    limitaciones: str = Field(
        description="Limitaciones metodológicas (ej. muestras pequeñas, sin grupo control, riesgo de sesgo)."
    )
    contenido_chunk: str = Field(
        description="Síntesis factual completa del fragmento. Si es tabla_estudio_primario, incluir el título de la RS como contexto."
    )


class ExtraccionRS(BaseModel):
    titulo_rs: str = Field(description="Título completo de la Revisión Sistemática.")
    chunks: List[ChunkRS]


def extraer_texto_pdf(archivo_subido) -> str:
    lector = PdfReader(archivo_subido)
    paginas = [p.extract_text() or "" for p in lector.pages]
    return "\n\n".join(paginas)


def procesar_rs_con_llm(texto_pdf: str, id_base: str, anio: int, patologia: str, api_key: str) -> ExtraccionRS:
    cliente = OpenAI(api_key=api_key)
    prompt = (
        f"Analiza la siguiente Revisión Sistemática (ID base: {id_base}, Año: {anio}, Patología: {patologia}) "
        "y extrae una capa estructurada de metadatos y chunks siguiendo ESTRICTAMENTE estas 3 reglas de fragmentación:\n\n"
        "1. RESUMEN / CONCLUSIONES DE LOS AUTORES ('resumen_conclusiones', prioridad 'alta'): "
        "Genera UN ÚNICO chunk completo que sintetice las conclusiones globales y el grado de certeza estadística (ej. ausencia de ECA en Cochrane).\n"
        "2. TABLAS DE CARACTERÍSTICAS DE ESTUDIOS INCLUIDOS ('tabla_estudio_primario', prioridad 'media'): "
        "Convierte CADA FILA o estudio primario relevante analizado dentro de la revisión en un chunk independiente. "
        "Incluye siempre el título de la RS como contexto inicial y especifica autor, año, subtipo de ataxia, intervención, herramientas y resultados.\n"
        "3. DISCUSIÓN / IMPLICACIONES PARA LA PRÁCTICA CLÍNICA ('discusion_practica_clinica', prioridad 'media'): "
        "Fragmenta semánticamente por subdominio ('Evaluación', 'Disartria', 'Disfagia Neurogénica' o 'General'), "
        "conectando las limitaciones estadísticas con la práctica clínica y la seguridad del paciente.\n\n"
        f"TEXTO DE LA REVISIÓN SISTEMÁTICA:\n{texto_pdf[:95000]}"
    )

    respuesta = cliente.beta.chat.completions.parse(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "Eres un metodólogo experto en Medicina Basada en la Evidencia y Logopedia Neurológica."
            },
            {"role": "user", "content": prompt},
        ],
        response_format=ExtraccionRS,
    )
    return respuesta.choices[0].message.parsed


# 2. INTERFAZ VISUAL EN STREAMLIT
st.set_page_config(page_title="LogoPilot | Ingesta de Revisiones Sistemáticas", page_icon="📊", layout="wide")

st.markdown("""
<div style="background: linear-gradient(90deg, #1E1B4B 0%, #4C1D95 100%); color: white; padding: 1.5rem 2rem; border-radius: 10px; margin-bottom: 1.5rem; border-left: 6px solid #A855F7;">
    <h2 style="margin: 0; font-size: 1.4rem;">LOGO-Pilot · Procesador de Revisiones Sistemáticas (Capa 2 RAG)</h2>
    <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem; color: #E2E8F0;">Genera metadatos jerárquicos y chunks por sección sin alterar los protocolos clínicos principales</p>
</div>
""", unsafe_allow_html=True)

api_key_env = os.getenv("OPENAI_API_KEY", "")
if not api_key_env:
    api_key_input = st.sidebar.text_input("Clave API de OpenAI (OPENAI_API_KEY):", type="password")
else:
    api_key_input = api_key_env
    st.sidebar.success("OPENAI_API_KEY detectada automáticamente.")

# Inicializar almacén en memoria (cargando capa_evidencia_rs.json si ya existe en el repositorio)
if "chunks_rs_acumulados" not in st.session_state:
    if os.path.exists("capa_evidencia_rs.json"):
        with open("capa_evidencia_rs.json", "r", encoding="utf-8") as f:
            st.session_state["chunks_rs_acumulados"] = json.load(f)
    else:
        st.session_state["chunks_rs_acumulados"] = []

col_form, col_preview = st.columns([1, 1.3])

with col_form:
    st.subheader("1. Cargar Revisión Sistemática (PDF)")
    archivo_pdf = st.file_uploader("Arrastra aquí el PDF de la Revisión Sistemática:", type=["pdf"])
    id_base = st.text_input("Identificador corto de la RS:", placeholder="Ej. COCHRANE-2014-ATX o RS-2022-SCA3")
    anio_rs = st.number_input("Año de publicación:", min_value=1990, max_value=2030, value=2022, step=1)
    patologia_rs = st.selectbox(
        "Patología Neurológica Base:",
        ["Ataxia Hereditaria", "Enfermedad de Parkinson", "Daño Cerebral Adquirido"]
    )

    procesar = st.button("⚙️ Extraer Metadatos y Fragmentar RS", type="primary", use_container_width=True)

if procesar:
    if not api_key_input:
        st.error("Introduce tu clave OPENAI_API_KEY en la barra lateral antes de procesar.")
    elif not archivo_pdf or not id_base.strip():
        st.warning("Sube un archivo PDF y escribe un identificador corto (ej. COCHRANE-2014-ATX).")
    else:
        with st.spinner("Extrayendo texto y estructurando Resumen, Tablas de Estudios y Discusión..."):
            texto_extraido = extraer_texto_pdf(archivo_pdf)
            resultado = procesar_rs_con_llm(
                texto_pdf=texto_extraido,
                id_base=id_base.strip().upper(),
                anio=int(anio_rs),
                patologia=patologia_rs,
                api_key=api_key_input
            )
            nuevos_chunks = [c.model_dump() for c in resultado.chunks]

            # Evitar duplicados si se reprocesa el mismo ID base
            prefix = id_base.strip().upper()
            existentes_filtrados = [
                item for item in st.session_state["chunks_rs_acumulados"]
                if not item.get("rs_id", "").startswith(prefix)
            ]
            st.session_state["chunks_rs_acumulados"] = existentes_filtrados + nuevos_chunks
            st.success(f"Procesada: '{resultado.titulo_rs}' ({len(nuevos_chunks)} chunks generados).")

with col_preview:
    st.subheader("2. Vista Previa y Descarga de `capa_evidencia_rs.json`")
    total_chunks = len(st.session_state["chunks_rs_acumulados"])
    st.info(f"Total de fragmentos acumulados en la capa de evidencia: **{total_chunks}**")

    if total_chunks > 0:
        json_final = json.dumps(st.session_state["chunks_rs_acumulados"], ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Descargar capa_evidencia_rs.json (Listo para subir a GitHub)",
            data=json_final,
            file_name="capa_evidencia_rs.json",
            mime="application/json",
            type="primary",
            use_container_width=True
        )

        st.markdown("#### Inspección de fragmentos generados:")
        for item in reversed(st.session_state["chunks_rs_acumulados"]):
            etiqueta = {
                "resumen_conclusiones": "🔴 [ALTA] Resumen / Conclusiones",
                "tabla_estudio_primario": "🔵 [MEDIA] Estudio Primario (Tabla)",
                "discusion_practica_clinica": "🟢 [MEDIA] Discusión Práctica Clínica"
            }.get(item.get("seccion_rs", ""), item.get("seccion_rs", ""))

            with st.expander(f"{etiqueta} · {item.get('rs_id')} ({item.get('anio_publicacion')})"):
                st.caption(
                    f"**Subtipo:** {item.get('subtipo_ataxia')} | **Dominio:** {item.get('dominio_clinico')} | "
                    f"**Nivel evidencia:** {item.get('nivel_evidencia')}"
                )
                st.markdown(f"**Intervención / Herramientas:** {item.get('intervencion_evaluada')} ({item.get('herramientas_evaluacion')})")
                st.write(item.get("contenido_chunk"))
                st.caption(f"*Limitaciones:* {item.get('limitaciones')}")
