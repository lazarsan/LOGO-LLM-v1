import json
import os
from typing import List, Literal, Optional
import streamlit as st
from pydantic import BaseModel, Field
from openai import OpenAI
from pypdf import PdfReader

st.set_page_config(
    page_title="Extractor de Evidencia Científica | LOGO-Pilot",
    page_icon="📊",
    layout="wide"
)

# 1. ESQUEMA PYDANTIC DE EXTRACCIÓN CLÍNICA Y ESTADÍSTICA
class DesenlaceMedido(BaseModel):
    tipo_desenlace: Literal["Primario", "Secundario"] = Field(
        description="Indica si es desenlace primario o secundario."
    )
    nombre_desenlace: str = Field(
        description="Variable clínica o funcional medida (ej. inteligibilidad del habla, seguridad deglutoria, presión espiratoria máxima)."
    )
    instrumento_escala: str = Field(
        description="Escala, prueba acústica o instrumental empleada (ej. SARA, FDA-2, VHI, VFSS, FEES, SPL)."
    )
    momentos_medicion: str = Field(
        description="Tiempos de evaluación (ej. basal, post-tratamiento inmediato, seguimiento a 3 meses)."
    )

class VariablesContinuas(BaseModel):
    tamano_muestral_grupos: str = Field(
        description="Tamaño muestral total y por grupo (ej. n=24; Experimental n=12, Control n=12)."
    )
    media_y_de: str = Field(
        description="Media y desviación estándar (DE) por grupo y momento si aplica; o 'No reportado'."
    )
    mediana_y_ric: str = Field(
        description="Mediana y rango intercuartílico (RIC / IQR) si aplica; o 'No reportado'."
    )
    error_estandar: str = Field(
        description="Error estándar (EE / SE) si aplica; o 'No reportado'."
    )

class VariablesCategoricasEfecto(BaseModel):
    eventos_y_proporciones: str = Field(
        description="Número de eventos y proporciones (%) por grupo si aplica; o 'No reportado'."
    )
    medidas_asociacion_ic_p: str = Field(
        description="Odds Ratio (OR), Riesgo Relativo (RR), tamaño del efecto (d de Cohen), intervalos de confianza (IC 95%) y valores p; o 'No reportado'."
    )

class ChunkEvidenciaCientifica(BaseModel):
    rs_id: str = Field(
        description="Identificador único compuesto del bloque (ej. COCHRANE-2014-ATX-RESUMEN, ECA-2023-SCA3-TABLA-01)."
    )
    tipo_documento: Literal[
        "Revisión Sistemática / Metaanálisis",
        "Ensayo Clínico (ECA / Clinical Trial)",
        "Artículo Científico Primario / Observacional"
    ] = Field(description="Tipo de publicación científica analizada.")
    seccion_rs: Literal[
        "resumen_conclusiones",
        "tabla_estudio_primario",
        "discusion_practica_clinica"
    ] = Field(
        description="Sección estratégica: 'resumen_conclusiones' para síntesis global, 'tabla_estudio_primario' para cada estudio/brazo analizado, 'discusion_practica_clinica' para implicaciones por subdominio."
    )
    prioridad_recuperacion: Literal["alta", "media"] = Field(
        description="'alta' para resumen_conclusiones; 'media' para estudios individuales y discusión."
    )
    anio_publicacion: int = Field(description="Año de publicación del documento.")
    patologia: Literal["Ataxia Hereditaria", "Enfermedad de Parkinson", "Daño Cerebral Adquirido"]
    subtipo_ataxia: str = Field(
        description="Subtipo específico (ej. SCA1, SCA3, Ataxia de Friedreich, Ataxias mixtas) o fenotipo clínico."
    )
    dominio_clinico: Literal["Evaluación", "Disartria", "Disfagia Neurogénica", "Afasia", "Dispraxia", "General"]
    intervencion_evaluada: str = Field(
        description="Descripción detallada de la intervención o exposición (técnica, dosis, frecuencia, duración)."
    )
    comparadores: str = Field(
        description="Grupo control, placebo, terapia estándar o ausencia de comparador (ej. serie de casos)."
    )
    desenlaces_e_instrumentos: List[DesenlaceMedido] = Field(
        description="Lista estructurada de desenlaces primarios/secundarios, escalas y momentos de medición."
    )
    herramientas_evaluacion: str = Field(
        description="Resumen consolidado de todas las herramientas y escalas empleadas."
    )
    variables_continuas: VariablesContinuas
    variables_categoricas_efecto: VariablesCategoricasEfecto
    hallazgos_cualitativos_citas: str = Field(
        description="Categorías temáticas, citas textuales relevantes o hallazgos cualitativos/narrativos clave."
    )
    riesgo_sesgo_calidad: str = Field(
        description="Valoración del riesgo de sesgo o calidad metodológica según herramienta empleada (ej. Cochrane RoB 2, ROBINS-I, JBI, escala PEDro, GRADE) y dominios afectados."
    )
    nivel_evidencia: str = Field(
        description="Grado de certeza global (ej. GRADE Muy bajo, ECA piloto, Serie de casos Nivel IV)."
    )
    limitaciones: str = Field(
        description="Limitaciones metodológicas (ej. muestra pequeña, sin cegamiento, heterogeneidad, seguimiento corto)."
    )
    contenido_chunk: str = Field(
        description="Síntesis clínica densa y autocontenida del bloque lista para ser recuperada por el CDSS, incluyendo título de la fuente como contexto."
    )

class ExtraccionDocumentoCientifico(BaseModel):
    chunks: List[ChunkEvidenciaCientifica]


# 2. FUNCIONES AUXILIARES
def extraer_texto_pdf(archivo_subido) -> str:
    lector = PdfReader(archivo_subido)
    paginas = []
    for i, page in enumerate(lector.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            paginas.append(f"--- PÁGINA {i + 1} ---
{txt}")
    return "

".join(paginas)


def construir_prompt_extraccion(
    texto_pdf: str,
    id_base: str,
    anio: int,
    patologia: str,
    tipo_doc: str
) -> str:
    return f"""Eres un metodólogo clínico experto en Medicina Basada en la Evidencia y Logopedia Neurológica.
Analiza el siguiente texto científico ({tipo_doc}, año {anio}, patología: {patologia}, ID base: {id_base}) y extrae bloques estructurados siguiendo estrictamente estas reglas:

REGLAS DE FRAGMENTACIÓN ESTRATÉGICA:
1. RESUMEN Y CONCLUSIONES ('resumen_conclusiones', prioridad 'alta'): Genera un único chunk completo que sintetice las conclusiones de los autores, el objetivo PICO y el nivel global de evidencia.
2. ESTUDIOS PRIMARIOS / BRAZOS DE INTERVENCIÓN ('tabla_estudio_primario', prioridad 'media'):
   - Si es una Revisión Sistemática: convierte CADA estudio incluido en las tablas de características o resultados en un chunk independiente, manteniendo el título de la RS como contexto.
   - Si es un Clinical Trial (ECA) o Artículo Primario: genera un chunk detallado por cada comparación principal o subgrupo analizado.
3. DISCUSIÓN E IMPLICACIONES CLÍNICAS ('discusion_practica_clinica', prioridad 'media'): Fragmenta semánticamente por subdominio clínico (Evaluación, Disartria, Disfagia Neurogénica, etc.) conectando los hallazgos con la seguridad y práctica clínica.

REGLAS DE EXTRACCIÓN METODOLÓGICA Y ESTADÍSTICA (OBLIGATORIAS):
- Intervenciones y comparadores: Especifica dosis, intensidad, frecuencia, duración y características del grupo comparador.
- Desenlaces: Distingue desenlaces Primarios y Secundarios, su instrumento/escala exacta y todos los momentos de medición.
- Variables continuas: Extrae tamaño muestral (n) por grupos, media y desviación estándar (DE), mediana y rango intercuartílico (RIC) y error estándar (EE) cuando consten en el texto o tablas.
- Variables categóricas y efecto: Extrae número de eventos, proporciones (%), Odds Ratio (OR), Riesgo Relativo (RR), intervalos de confianza (IC 95%) y valores p.
- Cualitativo / Narrativo: Extrae categorías, citas o hallazgos relevantes.
- Riesgo de sesgo y calidad: Extrae la evaluación de sesgo según Cochrane (RoB 2 / ROBINS-I), JBI, PEDro o GRADE tal como aparezca en el documento.
- Identificadores: Nombra cada 'rs_id' empezando por '{id_base}-' (ej. '{id_base}-RESUMEN', '{id_base}-ESTUDIO-01', '{id_base}-DISC-DISARTRIA'). No inventes cifras no presentes en el texto.

TEXTO DEL DOCUMENTO:
{texto_pdf[:95000]}
"""


# 3. INTERFAZ VISUAL STREAMLIT
st.title("📊 Extractor de Evidencia Científica (RS · Clinical Trials · Artículos)")
st.markdown(
    "Genera y actualiza la capa estructurada **`capa_evidencia_rs.json`** con extracción PICO, "
    "estadística (variables continuas/categóricas) y riesgo de sesgo (Cochrane / JBI) sin alterar tu índice de guías clínicas."
)

if not os.getenv("OPENAI_API_KEY"):
    st.error("Falta configurar OPENAI_API_KEY en Secrets o en las variables de entorno.")
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
        "Tipo de documento:",
        [
            "Revisión Sistemática / Metaanálisis",
            "Ensayo Clínico (ECA / Clinical Trial)",
            "Artículo Científico Primario / Observacional"
        ]
    )
    patologia_sel = st.selectbox(
        "Patología neurológica:",
        ["Ataxia Hereditaria", "Enfermedad de Parkinson", "Daño Cerebral Adquirido"]
    )
    c_id, c_anio = st.columns([0.65, 0.35])
    id_base = c_id.text_input("Identificador corto (sin espacios):", placeholder="ej. COCHRANE-2014-ATX").strip().upper()
    anio_pub = c_anio.number_input("Año:", min_value=1990, max_value=2030, value=2024, step=1)

    procesar = st.button("⚙️ Extraer Datos Metodológicos y Fragmentar", type="primary", use_container_width=True)

if procesar:
    if not archivo_pdf or not id_base:
        st.warning("Por favor, sube un archivo PDF e indica un identificador corto antes de procesar.")
    else:
        with st.spinner("Analizando texto, tablas estadísticas y riesgo de sesgo con GPT-4o..."):
            texto_extraido = extraer_texto_pdf(archivo_pdf)
            prompt = construir_prompt_extraccion(texto_extraido, id_base, int(anio_pub), patologia_sel, tipo_doc)
            cliente = OpenAI()
            respuesta = cliente.beta.chat.completions.parse(
                model="gpt-4o",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Eres un extractor experto de datos biomédicos y estadísticos para medicina basada en la evidencia."},
                    {"role": "user", "content": prompt}
                ],
                response_format=ExtraccionDocumentoCientifico,
            )
            resultado = respuesta.choices[0].message.parsed

            # Eliminar bloques previos del mismo id_base para evitar duplicados al re-procesar
            st.session_state["acumulado_rs"] = [
                item for item in st.session_state["acumulado_rs"]
                if not str(item.get("rs_id", "")).upper().startswith(f"{id_base}-")
            ]

            nuevos_registros = []
            for chunk in resultado.chunks:
                reg = chunk.model_dump()
                reg["archivo_origen"] = archivo_pdf.name
                nuevos_registros.append(reg)

            st.session_state["acumulado_rs"].extend(nuevos_registros)
            st.success(f"✅ Se extrajeron {len(nuevos_registros)} bloques estructurados de '{archivo_pdf.name}'.")

with col2:
    st.subheader(f"2. Capa de Evidencia Acumulada ({len(st.session_state['acumulado_rs'])} bloques)")

    if st.session_state["acumulado_rs"]:
        json_final = json.dumps(st.session_state["acumulado_rs"], ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Descargar capa_evidencia_rs.json actualizado",
            data=json_final,
            file_name="capa_evidencia_rs.json",
            mime="application/json",
            type="primary",
            use_container_width=True
        )

        st.divider()
        for idx, item in enumerate(reversed(st.session_state["acumulado_rs"][-10:])):
            with st.expander(f"{item.get('rs_id')} · {item.get('dominio_clinico')} ({item.get('seccion_rs')})", expanded=(idx == 0)):
                st.markdown(f"**Documento:** {item.get('tipo_documento')} | **Subtipo:** {item.get('subtipo_ataxia')} | **Archivo:** `{item.get('archivo_origen', '')}`")
                st.markdown(f"**Intervención:** {item.get('intervencion_evaluada')} **vs.** {item.get('comparadores', 'No especificado')}")
                st.markdown(f"**Riesgo de sesgo / Calidad:** {item.get('riesgo_sesgo_calidad', 'No especificado')} | **Certeza:** {item.get('nivel_evidencia')}")
                vc = item.get("variables_continuas", {})
                vcat = item.get("variables_categoricas_efecto", {})
                if vc:
                    st.caption(f"📈 Continuas: n={vc.get('tamano_muestral_grupos')} | Media/DE: {vc.get('media_y_de')} | Mediana/RIC: {vc.get('mediana_y_ric')} | EE: {vc.get('error_estandar')}")
                if vcat:
                    st.caption(f"📊 Categóricas/Efecto: Eventos: {vcat.get('eventos_y_proporciones')} | OR/RR/IC/p: {vcat.get('medidas_asociacion_ic_p')}")
                st.write(item.get("contenido_chunk"))
    else:
        st.info("Aún no hay bloques cargados. Sube un PDF a la izquierda para comenzar.")
