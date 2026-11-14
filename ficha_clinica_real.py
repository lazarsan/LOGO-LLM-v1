import streamlit as st

def renderizar_ficha_real() -> str:
    st.info("Cumplimente únicamente los campos disponibles. Use códigos seudonimizados (sin nombres ni NHC).")
    
    with st.expander("0. Datos Generales de la Ficha", expanded=True):
        c1, c2 = st.columns(2)
        cod_caso = c1.text_input("Código del caso (ej. REAL-01)", value="REAL-01")
        cod_eval = c2.text_input("Código profesional evaluador", value="EXP-01")
        estado_ficha = st.selectbox("Estado de la ficha", ["Completa", "Incompleta", "Pendiente de verificación"])

    with st.expander("1. Identificación clínica y diagnóstico médico"):
        edad_sexo = st.text_input("Edad (o intervalo) y Sexo")
        diag_med = st.text_input("Diagnóstico médico principal, etiología y subtipo")
        evol_estadio = st.text_input("Tiempo de evolución y estadio/gravedad (ej. Hoehn & Yahr, SARA, NIHSS)")
        comorb_farm = st.text_area("Comorbilidades relevantes, farmacología (ON/OFF) y motivo de consulta", height=80)

    with st.expander("2. Perfil logopédico actual (A–G)"):
        resp_fon = st.text_area("A. Respiración y B. Fonación (patrón, TMF, calidad vocal, acústica)", height=70)
        art_res_pros = st.text_area("C. Articulación, D. Resonancia y E. Prosodia (inteligibilidad, errores)", height=70)
        leng_imp = st.text_area("F. Lenguaje/Comunicación funcional y G. Impacto funcional (CAA, participación)", height=70)

    with st.expander("3. Diagnóstico logopédico"):
        diag_log = st.text_input("Diagnóstico logopédico principal y asociados")
        grav_cert = st.selectbox("Nivel de certeza diagnóstica", ["Confirmado", "Probable", "Provisional", "Información insuficiente"])
        dif_fund = st.text_area("Gravedad, diagnóstico diferencial y fundamento", height=70)

    with st.expander("4. Seguridad deglutoria"):
        st.caption("Nota: La ausencia de tos o caída aislada de SpO₂ no confirman aspiración silente.")
        deg_sint = st.text_area("Antecedentes, evaluación clínica (MECV-V, bolos, SpO₂) y conclusión", height=70)
        deg_inst = st.selectbox("Estado de prueba instrumental (FEES / VFSS)", [
            "No realizada / No disponible",
            "FEES realizada (detallar hallazgos abajo)",
            "VFSS realizada (detallar hallazgos abajo)"
        ])
        deg_hallazgos = st.text_area("Hallazgos instrumentales (PAS, residuos) y recomendaciones vigentes", height=60)

    with st.expander("5. Estado cognitivo-conductual"):
        cog_cond = st.text_area("Alerta, atención, memoria, funciones ejecutivas, comprensión, anosognosia", height=70)

    with st.expander("6 y 7. Perfil neurológico y pruebas adicionales"):
        neuro_pruebas = st.text_area("Neuroimagen/genética, fluctuaciones ON/OFF y otras pruebas (nombre, fecha, resultado)", height=70)

    with st.expander("8. Propuesta o consulta de intervención"):
        prop_int = st.text_area("Objetivos, técnicas a consultar, dudas clínicas o dilema terapéutico", height=80)

    # Compilar solo los campos que el profesional haya rellenado
    secciones = [
        f"# FICHA DE CASO CLÍNICO REAL ({cod_caso} | Evaluador: {cod_eval} | Estado: {estado_ficha})",
        f"1. DIAGNÓSTICO MÉDICO: {edad_sexo} | {diag_med} | {evol_estadio} | {comorb_farm}",
        f"2. PERFIL LOGOPÉDICO: Resp/Fon: {resp_fon} | Art/Res/Pros: {art_res_pros} | Leng/Imp: {leng_imp}",
        f"3. DIAGNÓSTICO LOGOPÉDICO: {diag_log} (Certeza: {grav_cert}) | {dif_fund}",
        f"4. DEGLUCIÓN: {deg_sint} | Instrumental: {deg_inst} | {deg_hallazgos}",
        f"5. COGNICIÓN Y CONDUCTA: {cog_cond}",
        f"6-7. NEUROLOGÍA Y PRUEBAS: {neuro_pruebas}",
        f"8. CONSULTA / INTERVENCIÓN: {prop_int}"
    ]
    
    # Filtrar líneas vacías para enviar un texto limpio al motor
    ficha_compilada = "\n".join([s for s in secciones if len(s.split(": ", 1)[1].strip(" |()")) > 15])
    
    if ficha_compilada:
        st.download_button(
            label="Descargar Ficha Cumplimentada (.md)",
            data=ficha_compilada,
            file_name=f"Ficha_{cod_caso}.md",
            mime="text/markdown",
            use_container_width=True
        )
    return ficha_compilada
