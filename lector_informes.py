import io
import re
import docx
import openpyxl
from pypdf import PdfReader

def anonimizar_texto_clinico(texto: str) -> str:
    # Censurar DNI/NIE españoles, teléfonos, correos y números de historia clínica
    texto = re.sub(r'\b[0-9]{8}[A-Za-z]\b', '[ID_CENSURADO]', texto)
    texto = re.sub(r'\b[XYZxyz][0-9]{7}[A-Za-z]\b', '[ID_CENSURADO]', texto)
    texto = re.sub(r'\b[6789][0-9]{8}\b', '[TEL_CENSURADO]', texto)
    texto = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_CENSURADO]', texto)
    texto = re.sub(r'(?i)(nombre|paciente|apellidos|nhc|historia)\s*:\s*[^\n]+', r'\1: [DATO_ANONIMIZADO]', texto)
    return texto.strip()

def extraer_texto_archivo(archivo_subido) -> str:
    nombre = archivo_subido.name.lower()
    bytes_archivo = archivo_subido.read()
    texto_extraido = ""

    if nombre.endswith(".txt"):
        texto_extraido = bytes_archivo.decode("utf-8", errors="ignore")

    elif nombre.endswith(".pdf"):
        lector = PdfReader(io.BytesIO(bytes_archivo))
        texto_extraido = "\n".join([pag.extract_text() or "" for pag in lector.pages])

    elif nombre.endswith(".docx"):
        doc = docx.Document(io.BytesIO(bytes_archivo))
        texto_extraido = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    elif nombre.endswith(".xlsx"):
        libro = openpyxl.load_workbook(io.BytesIO(bytes_archivo), data_only=True)
        filas = []
        for hoja in libro.worksheets:
            for fila in hoja.iter_rows(values_only=True):
                celdas = [str(c) for c in fila if c is not None]
                if celdas:
                    filas.append(" | ".join(celdas))
        texto_extraido = "\n".join(filas)

    return anonimizar_texto_clinico(texto_extraido)
