"""
preprocesamiento.py
-------------------
Módulo compartido entre el notebook y la app de Streamlit.
Contiene los diccionarios de códigos (traducidos al español), la limpieza del
dataset original de la UCI y el transformador de ingeniería de características
que va dentro del pipeline del modelo. Tenerlo en un solo archivo evita que el
notebook y la app terminen procesando los datos de forma distinta.

ACA 1 - Aprendizaje Automático | Especialización en IA - CUN
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SEED = 42

# ---------------------------------------------------------------------------
# 1. Nombres de columnas: original UCI -> español (snake_case)
#    Ojo: en el CSV original la columna de jornada trae un tabulador al final
#    ("Daytime/evening attendance\t") y nacionalidad viene mal escrita
#    ("Nacionality"). Por eso se limpian los nombres antes de renombrar.
# ---------------------------------------------------------------------------
RENOMBRAR = {
    "Marital status": "estado_civil",
    "Application mode": "modo_ingreso",
    "Application order": "orden_opcion",
    "Course": "programa",
    "Daytime/evening attendance": "jornada_diurna",
    "Previous qualification": "formacion_previa",
    "Previous qualification (grade)": "nota_formacion_previa",
    "Nacionality": "nacionalidad",
    "Mother's qualification": "educacion_madre",
    "Father's qualification": "educacion_padre",
    "Mother's occupation": "ocupacion_madre",
    "Father's occupation": "ocupacion_padre",
    "Admission grade": "nota_admision",
    "Displaced": "desplazado",
    "Educational special needs": "necesidades_especiales",
    "Debtor": "deudor",
    "Tuition fees up to date": "matricula_al_dia",
    "Gender": "genero_masculino",
    "Scholarship holder": "becado",
    "Age at enrollment": "edad_ingreso",
    "International": "internacional",
    "Curricular units 1st sem (credited)": "s1_convalidadas",
    "Curricular units 1st sem (enrolled)": "s1_inscritas",
    "Curricular units 1st sem (evaluations)": "s1_evaluaciones",
    "Curricular units 1st sem (approved)": "s1_aprobadas",
    "Curricular units 1st sem (grade)": "s1_nota",
    "Curricular units 1st sem (without evaluations)": "s1_sin_evaluacion",
    "Curricular units 2nd sem (credited)": "s2_convalidadas",
    "Curricular units 2nd sem (enrolled)": "s2_inscritas",
    "Curricular units 2nd sem (evaluations)": "s2_evaluaciones",
    "Curricular units 2nd sem (approved)": "s2_aprobadas",
    "Curricular units 2nd sem (grade)": "s2_nota",
    "Curricular units 2nd sem (without evaluations)": "s2_sin_evaluacion",
    "Unemployment rate": "tasa_desempleo",
    "Inflation rate": "tasa_inflacion",
    "GDP": "pib",
    "Target": "resultado",
}

CLASES = {"Dropout": "Deserción", "Enrolled": "Matriculado", "Graduate": "Graduado"}
ORDEN_CLASES = ["Deserción", "Matriculado", "Graduado"]

# ---------------------------------------------------------------------------
# 2. Diccionarios de códigos (fuente: ficha del dataset en UCI, traducidos)
# ---------------------------------------------------------------------------
ESTADO_CIVIL = {1: "Soltero(a)", 2: "Casado(a)", 3: "Viudo(a)", 4: "Divorciado(a)",
                5: "Unión de hecho", 6: "Separado(a) legalmente"}

PROGRAMAS = {
    33: "Tecnologías de producción de biocombustibles",
    171: "Animación y diseño multimedia",
    8014: "Servicio social (nocturno)",
    9003: "Agronomía",
    9070: "Diseño de comunicación",
    9085: "Enfermería veterinaria",
    9119: "Ingeniería informática",
    9130: "Equinocultura",
    9147: "Administración",
    9238: "Servicio social",
    9254: "Turismo",
    9500: "Enfermería",
    9556: "Higiene oral",
    9670: "Publicidad y marketing",
    9773: "Periodismo y comunicación",
    9853: "Educación básica",
    9991: "Administración (nocturno)",
}

MODO_INGRESO = {
    1: "1.ª fase - contingente general",
    2: "Ordenanza 612/93",
    5: "1.ª fase - contingente especial (Azores)",
    7: "Titulado de otro programa superior",
    10: "Ordenanza 854-B/99",
    15: "Estudiante internacional (licenciatura)",
    16: "1.ª fase - contingente especial (Madeira)",
    17: "2.ª fase - contingente general",
    18: "3.ª fase - contingente general",
    26: "Ordenanza 533-A/99 b2 (plan diferente)",
    27: "Ordenanza 533-A/99 b3 (otra institución)",
    39: "Mayores de 23 años",
    42: "Traslado",
    43: "Cambio de programa",
    44: "Diploma de especialización tecnológica",
    51: "Cambio de institución/programa",
    53: "Diploma de ciclo corto",
    57: "Cambio de institución/programa (internacional)",
}

FORMACION_PREVIA = {
    1: "Secundaria completa", 2: "Superior - bachiller", 3: "Superior - licenciatura",
    4: "Superior - maestría", 5: "Superior - doctorado", 6: "Estudios superiores sin terminar",
    9: "12.º año sin terminar", 10: "11.º año sin terminar", 12: "Otro - 11.º año",
    14: "10.º año", 15: "10.º año sin terminar", 19: "Básica 3.er ciclo",
    38: "Básica 2.º ciclo", 39: "Especialización tecnológica", 40: "Superior - grado (1.er ciclo)",
    42: "Técnico superior profesional", 43: "Superior - maestría (2.º ciclo)",
}

# Agrupaciones para reducir categorías muy dispersas (29 a 46 códigos en los
# padres). Se agrupa por nivel educativo y por gran grupo ocupacional (CIUO).
NIVEL_EDUCATIVO = {
    "Sin estudios": [35, 36],
    "Básica": [11, 19, 26, 29, 30, 37, 38],
    "Media": [1, 9, 10, 12, 13, 14, 15, 18, 20, 25, 27, 31, 33],
    "Técnica/tecnológica": [22, 39, 42],
    "Superior": [2, 3, 4, 5, 6, 40, 41, 43, 44],
    "Desconocido": [34],
}
COD_A_NIVEL = {c: n for n, cods in NIVEL_EDUCATIVO.items() for c in cods}

GRUPO_OCUPACION = {
    0: "Estudiante", 1: "Directivos", 2: "Profesionales e intelectuales",
    3: "Técnicos de nivel medio", 4: "Personal administrativo", 5: "Servicios y ventas",
    6: "Agropecuario", 7: "Oficios industriales y construcción", 8: "Operadores de máquinas",
    9: "Trabajadores no calificados", 10: "Fuerzas armadas", 90: "Otra situación",
    99: "Sin información",
}


def grupo_ocupacion(codigo):
    """Lleva un código detallado (p. ej. 122, 153) a su gran grupo (2, 5)."""
    if pd.isna(codigo):
        return np.nan
    c = int(codigo)
    if c in GRUPO_OCUPACION:
        return GRUPO_OCUPACION[c]
    if 101 <= c <= 103:
        return GRUPO_OCUPACION[10]
    if 110 <= c <= 199:
        return GRUPO_OCUPACION.get(c // 10 - 10, "Otra situación")
    return "Otra situación"


# ---------------------------------------------------------------------------
# 3. Limpieza del dataset original
# ---------------------------------------------------------------------------
CURRICULARES = [f"s{s}_{v}" for s in (1, 2) for v in
                ("convalidadas", "inscritas", "evaluaciones", "aprobadas", "nota", "sin_evaluacion")]


def limpiar_dataset(df_original: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres, traduce el objetivo y marca como faltantes los
    registros inconsistentes. Devuelve un DataFrame nuevo."""
    df = df_original.copy()
    df.columns = [c.strip().replace("\t", "") for c in df.columns]
    df = df.rename(columns=RENOMBRAR)

    # (a) Programa 171 (Animación y diseño multimedia): los 180 estudiantes que
    #     tiene registran 0 unidades inscritas, evaluadas y aprobadas en ambos
    #     semestres, incluso los que se graduaron. No son ceros reales sino
    #     ausencia de registro, así que se pasan a NaN para imputarlos.
    sin_registro = (df["s1_inscritas"] == 0) & (df["s2_inscritas"] == 0)
    df["sin_registro_curricular"] = sin_registro.astype(int)
    df.loc[sin_registro, CURRICULARES] = np.nan

    # (b) En Portugal se aprueba con 10/20. Un promedio de aprobadas por debajo
    #     de 10 con unidades aprobadas es imposible -> NaN.
    for s in (1, 2):
        malo = (df[f"s{s}_aprobadas"] > 0) & (df[f"s{s}_nota"] < 10)
        df.loc[malo, f"s{s}_nota"] = np.nan

    # (c) 'nacionalidad' es redundante con 'internacional' (coinciden en el
    #     100 % de los registros), se conserva solo la binaria.
    df = df.drop(columns=["nacionalidad"])

    if "resultado" in df.columns:
        df["resultado"] = df["resultado"].map(CLASES)
    return df


# ---------------------------------------------------------------------------
# 4. Ingeniería de características (va dentro del pipeline)
# ---------------------------------------------------------------------------
def _razon(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return np.where(den > 0, num / den.where(den > 0, 1), np.where(den == 0, 0.0, np.nan))


class IngenieriaCaracteristicas(BaseEstimator, TransformerMixin):
    """Crea variables derivadas y agrupa códigos de alta cardinalidad.
    Solo calcula lo que las columnas disponibles permiten, así sirve para los
    tres escenarios (al ingreso, tras el 1.er semestre y tras el 1.er año)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for s in (1, 2):
            if {f"s{s}_aprobadas", f"s{s}_inscritas"} <= set(X.columns):
                X[f"s{s}_tasa_aprobacion"] = _razon(X[f"s{s}_aprobadas"], X[f"s{s}_inscritas"])
        if "s2_tasa_aprobacion" in X.columns:
            X["tasa_aprobacion_anual"] = _razon(X["s1_aprobadas"] + X["s2_aprobadas"],
                                                X["s1_inscritas"] + X["s2_inscritas"])
            X["cambio_nota"] = X["s2_nota"] - X["s1_nota"]
            X["cambio_tasa_aprobacion"] = X["s2_tasa_aprobacion"] - X["s1_tasa_aprobacion"]
        X["nivel_educ_madre"] = X["educacion_madre"].map(COD_A_NIVEL)
        X["nivel_educ_padre"] = X["educacion_padre"].map(COD_A_NIVEL)
        X["nivel_formacion_previa"] = X["formacion_previa"].map(COD_A_NIVEL)
        X["grupo_ocup_madre"] = X["ocupacion_madre"].map(grupo_ocupacion)
        X["grupo_ocup_padre"] = X["ocupacion_padre"].map(grupo_ocupacion)
        return X


# ---------------------------------------------------------------------------
# 5. Columnas por escenario
#    - "ingreso":   solo lo que se sabe el día de la matrícula
#    - "semestre1": + resultados del primer semestre
#    - "anual":     + resultados del segundo semestre (modelo principal)
# ---------------------------------------------------------------------------
ESCENARIOS = {
    "ingreso": "Al ingreso (sin notas)",
    "semestre1": "Tras el 1.er semestre",
    "anual": "Tras el 1.er año (modelo principal)",
}

_BASE_NUM = ["orden_opcion", "nota_formacion_previa", "nota_admision", "edad_ingreso",
             "tasa_desempleo", "tasa_inflacion", "pib"]
_S1_NUM = ["s1_convalidadas", "s1_inscritas", "s1_evaluaciones", "s1_aprobadas", "s1_nota",
           "s1_sin_evaluacion", "s1_tasa_aprobacion"]
_S2_NUM = ["s2_convalidadas", "s2_inscritas", "s2_evaluaciones", "s2_aprobadas", "s2_nota",
           "s2_sin_evaluacion", "s2_tasa_aprobacion", "tasa_aprobacion_anual", "cambio_nota",
           "cambio_tasa_aprobacion"]
_BASE_BIN = ["jornada_diurna", "desplazado", "necesidades_especiales", "deudor", "matricula_al_dia",
             "genero_masculino", "becado", "internacional"]
CATEGORICAS = ["estado_civil", "modo_ingreso", "programa", "nivel_formacion_previa",
               "nivel_educ_madre", "nivel_educ_padre", "grupo_ocup_madre", "grupo_ocup_padre"]
_BASE_ENT = ["estado_civil", "modo_ingreso", "orden_opcion", "programa", "jornada_diurna",
             "formacion_previa", "nota_formacion_previa", "educacion_madre", "educacion_padre",
             "ocupacion_madre", "ocupacion_padre", "nota_admision", "desplazado",
             "necesidades_especiales", "deudor", "matricula_al_dia", "genero_masculino", "becado",
             "edad_ingreso", "internacional", "tasa_desempleo", "tasa_inflacion", "pib"]
_S1_ENT = [c for c in CURRICULARES if c.startswith("s1_")]
_S2_ENT = [c for c in CURRICULARES if c.startswith("s2_")]


def columnas(escenario: str = "anual") -> dict:
    """Devuelve las listas de columnas (entrada, numéricas, binarias y
    categóricas) que usa cada escenario."""
    if escenario == "ingreso":
        return {"entradas": _BASE_ENT, "numericas": _BASE_NUM,
                "binarias": _BASE_BIN, "categoricas": CATEGORICAS}
    if escenario == "semestre1":
        return {"entradas": _BASE_ENT + _S1_ENT + ["sin_registro_curricular"],
                "numericas": _BASE_NUM + _S1_NUM,
                "binarias": _BASE_BIN + ["sin_registro_curricular"], "categoricas": CATEGORICAS}
    if escenario == "anual":
        return {"entradas": _BASE_ENT + _S1_ENT + _S2_ENT + ["sin_registro_curricular"],
                "numericas": _BASE_NUM + _S1_NUM + _S2_NUM,
                "binarias": _BASE_BIN + ["sin_registro_curricular"], "categoricas": CATEGORICAS}
    raise ValueError(f"Escenario desconocido: {escenario}")


# Nombres legibles para gráficas y para la app
ETIQUETAS = {
    "estado_civil": "Estado civil", "modo_ingreso": "Modalidad de ingreso",
    "orden_opcion": "Orden de preferencia del programa", "programa": "Programa",
    "jornada_diurna": "Jornada diurna", "formacion_previa": "Formación previa",
    "nota_formacion_previa": "Nota de formación previa", "educacion_madre": "Educación de la madre",
    "educacion_padre": "Educación del padre", "ocupacion_madre": "Ocupación de la madre",
    "ocupacion_padre": "Ocupación del padre", "nota_admision": "Nota de admisión",
    "desplazado": "Vive fuera de su ciudad", "necesidades_especiales": "Necesidades educativas especiales",
    "deudor": "Deudor", "matricula_al_dia": "Matrícula al día", "genero_masculino": "Género (hombre)",
    "becado": "Becado", "edad_ingreso": "Edad de ingreso", "internacional": "Estudiante internacional",
    "s1_convalidadas": "S1 · unidades convalidadas", "s1_inscritas": "S1 · unidades inscritas",
    "s1_evaluaciones": "S1 · evaluaciones presentadas", "s1_aprobadas": "S1 · unidades aprobadas",
    "s1_nota": "S1 · promedio de notas", "s1_sin_evaluacion": "S1 · unidades sin evaluación",
    "s2_convalidadas": "S2 · unidades convalidadas", "s2_inscritas": "S2 · unidades inscritas",
    "s2_evaluaciones": "S2 · evaluaciones presentadas", "s2_aprobadas": "S2 · unidades aprobadas",
    "s2_nota": "S2 · promedio de notas", "s2_sin_evaluacion": "S2 · unidades sin evaluación",
    "tasa_desempleo": "Tasa de desempleo (%)", "tasa_inflacion": "Tasa de inflación (%)", "pib": "PIB (variación %)",
    "sin_registro_curricular": "Sin registro curricular",
}
ETIQUETAS.update({
    "s1_tasa_aprobacion": "S1 · tasa de aprobación", "s2_tasa_aprobacion": "S2 · tasa de aprobación",
    "tasa_aprobacion_anual": "Tasa de aprobación del año", "cambio_nota": "Cambio de promedio S1 → S2",
    "cambio_tasa_aprobacion": "Cambio de tasa de aprobación S1 → S2",
    "nivel_formacion_previa": "Nivel de formación previa", "nivel_educ_madre": "Nivel educativo de la madre",
    "nivel_educ_padre": "Nivel educativo del padre", "grupo_ocup_madre": "Ocupación de la madre",
    "grupo_ocup_padre": "Ocupación del padre",
})

# Código representativo de cada nivel/grupo: el modelo solo usa el nivel
# agrupado, así que en la app basta con elegir el nivel.
CODIGO_NIVEL = {"Sin estudios": 35, "Básica": 19, "Media": 1, "Técnica/tecnológica": 39,
                "Superior": 3, "Desconocido": 34}
CODIGO_OCUPACION = {v: k for k, v in GRUPO_OCUPACION.items()}
