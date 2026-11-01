"""
Alerta temprana de deserción estudiantil
App de Streamlit del ACA 1 - Aprendizaje Automático (Especialización en IA, CUN)

Grupo 7: Anternol de los Reyes Bedoya Naranjo, Bonny Darhyll Brayan Galindo Santos, Andrés Alberto López Herrera y Camilo José Rodríguez Ayubi
Docente: Dr. Hamilton Rivera Flor

Ejecutar en local:  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import preprocesamiento as pp

BASE = Path(__file__).parent
COLORES = {"Deserción": "#D1495B", "Matriculado": "#EDAE49", "Graduado": "#00798C"}
ESCALA = alt.Scale(domain=pp.ORDEN_CLASES, range=[COLORES[c] for c in pp.ORDEN_CLASES])
UMBRAL_ALTO, UMBRAL_MEDIO = 0.50, 0.25  # probabilidad de deserción
EJE_Y = alt.Axis(labelLimit=320, labelOverlap=False)   # muestra todas las etiquetas completas
EJE_X = alt.Axis(labelAngle=0)


def num(x, d=0):
    """Formato colombiano: punto para miles y coma para decimales."""
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=1):
    return num(x * 100, d) + " %"

st.set_page_config(page_title="Alerta temprana de deserción", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.riesgo {border-radius: 12px; padding: 18px 22px; color: white; margin-bottom: 8px;}
.riesgo h3 {margin: 0; color: white;} .riesgo p {margin: 4px 0 0 0; font-size: 0.95rem;}
.pie {font-size: 0.8rem; color: #6b7280;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carga de artefactos (se guardan en caché para no recargarlos en cada clic)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    return {esc: joblib.load(BASE / "modelo" / f"modelo_{esc}.joblib") for esc in pp.ESCENARIOS}


@st.cache_data
def cargar_metadatos():
    return json.loads((BASE / "modelo" / "metadatos.json").read_text(encoding="utf-8"))


@st.cache_data
def cargar_datos():
    crudo = pd.read_csv(BASE / "data" / "data.csv", sep=";", encoding="utf-8-sig")
    df = pp.limpiar_dataset(crudo)
    df["programa_nombre"] = df["programa"].map(pp.PROGRAMAS)
    return df


@st.cache_data
def cargar_ejemplos():
    return pd.read_csv(BASE / "data" / "estudiantes_ejemplo.csv")


modelos = cargar_modelos()
meta = cargar_metadatos()
datos = cargar_datos()
ejemplos = cargar_ejemplos()


# ---------------------------------------------------------------------------
# Funciones de apoyo
# ---------------------------------------------------------------------------
def nivel_riesgo(p_desercion: float):
    if p_desercion >= UMBRAL_ALTO:
        return "Alto", "#D1495B", "Priorizar contacto con bienestar y consejería académica."
    if p_desercion >= UMBRAL_MEDIO:
        return "Medio", "#E0892B", "Hacer seguimiento al rendimiento y a la situación de pago."
    return "Bajo", "#2A9D8F", "Acompañamiento regular."


def nombre_legible(col: str) -> str:
    """Convierte el nombre interno de una columna transformada en texto claro."""
    tipo, _, nombre = col.partition("__")
    if tipo in ("num", "bin"):
        return pp.ETIQUETAS.get(nombre, nombre)
    for var in pp.CATEGORICAS:
        if nombre.startswith(var + "_"):
            valor = nombre[len(var) + 1:]
            if valor == "infrequent_sklearn":
                valor = "otra (poco frecuente)"
            elif var == "programa":
                valor = pp.PROGRAMAS.get(int(valor), valor)
            elif var == "modo_ingreso":
                valor = pp.MODO_INGRESO.get(int(valor), valor)
            elif var == "estado_civil":
                valor = pp.ESTADO_CIVIL.get(int(valor), valor)
            return f"{pp.ETIQUETAS.get(var, var)}: {valor}"
    return nombre


def explicar(modelo, registro: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Aporte de cada variable a la clase Deserción en la regresión logística.
    Se usa el coeficiente de Deserción menos el promedio de los coeficientes
    de las tres clases (lo que realmente mueve la probabilidad en un softmax)
    multiplicado por el valor transformado de la variable."""
    fe = modelo.named_steps["ingenieria"].transform(registro)
    prep = modelo.named_steps["preprocesamiento"]
    x = prep.transform(fe)[0]
    lr = modelo.named_steps["modelo"]
    k = list(lr.classes_).index("Deserción")
    w = lr.coef_[k] - lr.coef_.mean(axis=0)
    aporte = pd.DataFrame({"variable": [nombre_legible(c) for c in prep.get_feature_names_out()],
                           "aporte": x * w})
    aporte = aporte[aporte["aporte"].abs() > 1e-6]
    return aporte.reindex(aporte["aporte"].abs().sort_values(ascending=False).index).head(top)


def grafica_probabilidades(proba: dict):
    d = pd.DataFrame({"Resultado": list(proba), "Probabilidad": list(proba.values())})
    base = alt.Chart(d).encode(
        y=alt.Y("Resultado:N", sort=pp.ORDEN_CLASES, title=None, axis=EJE_Y),
        x=alt.X("Probabilidad:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")))
    barras = base.mark_bar(cornerRadiusEnd=4).encode(color=alt.Color("Resultado:N", scale=ESCALA, legend=None))
    texto = base.mark_text(align="left", dx=4).encode(text=alt.Text("Probabilidad:Q", format=".1%"))
    return (barras + texto).properties(height=150)


def validar(r: dict, escenario: str) -> list:
    errores = []
    semestres = [1] if escenario == "semestre1" else [1, 2] if escenario == "anual" else []
    for s in semestres:
        if r[f"s{s}_aprobadas"] > r[f"s{s}_inscritas"]:
            errores.append(f"Semestre {s}: no puede haber más unidades aprobadas que inscritas.")
        if r[f"s{s}_aprobadas"] > r[f"s{s}_evaluaciones"]:
            errores.append(f"Semestre {s}: las unidades aprobadas no pueden superar las evaluaciones presentadas.")
        if r[f"s{s}_aprobadas"] > 0 and r[f"s{s}_nota"] < 10:
            errores.append(f"Semestre {s}: si aprobó unidades, el promedio debe ser al menos 10/20.")
        if r[f"s{s}_aprobadas"] == 0 and r[f"s{s}_nota"] > 0:
            errores.append(f"Semestre {s}: sin unidades aprobadas el promedio registrado es 0.")
    return errores


def predecir_lote(df_in: pd.DataFrame, escenario: str) -> pd.DataFrame:
    modelo = modelos[escenario]
    cols = pp.columnas(escenario)["entradas"]
    X = df_in.copy()
    if "sin_registro_curricular" in cols and "sin_registro_curricular" not in X:
        X["sin_registro_curricular"] = 0
    proba = modelo.predict_proba(X[cols])
    salida = df_in.copy()
    salida.insert(0, "prediccion", modelo.predict(X[cols]))
    for i, c in enumerate(modelo.classes_):
        salida.insert(1 + i, f"p_{c.lower()}", proba[:, i].round(3))
    salida.insert(1, "riesgo_desercion", [nivel_riesgo(p)[0] for p in salida["p_deserción"]])
    return salida


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Alerta temprana")
    st.caption("ACA 1 · Aprendizaje Automático · CUN")
    escenario = st.radio(
        "¿Qué información tiene del estudiante?",
        options=list(pp.ESCENARIOS), format_func=pp.ESCENARIOS.get, index=2,
        help="Se usa un modelo distinto según el momento: entre más información, más precisa la predicción.")
    fila_esc = next(e for e in meta["escenarios"] if e["escenario"] == pp.ESCENARIOS[escenario])
    st.metric("F1 macro de este modelo (prueba)", num(fila_esc["F1 macro"], 3))
    st.metric("Detecta desertores (sensibilidad)", pct(fila_esc["Sensibilidad Deserción"]))
    st.divider()
    st.markdown("**Equipo · Grupo 7**\n\n- Anternol de los R. Bedoya Naranjo\n- Bonny D. B. Galindo Santos\n- Andrés A. López Herrera\n- Camilo J. Rodríguez Ayubi")
    st.markdown("**Docente:** Dr. Hamilton Rivera Flor")
    st.caption("Datos: Realinho et al. (2021), UCI Machine Learning Repository, CC BY 4.0.")

st.title("Alerta temprana de deserción estudiantil")
st.markdown(
    "Estima si un estudiante de pregrado terminará **desertando**, seguirá **matriculado** o se **graduará**, "
    "a partir de sus datos de ingreso, su situación económica y su rendimiento del primer año. "
    f"Modelo: **{meta['modelo']}** entrenado con {num(meta['n_train'])} estudiantes y evaluado con {num(meta['n_test'])} que no vio.")

tab_ind, tab_lote, tab_eda, tab_modelo, tab_info = st.tabs(
    ["🔮 Predicción individual", "📋 Predicción por lotes", "📊 Exploración de datos",
     "📈 Desempeño del modelo", "ℹ️ Acerca del proyecto"])

# ---------------------------------------------------------------------------
# 1. Predicción individual
# ---------------------------------------------------------------------------
with tab_ind:
    moda = lambda c: int(datos[c].mode()[0])
    med = lambda c: float(datos[c].median())

    c1, c2 = st.columns([3, 1])
    with c2:
        opciones_ej = ["(valores típicos)"] + ejemplos["id_estudiante"].tolist()
        elegido = st.selectbox("Cargar un estudiante de ejemplo", opciones_ej,
                               help="Estudiantes reales del conjunto de prueba (el modelo no los vio al entrenar).")
    ej = None if elegido == "(valores típicos)" else ejemplos.set_index("id_estudiante").loc[elegido]
    v = (lambda c, d: d if ej is None or pd.isna(ej.get(c, np.nan)) else ej[c])
    with c1:
        st.markdown("Complete los datos del estudiante o cargue uno de ejemplo y presione **Predecir**.")

    def idx(opciones, valor):
        opciones = list(opciones)
        return opciones.index(valor) if valor in opciones else 0

    with st.form("formulario"):
        st.markdown("##### 👤 Perfil y admisión")
        a1, a2, a3, a4 = st.columns(4)
        programa = a1.selectbox("Programa", list(pp.PROGRAMAS), format_func=pp.PROGRAMAS.get,
                                index=idx(pp.PROGRAMAS, int(v("programa", 9500))))
        modo = a2.selectbox("Modalidad de ingreso", list(pp.MODO_INGRESO), format_func=pp.MODO_INGRESO.get,
                            index=idx(pp.MODO_INGRESO, int(v("modo_ingreso", 1))))
        orden = a3.number_input("Orden de preferencia del programa", 0, 9, int(v("orden_opcion", 1)),
                                help="1 = era su primera opción")
        jornada = a4.radio("Jornada", [1, 0], format_func={1: "Diurna", 0: "Nocturna"}.get,
                           index=0 if int(v("jornada_diurna", 1)) == 1 else 1, horizontal=True)
        b1, b2, b3, b4 = st.columns(4)
        edad = b1.number_input("Edad al ingresar", 16, 75, int(v("edad_ingreso", 20)))
        genero = b2.radio("Género", [0, 1], format_func={0: "Mujer", 1: "Hombre"}.get,
                          index=int(v("genero_masculino", 0)), horizontal=True)
        civil = b3.selectbox("Estado civil", list(pp.ESTADO_CIVIL), format_func=pp.ESTADO_CIVIL.get,
                             index=idx(pp.ESTADO_CIVIL, int(v("estado_civil", 1))))
        internacional = b4.checkbox("Estudiante internacional", bool(v("internacional", 0)))
        c1_, c2_, c3_, c4_ = st.columns(4)
        nivel_prev = c1_.selectbox("Nivel de formación previa", ["Media", "Básica", "Técnica/tecnológica", "Superior"],
                                   index=idx(["Media", "Básica", "Técnica/tecnológica", "Superior"],
                                             pp.COD_A_NIVEL.get(int(v("formacion_previa", 1)), "Media")))
        nota_prev = c2_.number_input("Nota de formación previa (0-200)", 0.0, 200.0,
                                     float(v("nota_formacion_previa", med("nota_formacion_previa"))), step=0.5)
        nota_adm = c3_.number_input("Nota de admisión (0-200)", 0.0, 200.0,
                                    float(v("nota_admision", med("nota_admision"))), step=0.5)
        desplazado = c4_.checkbox("Vive fuera de su ciudad de origen", bool(v("desplazado", 1)))

        st.markdown("##### 💰 Situación económica y familiar")
        d1, d2, d3, d4 = st.columns(4)
        al_dia = d1.checkbox("Matrícula al día", bool(v("matricula_al_dia", 1)))
        deudor = d2.checkbox("Tiene deudas con la institución", bool(v("deudor", 0)))
        becado = d3.checkbox("Becado", bool(v("becado", 0)))
        nee = d4.checkbox("Necesidades educativas especiales", bool(v("necesidades_especiales", 0)))
        niveles = list(pp.CODIGO_NIVEL)
        grupos = list(pp.CODIGO_OCUPACION)
        e1, e2, e3, e4 = st.columns(4)
        ed_madre = e1.selectbox("Nivel educativo de la madre", niveles,
                                index=idx(niveles, pp.COD_A_NIVEL.get(int(v("educacion_madre", 1)), "Media")))
        ed_padre = e2.selectbox("Nivel educativo del padre", niveles,
                                index=idx(niveles, pp.COD_A_NIVEL.get(int(v("educacion_padre", 1)), "Media")))
        oc_madre = e3.selectbox("Ocupación de la madre", grupos,
                                index=idx(grupos, pp.grupo_ocupacion(v("ocupacion_madre", 9))))
        oc_padre = e4.selectbox("Ocupación del padre", grupos,
                                index=idx(grupos, pp.grupo_ocupacion(v("ocupacion_padre", 9))))

        registro = {
            "estado_civil": civil, "modo_ingreso": modo, "orden_opcion": orden, "programa": programa,
            "jornada_diurna": jornada, "formacion_previa": pp.CODIGO_NIVEL[nivel_prev],
            "nota_formacion_previa": nota_prev, "educacion_madre": pp.CODIGO_NIVEL[ed_madre],
            "educacion_padre": pp.CODIGO_NIVEL[ed_padre], "ocupacion_madre": pp.CODIGO_OCUPACION[oc_madre],
            "ocupacion_padre": pp.CODIGO_OCUPACION[oc_padre], "nota_admision": nota_adm,
            "desplazado": int(desplazado), "necesidades_especiales": int(nee), "deudor": int(deudor),
            "matricula_al_dia": int(al_dia), "genero_masculino": genero, "becado": int(becado),
            "edad_ingreso": edad, "internacional": int(internacional), "sin_registro_curricular": 0,
        }

        semestres = {"ingreso": [], "semestre1": [1], "anual": [1, 2]}[escenario]
        if semestres:
            st.markdown("##### 📚 Rendimiento académico (unidades curriculares = asignaturas)")
        for s in semestres:
            st.caption(f"Semestre {s}")
            f = st.columns(6)
            registro[f"s{s}_inscritas"] = f[0].number_input("Inscritas", 0, 30, int(v(f"s{s}_inscritas", 6)), key=f"i{s}_{elegido}")
            registro[f"s{s}_evaluaciones"] = f[1].number_input("Evaluaciones presentadas", 0, 50, int(v(f"s{s}_evaluaciones", 8)), key=f"e{s}_{elegido}")
            registro[f"s{s}_aprobadas"] = f[2].number_input("Aprobadas", 0, 30, int(v(f"s{s}_aprobadas", 5)), key=f"a{s}_{elegido}")
            registro[f"s{s}_nota"] = f[3].number_input("Promedio (0-20)", 0.0, 20.0, float(v(f"s{s}_nota", 12.5)), step=0.1, key=f"n{s}_{elegido}")
            registro[f"s{s}_convalidadas"] = f[4].number_input("Convalidadas", 0, 30, int(v(f"s{s}_convalidadas", 0)), key=f"c{s}_{elegido}")
            registro[f"s{s}_sin_evaluacion"] = f[5].number_input("Sin evaluación", 0, 30, int(v(f"s{s}_sin_evaluacion", 0)), key=f"w{s}_{elegido}")

        with st.expander("🌍 Contexto económico del año de ingreso (opcional)"):
            g1, g2, g3 = st.columns(3)
            registro["tasa_desempleo"] = g1.number_input("Desempleo (%)", 0.0, 30.0, float(v("tasa_desempleo", med("tasa_desempleo"))), step=0.1)
            registro["tasa_inflacion"] = g2.number_input("Inflación (%)", -5.0, 15.0, float(v("tasa_inflacion", med("tasa_inflacion"))), step=0.1)
            registro["pib"] = g3.number_input("Variación del PIB (%)", -10.0, 10.0, float(v("pib", med("pib"))), step=0.01)

        enviado = st.form_submit_button("Predecir", type="primary", width="stretch")

    if enviado:
        errores = validar(registro, escenario)
        if errores:
            for e in errores:
                st.error(e)
        else:
            modelo = modelos[escenario]
            X1 = pd.DataFrame([registro])[pp.columnas(escenario)["entradas"]]
            proba = dict(zip(modelo.classes_, modelo.predict_proba(X1)[0]))
            pred = max(proba, key=proba.get)
            nivel, color, accion = nivel_riesgo(proba["Deserción"])

            r1, r2 = st.columns([1, 1.4])
            with r1:
                st.markdown(f"""<div class="riesgo" style="background:{color}">
                    <p>Probabilidad de deserción</p><h3>Riesgo {nivel.lower()} · {pct(proba['Deserción'])}</h3>
                    <p>{accion}</p></div>""", unsafe_allow_html=True)
                st.markdown(f"Resultado más probable: **{pred}**")
                if ej is not None and "resultado" in datos.columns:
                    real = datos.loc[int(elegido.split("-")[1]), "resultado"]
                    st.caption(f"Resultado real de este estudiante en el dataset: **{real}**")
            with r2:
                st.altair_chart(grafica_probabilidades({c: proba[c] for c in pp.ORDEN_CLASES}), width="stretch")

            st.markdown("##### ¿Qué pesó más en esta predicción?")
            exp = explicar(modelo, X1)
            exp["efecto"] = np.where(exp["aporte"] > 0, "Aumenta el riesgo", "Reduce el riesgo")
            chart = alt.Chart(exp).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("aporte:Q", title="Aporte al riesgo de deserción (escala log-odds)"),
                y=alt.Y("variable:N", sort=None, title=None, axis=EJE_Y),
                color=alt.Color("efecto:N", scale=alt.Scale(domain=["Aumenta el riesgo", "Reduce el riesgo"],
                                                            range=["#D1495B", "#00798C"]), title=None),
                tooltip=["variable", alt.Tooltip("aporte:Q", format=".2f")]).properties(height=230)
            st.altair_chart(chart, width="stretch")
            st.caption("Aporte = coeficiente de la regresión logística × valor estandarizado de la variable. "
                       "Los valores numéricos se comparan contra el promedio de los estudiantes de entrenamiento.")

# ---------------------------------------------------------------------------
# 2. Predicción por lotes
# ---------------------------------------------------------------------------
with tab_lote:
    st.markdown(f"Suba un archivo CSV con las columnas del escenario **{pp.ESCENARIOS[escenario]}** "
                "y obtenga la predicción y el nivel de riesgo de cada estudiante.")
    cols_req = pp.columnas(escenario)["entradas"]
    l1, l2 = st.columns(2)
    l1.download_button("⬇️ Descargar archivo de ejemplo (25 estudiantes)",
                       ejemplos.to_csv(index=False).encode("utf-8"), "estudiantes_ejemplo.csv", "text/csv")
    usar_ejemplo = l2.toggle("Usar el archivo de ejemplo", value=True)
    archivo = st.file_uploader("Archivo CSV (separado por comas)", type="csv", disabled=usar_ejemplo)
    lote = ejemplos if usar_ejemplo else (pd.read_csv(archivo) if archivo is not None else None)

    if lote is not None:
        faltan = [c for c in cols_req if c not in lote.columns and c != "sin_registro_curricular"]
        if faltan:
            st.error("Al archivo le faltan estas columnas: " + ", ".join(faltan))
        else:
            res = predecir_lote(lote, escenario)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Estudiantes", len(res))
            k2.metric("Riesgo alto", int((res["riesgo_desercion"] == "Alto").sum()))
            k3.metric("Riesgo medio", int((res["riesgo_desercion"] == "Medio").sum()))
            k4.metric("Riesgo bajo", int((res["riesgo_desercion"] == "Bajo").sum()))
            vista = res.sort_values("p_deserción", ascending=False)
            cols_vista = [c for c in ["id_estudiante", "prediccion", "riesgo_desercion", "p_deserción",
                                      "p_matriculado", "p_graduado"] if c in vista.columns]
            st.dataframe(vista[cols_vista], width="stretch", hide_index=True,
                         column_config={
                             "id_estudiante": "Estudiante", "prediccion": "Predicción", "riesgo_desercion": "Riesgo",
                             "p_deserción": st.column_config.ProgressColumn(
                                 "P(deserción)", format="%.2f", min_value=0, max_value=1),
                             "p_matriculado": st.column_config.NumberColumn("P(matriculado)", format="%.2f"),
                             "p_graduado": st.column_config.NumberColumn("P(graduado)", format="%.2f")})
            st.download_button("⬇️ Descargar resultados", res.to_csv(index=False).encode("utf-8"),
                               "predicciones_desercion.csv", "text/csv")

# ---------------------------------------------------------------------------
# 3. Exploración de datos
# ---------------------------------------------------------------------------
with tab_eda:
    f1, f2 = st.columns([3, 1])
    progs = f1.multiselect("Filtrar por programa", sorted(datos["programa_nombre"].unique()),
                           placeholder="Todos los programas")
    jorn = f2.selectbox("Jornada", ["Todas", "Diurna", "Nocturna"])
    d = datos.copy()
    if progs:
        d = d[d["programa_nombre"].isin(progs)]
    if jorn != "Todas":
        d = d[d["jornada_diurna"] == (1 if jorn == "Diurna" else 0)]

    m = st.columns(4)
    m[0].metric("Estudiantes", num(len(d)))
    for i, c in enumerate(pp.ORDEN_CLASES, start=1):
        m[i].metric(c, pct((d["resultado"] == c).mean()))

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Distribución del resultado**")
        st.altair_chart(alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("count():Q", title="Estudiantes"),
            y=alt.Y("resultado:N", sort=pp.ORDEN_CLASES, title=None, axis=EJE_Y),
            color=alt.Color("resultado:N", scale=ESCALA, legend=None)).properties(height=240),
            width="stretch")
    with g2:
        st.markdown("**Tasa de deserción según factor**")
        factores = {"matricula_al_dia": ("Matrícula al día", "Matrícula NO al día"),
                    "deudor": ("Deudor", "No deudor"), "becado": ("Becado", "Sin beca"),
                    "genero_masculino": ("Hombre", "Mujer")}
        filas = []
        for col, (si, no) in factores.items():
            for val, et in ((1, si), (0, no)):
                sub = d[d[col] == val]
                if len(sub):
                    filas.append({"grupo": et, "tasa": (sub["resultado"] == "Deserción").mean(), "n": len(sub)})
        st.altair_chart(alt.Chart(pd.DataFrame(filas)).mark_bar(color="#D1495B", cornerRadiusEnd=4).encode(
            x=alt.X("tasa:Q", axis=alt.Axis(format="%"), title="Deserción"),
            y=alt.Y("grupo:N", sort="-x", title=None, axis=EJE_Y),
            tooltip=["grupo", alt.Tooltip("tasa:Q", format=".1%"), "n"]).properties(height=240),
            width="stretch")

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("**Unidades aprobadas en el 2.º semestre**")
        st.altair_chart(alt.Chart(d.dropna(subset=["s2_aprobadas"])).mark_boxplot(extent=1.5).encode(
            x=alt.X("resultado:N", sort=pp.ORDEN_CLASES, title=None, axis=EJE_X),
            y=alt.Y("s2_aprobadas:Q", title="Unidades aprobadas"),
            color=alt.Color("resultado:N", scale=ESCALA, legend=None)).properties(height=260),
            width="stretch")
    with g4:
        st.markdown("**Resultado según edad de ingreso**")
        d2 = d.assign(edad=pd.cut(d["edad_ingreso"], [16, 19, 22, 25, 30, 40, 75],
                                  labels=["17-19", "20-22", "23-25", "26-30", "31-40", "41+"]).astype(str))
        st.altair_chart(alt.Chart(d2).mark_bar().encode(
            x=alt.X("edad:N", sort=["17-19", "20-22", "23-25", "26-30", "31-40", "41+"], title="Edad", axis=EJE_X),
            y=alt.Y("count():Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
            color=alt.Color("resultado:N", scale=ESCALA, sort=pp.ORDEN_CLASES, title=None),
            order=alt.Order("orden:Q")).transform_calculate(
                orden="datum.resultado == 'Deserción' ? 0 : datum.resultado == 'Matriculado' ? 1 : 2"
            ).properties(height=260), width="stretch")

    st.markdown("**Deserción por programa**")
    tp = (d.groupby("programa_nombre")["resultado"].apply(lambda s: (s == "Deserción").mean())
          .reset_index(name="tasa").merge(d["programa_nombre"].value_counts().rename("n").reset_index()))
    st.altair_chart(alt.Chart(tp).mark_bar(color="#D1495B", cornerRadiusEnd=4).encode(
        x=alt.X("tasa:Q", axis=alt.Axis(format="%"), title="Tasa de deserción"),
        y=alt.Y("programa_nombre:N", sort="-x", title=None, axis=EJE_Y),
        tooltip=["programa_nombre", alt.Tooltip("tasa:Q", format=".1%"), "n"]).properties(height=430),
        width="stretch")

# ---------------------------------------------------------------------------
# 4. Desempeño del modelo
# ---------------------------------------------------------------------------
with tab_modelo:
    mt, ic = meta["metricas_test"], meta["ic95"]
    st.markdown(f"Resultados del modelo principal (**{meta['modelo']}**, "
                f"C = {num(meta['hiperparametros'].get('C'), 1)}, pesos de clase balanceados) sobre "
                f"{meta['n_test']} estudiantes de prueba que no se usaron para entrenar.")
    k = st.columns(5)
    k[0].metric("Exactitud", pct(mt["Exactitud"]), help=f"IC 95 %: {pct(ic['Exactitud'][0])} – {pct(ic['Exactitud'][1])}")
    k[1].metric("F1 macro", num(mt["F1 macro"], 3), help=f"IC 95 %: {num(ic['F1 macro'][0], 3)} – {num(ic['F1 macro'][1], 3)}")
    k[2].metric("ROC-AUC", num(mt["ROC-AUC (OvR macro)"], 3))
    k[3].metric("Exactitud balanceada", pct(mt["Exactitud balanceada"]))
    k[4].metric("Kappa de Cohen", num(mt["Kappa de Cohen"], 3))

    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Matriz de confusión (prueba)**")
        mc = meta["matriz_confusion"]
        filas = [{"Real": r, "Predicho": p, "n": mc["valores"][i][j],
                  "pct": mc["valores"][i][j] / sum(mc["valores"][i])}
                 for i, r in enumerate(mc["clases"]) for j, p in enumerate(mc["clases"])]
        dm = pd.DataFrame(filas)
        base = alt.Chart(dm).encode(x=alt.X("Predicho:N", sort=mc["clases"], axis=EJE_X), y=alt.Y("Real:N", sort=mc["clases"]))
        st.altair_chart((base.mark_rect().encode(color=alt.Color("pct:Q", scale=alt.Scale(scheme="blues"), legend=None))
                         + base.mark_text(fontSize=14).encode(
                             text=alt.Text("n:Q"),
                             color=alt.condition("datum.pct > 0.5", alt.value("white"), alt.value("#1f2937"))))
                        .properties(height=300), width="stretch")
    with h2:
        st.markdown("**Métricas por clase**")
        pc = pd.DataFrame(meta["por_clase"]).T.rename(columns={"precision": "Precisión", "recall": "Sensibilidad",
                                                               "f1-score": "F1", "support": "Casos"})
        pc["Casos"] = pc["Casos"].astype(int)
        st.dataframe(pc.style.format({"Precisión": "{:.3f}", "Sensibilidad": "{:.3f}", "F1": "{:.3f}"}),
                     width="stretch")
        st.info(f"El modelo detecta al **{pct(meta['por_clase']['Deserción']['recall'])}** de los desertores y acierta el "
                f"**{pct(meta['por_clase']['Deserción']['precision'])}** de las veces que da la alerta. "
                "*Matriculado* es la clase más difícil: son estudiantes atrasados que aún no se van.")

    h3, h4 = st.columns(2)
    with h3:
        st.markdown("**Comparación de algoritmos (validación cruzada, 5 pliegues)**")
        cv = pd.DataFrame(meta["comparacion_cv"])
        st.altair_chart(alt.Chart(cv).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("F1 macro:Q", scale=alt.Scale(domain=[0, 0.8])),
            y=alt.Y("modelo:N", sort="-x", title=None, axis=EJE_Y),
            color=alt.condition(alt.datum.modelo == meta["modelo"], alt.value("#00798C"), alt.value("#9DB4C0")),
            tooltip=["modelo", alt.Tooltip("F1 macro:Q", format=".3f"), alt.Tooltip("± F1:Q", format=".3f"),
                     alt.Tooltip("ROC-AUC:Q", format=".3f")]).properties(height=280), width="stretch")
    with h4:
        st.markdown("**Variables más influyentes (importancia por permutación)**")
        imp = pd.DataFrame(meta["importancia"]).head(10)
        imp["nombre"] = imp["variable"].map(pp.ETIQUETAS)
        st.altair_chart(alt.Chart(imp).mark_bar(color="#00798C", cornerRadiusEnd=4).encode(
            x=alt.X("caida_f1:Q", title="Caída del F1 macro al permutar"),
            y=alt.Y("nombre:N", sort="-x", title=None, axis=EJE_Y),
            tooltip=["nombre", alt.Tooltip("caida_f1:Q", format=".3f")]).properties(height=280),
            width="stretch")

    st.markdown("**¿Qué tan temprano se puede alertar?**")
    esc = pd.DataFrame(meta["escenarios"]).set_index("escenario")
    st.dataframe(esc.style.format({"Exactitud": "{:.1%}", "F1 macro": "{:.3f}",
                                   "Sensibilidad Deserción": "{:.1%}", "ROC-AUC": "{:.3f}"}),
                 width="stretch")

# ---------------------------------------------------------------------------
# 5. Acerca del proyecto
# ---------------------------------------------------------------------------
with tab_info:
    st.markdown("""
#### Objetivo general
Desarrollar una solución de Machine Learning clásico que, a partir de datos académicos, demográficos y
socioeconómicos, clasifique a cada estudiante en *Deserción*, *Matriculado* o *Graduado*, desde la limpieza de
datos hasta su despliegue en esta aplicación.

#### Pregunta orientadora
¿Cómo transformar datos de sectores estratégicos (educación) en conocimiento predictivo mediante técnicas
clásicas de Machine Learning y presentar los resultados en una interfaz interactiva accesible para usuarios finales?

#### Ruta metodológica
| Paso | Qué se hizo |
|---|---|
| 1. Exploración y EDA | 4.424 estudiantes, 36 variables, 3 clases (49,9 % graduados, 32,1 % desertores, 17,9 % matriculados). |
| 2. Limpieza y datos | Se detectaron 180 registros con ceros que eran ausencia de información (programa de Animación), una nota inconsistente y una variable redundante. Imputación por mediana dentro del pipeline, agrupación de códigos y 5 variables nuevas. |
| 3. Modelado clásico | 7 algoritmos comparados con validación cruzada estratificada; ajuste de hiperparámetros con GridSearchCV. Se eligió regresión logística por empate técnico con Random Forest, menor sobreajuste e interpretabilidad. |
| 4. Evaluación | Exactitud, F1 macro, ROC-AUC, kappa, matriz de confusión, intervalos de confianza por bootstrap e importancia por permutación. |
| 5. Despliegue | Esta app: predicción individual con explicación, predicción por lotes, exploración y métricas. |

#### Uso responsable
El modelo se entrenó con datos de una institución portuguesa (2008-2019). Sirve para priorizar acompañamiento,
no para tomar decisiones sobre un estudiante sin el criterio de un profesional. Para usarlo en Colombia habría
que reentrenarlo con datos locales.

#### Referencias
- Laboratorio de Economía de la Educación. (2023). *Deserción en la educación superior en Colombia* (Informe n.º 74). Pontificia Universidad Javeriana.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146
- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict students' dropout and academic success* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
""")

st.markdown('<p class="pie">ACA 1 · Aprendizaje Automático · Especialización en Inteligencia Artificial · CUN · 2026</p>',
            unsafe_allow_html=True)
