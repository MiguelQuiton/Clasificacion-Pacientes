from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Clasificador de pacientes")
st.title("Clasificación de pacientes")
st.caption("Prototipo educativo para probar con casos simulados.")


# Carga el archivo pacientes.csv desde la misma carpeta que app.py.
datos = pd.read_csv(
    Path(__file__).with_name("Pacientes.csv"),
    sep=None,
    engine="python",
    encoding="utf-8-sig",
)

# Quita espacios accidentales de los encabezados.
datos.columns = datos.columns.astype(str).str.strip()


# Variables numéricas del formulario.
columnas_numericas = [
    "edad",
    "temperatura",
    "frecuencia_cardiaca",
    "saturacion_oxigeno",
    "sintomas",
    "antecedentes_relevantes",
]

columna_texto = "ejemplo_clinico"
columnas_entrada = columnas_numericas + [columna_texto]
columna_objetivo = "clasificacion"
columnas_necesarias = columnas_entrada + [columna_objetivo]


# Comprueba que el CSV contenga todas las columnas necesarias.
faltantes = [
    columna
    for columna in columnas_necesarias
    if columna not in datos.columns
]

if faltantes:
    st.error(f"Faltan estas columnas en pacientes.csv: {faltantes}")
    st.stop()


# Quita filas que tengan vacíos en las variables usadas.
datos = datos.dropna(subset=columnas_necesarias).copy()

if datos.empty:
    st.error("El archivo pacientes.csv no tiene filas completas para entrenar.")
    st.stop()


datos[columna_texto] = datos[columna_texto].astype(str).str.strip()
datos[columna_objetivo] = datos[columna_objetivo].astype(str).str.strip()
datos = datos.drop_duplicates(subset=columnas_necesarias)

# Prepara los datos de entrada (X) y las clasificaciones esperadas (y).
X = datos[columnas_entrada]
y = datos[columna_objetivo].astype(str).str.strip()

# Asegura que existan las cinco clasificaciones.
niveles_esperados = {
    "Nivel 1",
    "Nivel 2",
    "Nivel 3",
    "Nivel 4",
    "Nivel 5",
}
niveles_encontrados = set(y.unique())

if not niveles_esperados.issubset(niveles_encontrados):
    st.error(
        "El CSV debe tener ejemplos clasificados como "
        "Nivel 1, Nivel 2, Nivel 3, Nivel 4 y Nivel 5."
    )
    st.stop()


@st.cache_resource
def entrenar_modelo(X, y):
    grupos = X[columna_texto].str.normalize("NFKD").str.encode(
        "ascii", errors="ignore"
    ).str.decode("ascii").str.casefold().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    particiones = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    niveles = set(y)

    for entrenamiento, prueba in particiones.split(X, y, grupos):
        if set(y.iloc[entrenamiento]) == niveles and set(y.iloc[prueba]) == niveles:
            break
    else:
        raise ValueError("Se necesitan más descripciones distintas por nivel para evaluar el modelo.")

    preprocesador = ColumnTransformer(
        [
            ("numericas", StandardScaler(), columnas_numericas),
            ("descripcion", TfidfVectorizer(
                strip_accents="unicode", ngram_range=(1, 2)
            ), columna_texto),
        ]
    )
    modelo = make_pipeline(
        preprocesador,
        MLPClassifier(hidden_layer_sizes=(24,), solver="lbfgs", max_iter=300, random_state=42),
    )
    modelo.fit(X.iloc[entrenamiento], y.iloc[entrenamiento])
    exactitud = accuracy_score(y.iloc[prueba], modelo.predict(X.iloc[prueba]))
    return modelo, exactitud


try:
    modelo, exactitud = entrenar_modelo(X, y)
except ValueError as error:
    st.error(f"No se pudo entrenar el modelo: {error}")
    st.stop()


st.subheader("Datos de entrada")

edad = st.number_input(
    "Edad",
    min_value=0,
    max_value=120,
    value=30,
)

temperatura = st.number_input(
    "Temperatura (°C)",
    min_value=30.0,
    max_value=45.0,
    value=36.5,
    step=0.1,
)

frecuencia_cardiaca = st.number_input(
    "Frecuencia cardíaca (latidos por minuto)",
    min_value=20,
    max_value=250,
    value=75,
)

saturacion_oxigeno = st.number_input(
    "Saturación de oxígeno (%)",
    min_value=50,
    max_value=100,
    value=98,
)

sintomas = st.selectbox(
    "Cantidad de síntomas",
    options=range(5),
)

antecedentes_relevantes = st.selectbox(
    "¿Tiene antecedentes relevantes?",
    options=[0, 1],
    format_func=lambda valor: "No" if valor == 0 else "Sí",
)

antecedentes_especificos = []
if antecedentes_relevantes:
    antecedentes_especificos = st.multiselect(
        "Selecciona uno o más antecedentes específicos",
        options=[
            "Asma estable (sin exacerbación; sin relación aguda en este caso simulado)",
            "Hipertensión arterial",
            "Diabetes",
            "Enfermedad cardíaca",
            "Enfermedad pulmonar crónica",
            "Otro antecedente",
        ],
        help="El modelo usa si hay antecedentes. La selección permite detallarlos en el resultado.",
    )

descripcion = st.text_area("Describe qué está pasando", max_chars=1000)
enviar = st.button("Clasificar")


if enviar:
    if not descripcion.strip():
        st.warning("Escribe una descripción del caso.")
    else:
        nuevo_registro = pd.DataFrame([{
            "edad": edad,
            "temperatura": temperatura,
            "frecuencia_cardiaca": frecuencia_cardiaca,
            "saturacion_oxigeno": saturacion_oxigeno,
            "sintomas": sintomas,
            "antecedentes_relevantes": antecedentes_relevantes,
            columna_texto: descripcion.strip(),
        }])
        probabilidades = modelo.predict_proba(nuevo_registro)[0]
        nivel = modelo.classes_[probabilidades.argmax()]
        st.subheader("Resultado del prototipo")
        st.write(f"**Clasificación estimada: {nivel}**")
        if antecedentes_especificos:
            st.write("Antecedentes seleccionados")
            for antecedente in antecedentes_especificos:
                st.write(f"- {antecedente}")

        with st.expander("Probabilidades estimadas por el modelo"):
            for categoria, probabilidad in zip(modelo.classes_, probabilidades):
                st.write(f"- {categoria}: {probabilidad:.1%}")


st.write(f"Exactitud en el conjunto de prueba: {exactitud:.1%}")
