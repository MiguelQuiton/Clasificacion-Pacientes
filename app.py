import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(page_title="Clasificador de pacientes")
st.title("Prototipo de clasificación de pacientes")


# Carga el archivo pacientes.csv desde la misma carpeta que app.py.
datos = pd.read_csv(
    "pacientes.csv",
    sep=None,
    engine="python",
    encoding="utf-8-sig",
)

# Quita espacios accidentales de los encabezados.
datos.columns = datos.columns.astype(str).str.strip()


# Columnas numéricas que usará el árbol.
columnas_numericas = [
    "edad",
    "temperatura",
    "frecuencia_cardiaca",
    "saturacion_oxigeno",
    "sintomas",
    "antecedentes_relevantes",
]

# El ejemplo clínico también se usa como parámetro de entrada.
columnas_entrada = columnas_numericas + ["ejemplo_clinico"]
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


# Verifica que haya suficientes ejemplos por nivel para dividir los datos.
conteo_niveles = y.value_counts()

if conteo_niveles.min() < 2:
    st.error(
        "Cada nivel debe tener al menos 2 filas para poder entrenar "
        "y evaluar el modelo. Agrega más ejemplos a los niveles con pocos datos."
    )
    st.write("Cantidad de filas por nivel:")
    st.write(conteo_niveles)
    st.stop()


# Divide los datos en entrenamiento y prueba.
try:
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
except ValueError as error:
    st.error(f"No se pudieron dividir los datos: {error}")
    st.stop()


# Codifica el texto del ejemplo clínico y deja pasar las columnas numéricas.
preprocesador = ColumnTransformer(
    transformers=[
        ("numericas", "passthrough", columnas_numericas),
        (
            "ejemplo_clinico",
            OneHotEncoder(handle_unknown="ignore"),
            ["ejemplo_clinico"],
        ),
    ]
)

# El pipeline primero prepara los datos y luego entrena el árbol.
modelo = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador),
        ("arbol", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ]
)

modelo.fit(X_entrenamiento, y_entrenamiento)

predicciones_prueba = modelo.predict(X_prueba)
precision = accuracy_score(y_prueba, predicciones_prueba)


st.subheader("Datos de entrada")

# Las opciones del selector salen de los ejemplos del CSV.
ejemplos_disponibles = sorted(
    datos["ejemplo_clinico"].astype(str).unique()
)

with st.form("formulario_paciente"):
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
        options=[0, 1, 2, 3],
    )

    antecedentes_relevantes = st.selectbox(
        "¿Tiene antecedentes relevantes?",
        options=[0, 1],
        format_func=lambda valor: "No" if valor == 0 else "Sí",
    )

    ejemplo_clinico = st.selectbox(
        "Ejemplo clínico",
        options=ejemplos_disponibles,
    )

    enviar = st.form_submit_button("Clasificar")


if enviar:
    nuevo_registro = pd.DataFrame(
        [
            {
                "edad": edad,
                "temperatura": temperatura,
                "frecuencia_cardiaca": frecuencia_cardiaca,
                "saturacion_oxigeno": saturacion_oxigeno,
                "sintomas": sintomas,
                "antecedentes_relevantes": antecedentes_relevantes,
                "ejemplo_clinico": ejemplo_clinico,
            }
        ]
    )

    prediccion = modelo.predict(nuevo_registro)[0]
    probabilidades = modelo.predict_proba(nuevo_registro)[0]
    clases = modelo.named_steps["arbol"].classes_

    st.subheader("Resultado del prototipo")
    st.write(f"**Clasificación estimada: {prediccion}**")

    st.write("Probabilidades estimadas por el modelo:")
    for categoria, probabilidad in zip(clases, probabilidades):
        st.write(f"- {categoria}: {probabilidad:.1%}")


st.write(f"Precisión en el conjunto de prueba: {precision:.1%}")
