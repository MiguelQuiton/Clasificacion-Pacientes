import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Clasificador de pacientes")
st.title("Prototipo de clasificación de pacientes")


# Datos sintéticos para demostrar el funcionamiento del árbol.
# Los rangos y criterios son ficticios; deben reemplazarse por datos
# y etiquetas definidos para el proyecto.
# Carga datos que ya tienen una clasificación asignada
datos = pd.read_csv(
    "pacientes.csv",
    sep=None,
    engine="python",
    encoding="utf-8-sig",
)

# Quita espacios accidentales en los encabezados
datos.columns = datos.columns.astype(str).str.strip()


columnas_entrada = [
    "edad",
    "temperatura",
    "presion_sistolica",
    "presion_diastolica",
    "frecuencia_cardiaca",
    "saturacion_oxigeno",
    "sintomas",
    "antecedentes_relevantes",
]

columna_objetivo = "clasificacion"

# Comprueba que el CSV tenga todas las columnas necesarias
columnas_necesarias = columnas_entrada + [columna_objetivo]
faltantes = [col for col in columnas_necesarias if col not in datos.columns]

if faltantes:
    st.error(f"Faltan estas columnas en pacientes.csv: {faltantes}")
    st.stop()

# Elimina filas que no tengan datos o clasificación
datos = datos.dropna(subset=columnas_necesarias)

# X son las variables de entrada; y son las respuestas que aprenderá el árbol
X = datos[columnas_entrada]
y = datos[columna_objetivo].astype(str).str.strip().str.capitalize()

# Avisa si el archivo no contiene las tres categorías
categorias_esperadas = {"Baja", "Media", "Alta"}
categorias_encontradas = set(y.unique())

if not categorias_esperadas.issubset(categorias_encontradas):
    st.error(
        "El CSV debe incluir ejemplos clasificados como Baja, Media y Alta."
    )
    st.stop()

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y,
)

modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(X_entrenamiento, y_entrenamiento)
precision = accuracy_score(y_prueba, modelo.predict(X_prueba))

st.subheader("Datos de entrada")

with st.form("formulario_paciente"):
    edad = st.number_input("Edad", min_value=0, max_value=120, value=30)
    temperatura = st.number_input(
        "Temperatura (°C)", min_value=30.0, max_value=45.0,
        value=36.5, step=0.1
    )
    presion_sistolica = st.number_input(
        "Presión sistólica", min_value=50, max_value=250, value=120
    )
    presion_diastolica = st.number_input(
        "Presión diastólica", min_value=30, max_value=160, value=80
    )
    frecuencia_cardiaca = st.number_input(
        "Frecuencia cardíaca (latidos por minuto)",
        min_value=20, max_value=250, value=75
    )
    saturacion_oxigeno = st.number_input(
        "Saturación de oxígeno (%)", min_value=50, max_value=100, value=98
    )

    sintomas = st.selectbox(
        "Nivel de síntomas (valor de ejemplo)",
        options=[0, 1, 2, 3],
        format_func=lambda valor: {
            0: "0 - Ninguno",
            1: "1 - Leve",
            2: "2 - Moderado",
            3: "3 - Alto",
        }[valor],
    )

    antecedentes_relevantes = st.selectbox(
        "¿Presenta antecedentes relevantes?",
        options=[0, 1],
        format_func=lambda valor: "No" if valor == 0 else "Sí",
    )

    enviar = st.form_submit_button("Clasificar")

if enviar:
    nuevo_registro = pd.DataFrame([{
        "edad": edad,
        "temperatura": temperatura,
        "presion_sistolica": presion_sistolica,
        "presion_diastolica": presion_diastolica,
        "frecuencia_cardiaca": frecuencia_cardiaca,
        "saturacion_oxigeno": saturacion_oxigeno,
        "sintomas": sintomas,
        "antecedentes_relevantes": antecedentes_relevantes,
    }])

    prediccion = modelo.predict(nuevo_registro)[0]
    probabilidades = modelo.predict_proba(nuevo_registro)[0]

    st.subheader("Resultado del prototipo")
    st.write(f"**Clasificación estimada: {prediccion}**")
