import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Clasificador de pacientes", page_icon="🌿")
st.title("Prototipo de clasificación de pacientes")


# Datos sintéticos para demostrar el funcionamiento del árbol.
# Los rangos y criterios son ficticios; deben reemplazarse por datos
# y etiquetas definidos para el proyecto.
rng = np.random.default_rng(42)
cantidad_registros = 600

datos = pd.DataFrame({
    "edad": rng.integers(1, 100, size=cantidad_registros),
    "temperatura": rng.uniform(35.0, 41.0, size=cantidad_registros).round(1),
    "presion_sistolica": rng.integers(80, 201, size=cantidad_registros),
    "presion_diastolica": rng.integers(50, 131, size=cantidad_registros),
    "frecuencia_cardiaca": rng.integers(40, 161, size=cantidad_registros),
    "saturacion_oxigeno": rng.integers(75, 101, size=cantidad_registros),
    "sintomas": rng.integers(0, 4, size=cantidad_registros),
    "antecedentes_relevantes": rng.integers(0, 2, size=cantidad_registros),
})

# Etiquetas artificiales solo para que el prototipo sea ejecutable.
# No representan un protocolo de triaje ni una recomendación médica.
puntaje_demo = (
    (datos["temperatura"] > 38.0).astype(int)
    + (datos["sintomas"] >= 2).astype(int)
    + (datos["saturacion_oxigeno"] < 92).astype(int) * 2
    + (datos["frecuencia_cardiaca"] > 110).astype(int)
    + (datos["presion_sistolica"] < 90).astype(int) * 2
    + datos["antecedentes_relevantes"]
)

datos["clasificacion"] = pd.cut(
    puntaje_demo,
    bins=[-np.inf, 1, 3, np.inf],
    labels=["Baja", "Media", "Alta"],
)

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

X = datos[columnas_entrada]
y = datos["clasificacion"]

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.2,
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

    st.write("Probabilidades estimadas:")
    for categoria, probabilidad in zip(modelo.classes_, probabilidades):
        st.write(f"- {categoria}: {probabilidad:.1%}")
