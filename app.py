import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(page_title="Clasificador de pacientes")
st.title("Clasificación educativa de pacientes")

# El CSV completo debe estar en la misma carpeta que app.py.
archivo_datos = "Pacientes.csv"

try:
    datos = pd.read_csv(
        archivo_datos,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )
except FileNotFoundError:
    st.error(
        f"No se encontró {archivo_datos}. "
        "Guarda el CSV en la misma carpeta que app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"No se pudo leer el archivo: {error}")
    st.stop()


# Limpia los encabezados.
datos.columns = datos.columns.astype(str).str.strip()


# Variables numéricas originales.
columnas_numericas = [
    "edad",
    "temperatura",
    "frecuencia_cardiaca",
    "saturacion_oxigeno",
    "sintomas",
    "antecedentes_relevantes",
]

columna_caso = "ejemplo_clinico"
columna_antecedente = "antecedente_especifico"
columna_objetivo = "clasificacion"

columnas_requeridas = (
    columnas_numericas
    + [columna_caso, columna_antecedente, columna_objetivo]
)

faltantes = [
    columna
    for columna in columnas_requeridas
    if columna not in datos.columns
]

if faltantes:
    st.error(f"Faltan estas columnas en el CSV: {faltantes}")
    st.write("Columnas encontradas:", datos.columns.tolist())
    st.stop()


# Quita filas incompletas y limpia las columnas de texto.
datos = datos.dropna(subset=columnas_requeridas).copy()

datos[columna_caso] = (
    datos[columna_caso].astype(str).str.strip()
)
datos[columna_antecedente] = (
    datos[columna_antecedente].astype(str).str.strip()
)
datos[columna_objetivo] = (
    datos[columna_objetivo].astype(str).str.strip()
)

# Convierte las columnas numéricas.
for columna in columnas_numericas:
    datos[columna] = pd.to_numeric(datos[columna], errors="coerce")

datos = datos.dropna(subset=columnas_numericas).copy()

if datos.empty:
    st.error("No hay filas completas para entrenar el modelo.")
    st.stop()


# Crea una variable binaria para cada antecedente específico.
# Esto permite que el formulario acepte varios antecedentes a la vez.
antecedentes_en_csv = sorted(
    antecedente
    for antecedente in datos[columna_antecedente].unique()
    if antecedente.lower() != "ninguno"
)

antecedente_a_columna = {
    antecedente: f"antecedente_{indice}"
    for indice, antecedente in enumerate(antecedentes_en_csv)
}

for antecedente, nombre_columna in antecedente_a_columna.items():
    datos[nombre_columna] = (
        datos[columna_antecedente] == antecedente
    ).astype(int)


columnas_indicadoras_antecedentes = list(
    antecedente_a_columna.values()
)

columnas_numericas_modelo = (
    columnas_numericas + columnas_indicadoras_antecedentes
)

columnas_entrada = columnas_numericas_modelo + [columna_caso]

X = datos[columnas_entrada]
y = datos[columna_objetivo]


# Comprueba que el CSV incluya los cinco niveles.
niveles_esperados = {
    "Nivel 1",
    "Nivel 2",
    "Nivel 3",
    "Nivel 4",
    "Nivel 5",
}

niveles_encontrados = set(y.unique())

if not niveles_esperados.issubset(niveles_encontrados):
    faltan_niveles = sorted(niveles_esperados - niveles_encontrados)
    st.error(f"Faltan ejemplos de estos niveles en el CSV: {faltan_niveles}")
    st.stop()


# Divide los datos para entrenar y calcular la precisión de prueba.
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


# Las variables numéricas y los indicadores de antecedentes pasan al árbol.
# El ejemplo clínico se codifica como una variable categórica.
preprocesador = ColumnTransformer(
    transformers=[
        ("numericas", "passthrough", columnas_numericas_modelo),
        (
            "caso_clinico",
            OneHotEncoder(handle_unknown="ignore"),
            [columna_caso],
        ),
    ]
)

modelo = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador),
        (
            "arbol",
            DecisionTreeClassifier(max_depth=12, random_state=42),
        ),
    ]
)

modelo.fit(X_entrenamiento, y_entrenamiento)

predicciones_prueba = modelo.predict(X_prueba)
precision = accuracy_score(y_prueba, predicciones_prueba)


# Las opciones se obtienen de sus columnas correspondientes.
casos_clinicos_disponibles = sorted(
    caso
    for caso in datos[columna_caso].unique()
    if caso.lower() != "ninguno"
)

antecedentes_disponibles = [
    antecedente
    for antecedente in antecedentes_en_csv
    if "no especificado" not in antecedente.lower()
    and "pendiente de completar" not in antecedente.lower()
]


st.subheader("Datos de entrada")

with st.container(border=True):
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

    sintomas = st.number_input(
        "Cantidad de síntomas",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )

    antecedentes_relevantes = st.selectbox(
        "¿Tiene antecedentes relevantes?",
        options=[0, 1],
        format_func=lambda valor: "No" if valor == 0 else "Sí",
    )

    if antecedentes_relevantes == 1:
        antecedentes_seleccionados = st.multiselect(
            "Selecciona uno o más antecedentes específicos",
            options=antecedentes_disponibles,
            placeholder="Selecciona los antecedentes",
        )
    else:
        antecedentes_seleccionados = []

    caso_clinico = st.selectbox(
        "Caso clínico",
        options=["Ninguno"] + casos_clinicos_disponibles,
    )

    enviar = st.button("Clasificar")


if enviar:
    # Arma el registro con los datos numéricos y el caso clínico.
    registro = {
        "edad": edad,
        "temperatura": temperatura,
        "frecuencia_cardiaca": frecuencia_cardiaca,
        "saturacion_oxigeno": saturacion_oxigeno,
        "sintomas": sintomas,
        "antecedentes_relevantes": antecedentes_relevantes,
        "ejemplo_clinico": caso_clinico,
    }

    # Activa un indicador para cada antecedente seleccionado.
    for antecedente, nombre_columna in antecedente_a_columna.items():
        registro[nombre_columna] = int(
            antecedente in antecedentes_seleccionados
        )

    nuevo_registro = pd.DataFrame(
        [registro],
        columns=columnas_entrada,
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
