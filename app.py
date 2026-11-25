import json
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


def graficos_del_caso(modelo, registro, columnas_numericas, antecedentes):
    """Devuelve los dos gráficos DOT y las decisiones ordenadas del caso."""
    preparador = modelo.named_steps["preprocesamiento"]
    clasificador = modelo.named_steps["arbol"]
    estructura = clasificador.tree_
    fila = preparador.transform(registro)
    visitados = set(clasificador.decision_path(fila).indices)
    hoja = clasificador.apply(fila)[0]

    # El orden coincide con las dos salidas del ColumnTransformer.
    unidades = {
        "edad": "años", "temperatura": "°C", "frecuencia_cardiaca": "lat/min",
        "saturacion_oxigeno": "%", "sintomas": "", "antecedentes_relevantes": "",
    }
    etiquetas = {
        "edad": "Edad", "temperatura": "Temperatura",
        "frecuencia_cardiaca": "Frecuencia cardíaca",
        "saturacion_oxigeno": "Saturación de oxígeno", "sintomas": "Cantidad de síntomas",
        "antecedentes_relevantes": "Tiene antecedentes relevantes",
    }
    variables = [(etiquetas[c], unidades[c], c == "antecedentes_relevantes")
                 for c in columnas_numericas]
    variables.extend((f"Antecedente: {nombre}", "", True) for nombre in antecedentes)
    vectorizador = preparador.named_transformers_["descripcion"]
    variables.extend((f"Peso de «{termino}»", "", False)
                     for termino in vectorizador.get_feature_names_out())

    inicio = ['digraph Arbol {', 'graph [rankdir=TB];',
              'node [shape=box, style="rounded,filled", fontname="Arial"];']
    completo = inicio.copy()
    camino = ['digraph Caso {', 'graph [rankdir=TB, nodesep=0.36, ranksep=0.38, bgcolor=white];',
              'node [shape=box, style="filled,rounded", fontname="Arial", fontsize=16, fontcolor="#182C40", margin="0.15,0.10"];',
              'edge [fontname="Arial Bold", fontsize=14, arrowsize=0.8];']
    visibles = set(visitados)
    for padre in visitados:
        if padre != hoja:
            visibles.update((estructura.children_left[padre], estructura.children_right[padre]))
    pasos = []
    numero_por_nodo = {}
    nodo = 0
    while nodo != hoja:
        numero_por_nodo[nodo] = len(numero_por_nodo) + 1
        izquierda = estructura.children_left[nodo]
        nodo = izquierda if izquierda in visitados else estructura.children_right[nodo]

    for nodo in range(estructura.node_count):
        izquierda = estructura.children_left[nodo]
        derecha = estructura.children_right[nodo]
        en_ruta = nodo in visitados
        if izquierda == derecha:
            clase = clasificador.classes_[estructura.value[nodo, 0].argmax()]
            texto = f"Nivel estimado: {clase}"
            forma = "ellipse"
        else:
            indice = estructura.feature[nodo]
            nombre, unidad, binaria = variables[indice]
            umbral = estructura.threshold[nodo]
            sufijo = f" {unidad}" if unidad else ""
            if binaria:
                texto = f"¿{nombre}?"
            else:
                texto = f"¿{nombre} ≤ {umbral:g}{sufijo}?"
            if en_ruta:
                valor = float(fila[0, indice])
                mostrado = ("Sí" if valor else "No") if binaria else f"{valor:g}{sufijo}"
                va_izquierda = izquierda in visitados
                afirmativa = (not va_izquierda) if binaria else va_izquierda
                respuesta_caso = "Sí" if afirmativa else "No"
                paso = numero_por_nodo[nodo]
                pasos.append({"Paso": paso, "Pregunta del árbol": texto,
                              "Valor del caso": mostrado, "Respuesta": respuesta_caso})
                texto = f"Paso {paso} · {texto}\nValor del caso: {mostrado}"
            forma = "box"
        color = "#BCE6C9" if nodo == hoja else "#D6E9FF" if en_ruta else "#F3F4F6"
        linea = (f'n{nodo} [label={json.dumps(texto, ensure_ascii=False)}, '
                 f'shape={forma}, fillcolor="{color}"];')
        completo.append(linea)
        if nodo in visibles:
            if izquierda == derecha:
                rotulo = f"Nivel elegido\n{clase}" if nodo == hoja else str(clase)
                forma_caso = "ellipse"
            else:
                pregunta = texto
                if en_ruta:
                    pregunta = pasos[-1]["Pregunta del árbol"]
                rotulo = textwrap.fill(pregunta, width=32)
                if en_ruta:
                    rotulo = f"{paso}. {rotulo}\nDato: {mostrado}"
                else:
                    rotulo += "\nContinúa…"
                forma_caso = "box"
            borde = "#14633F" if nodo == hoja else "#18589D" if en_ruta else "#8996A3"
            grupo = "recorrido" if en_ruta else "alternativas"
            fuente = "Arial Bold" if nodo == hoja else "Arial"
            camino.append(f'n{nodo} [label={json.dumps(rotulo, ensure_ascii=False)}, '
                          f'shape={forma_caso}, group="{grupo}", fontname="{fuente}", '
                          f'fillcolor="{color}", color="{borde}", penwidth={3 if nodo == hoja else 1.5}];')
        if izquierda != derecha:
            # Un indicador vale 0 (ausente, izquierda) o 1 (presente, derecha).
            ramas = ("No", "Sí") if binaria else ("Sí", "No")
            for hijo, respuesta in zip((izquierda, derecha), ramas):
                activa = en_ruta and hijo in visitados
                estilo = ', color="#18589D", penwidth=3' if activa else ""
                arista = f'n{nodo} -> n{hijo} [label="{respuesta}"{estilo}];'
                completo.append(arista)
                if en_ruta:
                    atributos = ('color="#18589D", fontcolor="#18589D", penwidth=3, weight=30'
                                 if activa else 'color="#808D99", fontcolor="#455563", style=dashed')
                    camino.append(f'n{nodo} -> n{hijo} [label="{respuesta}", {atributos}];')
            if en_ruta:
                principal, alternativa = ((izquierda, derecha) if izquierda in visitados
                                          else (derecha, izquierda))
                camino.append(f'{{rank=same; n{principal}; n{alternativa}; '
                              f'n{principal} -> n{alternativa} [style=invis];}}')

    pasos.sort(key=lambda paso: paso["Paso"])
    return "\n".join(camino + ["}"]), "\n".join(completo + ["}"]), pasos


st.set_page_config(page_title="Clasificador de pacientes")
st.title("Clasificación de pacientes")

# Carga el archivo pacientes.csv desde la misma carpeta que app.py.
try:
    datos = pd.read_csv(
        Path(__file__).with_name("Pacientes.csv"),
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )
except (OSError, ValueError) as error:
    st.error(f"No se pudo leer Pacientes.csv: {error}")
    st.stop()


# Quita espacios accidentales de los encabezados.
datos.columns = datos.columns.astype(str).str.strip()

columna_antecedente = "antecedente_especifico"
opciones_antecedentes = []
if {"antecedentes_relevantes", columna_antecedente}.issubset(datos.columns):
    antecedentes_csv = datos.loc[datos["antecedentes_relevantes"].eq(
        1), columna_antecedente]
    antecedentes_csv = antecedentes_csv.dropna().astype(str).str.strip()
    opciones_antecedentes = sorted(
        antecedentes_csv[~antecedentes_csv.str.casefold().isin(
            ("", "ninguno"))].unique(),
        key=str.casefold,
    )


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
antecedente_a_columna = {
    nombre: f"antecedente_{i}" for i, nombre in enumerate(opciones_antecedentes)
}
for nombre, columna in antecedente_a_columna.items():
    datos[columna] = (datos["antecedentes_relevantes"].eq(1)
                      & datos[columna_antecedente].astype(str).str.strip().eq(nombre)).astype(int)

columnas_modelo = columnas_numericas + list(antecedente_a_columna.values())
columnas_entrada = columnas_modelo + [columna_texto]
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
for columna in columnas_numericas:
    datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
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
    particiones = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42)
    niveles = set(y)

    for entrenamiento, prueba in particiones.split(X, y, grupos):
        if set(y.iloc[entrenamiento]) == niveles and set(y.iloc[prueba]) == niveles:
            break
    else:
        raise ValueError(
            "Se necesitan más descripciones distintas por nivel para evaluar el modelo.")

    preprocesador = ColumnTransformer(
        [
            ("numericas", "passthrough", columnas_modelo),
            ("descripcion", TfidfVectorizer(
                strip_accents="unicode", ngram_range=(1, 2)
            ), columna_texto),
        ]
    )
    modelo = Pipeline([
        ("preprocesamiento", preprocesador),
        ("arbol", DecisionTreeClassifier(max_depth=12, random_state=42)),
    ])
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
        options=opciones_antecedentes,
        help="Cada antecedente seleccionado se incluye como indicador para el árbol.",
    )
    if not opciones_antecedentes:
        st.caption("El CSV no incluye antecedentes específicos disponibles.")

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
            **{col: int(nombre in antecedentes_especificos)
               for nombre, col in antecedente_a_columna.items()},
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


        recorrido, arbol_completo, pasos = graficos_del_caso(
            modelo, nuevo_registro, columnas_numericas, antecedente_a_columna
        )
        st.subheader("Árbol de decisión del caso")
        st.caption("Sigue la línea azul de arriba abajo hasta el nivel elegido. "
                   "Las ramas grises punteadas muestran las alternativas.")
        st.caption("En las preguntas sobre el peso de una palabra o frase, el valor es "
                   "su puntuación TF-IDF en la descripción escrita.")
        st.graphviz_chart(recorrido)
        cantidad = len(pasos)
        if pasos:
            condiciones = "1 condición" if cantidad == 1 else f"{cantidad} condiciones"
            st.write(f"El modelo evaluó {condiciones} y llegó a **{nivel}**.")
            with st.expander("Leer los pasos en una tabla"):
                st.dataframe(pd.DataFrame(pasos), hide_index=True)
        else:
            st.write(f"El árbol asigna **{nivel}** directamente en su raíz.")
        with st.expander("Ver el árbol completo"):
            st.caption("Las ramas resaltadas muestran el recorrido del caso ingresado.")
            st.graphviz_chart(arbol_completo)


st.write(f"Exactitud en el conjunto de prueba: {exactitud:.1%}")
