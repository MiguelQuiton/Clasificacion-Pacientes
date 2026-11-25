import json
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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
datos = pd.read_csv(
    Path(__file__).with_name("Pacientes.csv"),
    sep=None,
    engine="python",
    encoding="utf-8-sig",
)

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
indicadores = {
    antecedente: f"antecedente_{indice}"
    for indice, antecedente in enumerate(opciones_antecedentes)
}
columnas_arbol = columnas_numericas + list(indicadores.values())
columnas_entrada = columnas_arbol + [columna_texto]
columna_objetivo = "clasificacion"
columnas_necesarias = columnas_numericas + [columna_texto, columna_objetivo]


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
for antecedente, columna in indicadores.items():
    datos[columna] = (
        datos[columna_antecedente].astype("string").str.strip().eq(antecedente)
        & datos["antecedentes_relevantes"].eq(1)
    ).fillna(False).astype(int)
datos = datos.drop_duplicates(subset=columnas_entrada + [columna_objetivo])

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


def dividir_por_descripcion(X, y, grupos, n_particiones):
    particiones = StratifiedGroupKFold(n_splits=n_particiones, shuffle=True, random_state=42)
    niveles = set(y)
    for entrenamiento, prueba in particiones.split(X, y, grupos):
        if set(y.iloc[entrenamiento]) == niveles and set(y.iloc[prueba]) == niveles:
            return entrenamiento, prueba
    raise ValueError("Se necesitan más descripciones distintas por nivel para evaluar el modelo.")


@st.cache_resource
def entrenar_modelos(X, y, variables_arbol):
    grupos = (
        X[columna_texto].str.normalize("NFKD").str.encode("ascii", errors="ignore")
        .str.decode("ascii").str.casefold().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    )
    desarrollo, prueba = dividir_por_descripcion(X, y, grupos, 5)
    indices_ajuste, indices_validacion = dividir_por_descripcion(
        X.iloc[desarrollo], y.iloc[desarrollo], grupos.iloc[desarrollo], 4
    )
    ajuste = desarrollo[indices_ajuste]
    validacion = desarrollo[indices_validacion]

    arbol = Pipeline([
        ("preprocesamiento", ColumnTransformer([("datos", "passthrough", variables_arbol)])),
        ("arbol", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ])
    red = make_pipeline(
        ColumnTransformer([
            ("numericas", StandardScaler(), columnas_numericas),
            ("descripcion", TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2)), columna_texto),
        ]),
        MLPClassifier(hidden_layer_sizes=(24,), solver="lbfgs", max_iter=300, random_state=42),
    )
    arbol.fit(X.iloc[ajuste], y.iloc[ajuste])
    red.fit(X.iloc[ajuste], y.iloc[ajuste])
    clases = arbol.classes_
    prob_arbol = arbol.predict_proba(X.iloc[validacion])
    prob_red = red.predict_proba(X.iloc[validacion])
    objetivo = y.iloc[validacion]
    nivel_1 = objetivo == "Nivel 1"

    def puntuar(peso):
        predicciones = clases[((1 - peso) * prob_arbol + peso * prob_red).argmax(axis=1)]
        return (
            balanced_accuracy_score(objetivo, predicciones),
            (predicciones[nivel_1] == "Nivel 1").mean(),
            -peso,
        )

    peso_red = max((0.0, 0.25, 0.5, 0.75), key=puntuar)

    prob_arbol = arbol.predict_proba(X.iloc[prueba])
    prob_red = red.predict_proba(X.iloc[prueba])
    combinadas = (1 - peso_red) * prob_arbol + peso_red * prob_red
    objetivo = y.iloc[prueba]
    nivel_1 = objetivo == "Nivel 1"
    metricas = []
    for probabilidades in (prob_arbol, combinadas):
        predicciones = clases[probabilidades.argmax(axis=1)]
        metricas.append((
            accuracy_score(objetivo, predicciones),
            balanced_accuracy_score(objetivo, predicciones),
            (predicciones[nivel_1] == "Nivel 1").mean(),
        ))
    return arbol, red, peso_red, metricas


try:
    arbol, red, peso_red, metricas = entrenar_modelos(X, y, columnas_arbol)
except ValueError as error:
    st.error(f"No se pudieron entrenar los modelos: {error}")
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
        help="El árbol usa los antecedentes específicos seleccionados para estimar el nivel.",
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
            columna_texto: descripcion.strip(),
        }])
        for antecedente, columna in indicadores.items():
            nuevo_registro[columna] = int(antecedente in antecedentes_especificos)

        prob_arbol = arbol.predict_proba(nuevo_registro)[0]
        prob_red = red.predict_proba(nuevo_registro)[0]
        probabilidades = (1 - peso_red) * prob_arbol + peso_red * prob_red
        clases = arbol.classes_
        nivel = clases[probabilidades.argmax()]
        nivel_arbol = clases[prob_arbol.argmax()]
        nivel_red = red.classes_[prob_red.argmax()]
        st.subheader("Resultado del prototipo")
        st.write(f"**Clasificación estimada: {nivel}**")
        st.write(f"Árbol: {nivel_arbol} · Red neuronal: {nivel_red}")
        if antecedentes_especificos:
            st.write("Antecedentes seleccionados")
            for antecedente in antecedentes_especificos:
                st.write(f"- {antecedente}")

        with st.expander("Probabilidades combinadas"):
            for categoria, probabilidad in zip(clases, probabilidades):
                st.write(f"- {categoria}: {probabilidad:.1%}")

        recorrido, arbol_completo, pasos = graficos_del_caso(
            arbol, nuevo_registro, columnas_numericas, indicadores
        )
        st.subheader("Recorrido del árbol")
        st.caption(f"El recorrido explica la estimación del árbol: {nivel_arbol}.")
        st.graphviz_chart(recorrido)
        if pasos:
            st.dataframe(pasos, hide_index=True)
        with st.expander("Árbol completo"):
            st.graphviz_chart(arbol_completo)


with st.expander("Evaluación en el conjunto de prueba"):
    st.write(f"Peso elegido en validación: árbol {1 - peso_red:.0%}, red {peso_red:.0%}.")
    st.table(pd.DataFrame(
        [[f"{valor:.1%}" for valor in fila] for fila in metricas],
        index=["Árbol", "Combinación"],
        columns=["Exactitud", "Exactitud equilibrada", "Sensibilidad del Nivel 1"],
    ))
