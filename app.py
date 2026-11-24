import json
import textwrap

import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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
    codificador = preparador.named_transformers_["caso_clinico"]
    variables.extend((f"Caso clínico: {nombre}", "", True)
                     for nombre in codificador.categories_[0])

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

    recorrido, arbol_completo, pasos = graficos_del_caso(
        modelo, nuevo_registro, columnas_numericas, antecedente_a_columna
    )
    st.subheader("Árbol de decisión del caso")
    st.caption("Sigue la línea azul de arriba abajo hasta el nivel elegido. "
               "Las ramas grises punteadas muestran las alternativas.")
    st.graphviz_chart(recorrido)
    cantidad = len(pasos)
    if pasos:
        condiciones = "1 condición" if cantidad == 1 else f"{cantidad} condiciones"
        st.write(f"El modelo evaluó {condiciones} y llegó a **{prediccion}**.")
        with st.expander("Leer los pasos en una tabla"):
            st.dataframe(pd.DataFrame(pasos), hide_index=True)
    else:
        st.write(f"El árbol asigna **{prediccion}** directamente en su raíz.")
    with st.expander("Ver el árbol completo"):
        st.caption("Las ramas resaltadas muestran el recorrido del caso ingresado.")
        st.graphviz_chart(arbol_completo)


st.write(f"Precisión en el conjunto de prueba: {precision:.1%}")
