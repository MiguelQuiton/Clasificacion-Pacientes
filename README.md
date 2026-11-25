# Clasificacion-Pacientes

Prototipo educativo con casos simulados. Una red neuronal combina los datos
del formulario con la descripción libre para estimar uno de cinco niveles.

## Ejecución local

```powershell
python -m pip install pandas streamlit scikit-learn
python -m streamlit run app.py --global.developmentMode false --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

Abre http://127.0.0.1:8502 en el navegador.

## Arquitectura

El modelo se entrena en local con [Pacientes.csv](Pacientes.csv). La columna
`clasificacion` es la etiqueta que aprende a predecir. El pipeline de
[app.py](app.py) aplica los mismos pasos al entrenar y al clasificar un caso.

1. **Datos numéricos.** `StandardScaler` escala edad, temperatura, frecuencia
   cardíaca, saturación de oxígeno, cantidad de síntomas y antecedentes
   relevantes con estadísticas del conjunto de entrenamiento.
2. **Descripción.** `TfidfVectorizer` extrae palabras y pares de palabras
   consecutivas de `ejemplo_clinico`, normaliza los acentos y pondera los
   términos según su frecuencia. Al clasificar, el texto libre pasa por
   ese mismo vectorizador.
3. **Red y salida.** Un `MLPClassifier` con una capa oculta de 24 neuronas
   combina las dos entradas. Usa el optimizador L-BFGS, un máximo de 300
   iteraciones y la semilla 42. La app muestra el nivel con mayor probabilidad
   estimada de los cinco.

## Evaluación con el CSV del proyecto

Tras limpiar los datos y quitar duplicados quedaron 530 registros. Se usó
la primera partición de `StratifiedGroupKFold` con los cinco niveles
presentes en ambos conjuntos. El resultado fue **404 registros de
entrenamiento y 126 de prueba**, con cero descripciones normalizadas
compartidas entre ambos. Las métricas corresponden a esa partición.

**Exactitud global** — 57,1 % (72 de 126). **F1 macro** — 62,9 %.

| Nivel | Precisión | Sensibilidad (recall) | F1 | Casos de prueba |
| --- | ---: | ---: | ---: | ---: |
| 1 | 87,5 % | 13,7 % | 23,7 % | 51 |
| 2 | 57,4 % | 77,5 % | 66,0 % | 40 |
| 3 | 47,6 % | 100,0 % | 64,5 % | 20 |
| 4 | 52,9 % | 100,0 % | 69,2 % | 9 |
| 5 | 100,0 % | 83,3 % | 90,9 % | 6 |

La precisión indica cuántas predicciones de un nivel fueron correctas.
La sensibilidad indica cuántos casos de ese nivel detectó la red.
El F1 combina ambas métricas y su promedio macro da el mismo peso a cada nivel.

