# Clasificacion-Pacientes

Prototipo educativo con casos simulados. Una red neuronal combina los datos
del formulario con la descripción libre para estimar uno de cinco niveles.

## Ejecución local

```powershell
python -m pip install pandas streamlit scikit-learn
python -m streamlit run app.py --global.developmentMode false --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

Abre http://127.0.0.1:8502 en el navegador.

La red se entrena localmente con la columna `ejemplo_clinico`, los datos
numéricos y los niveles etiquetados del CSV. La evaluación mantiene las
descripciones iguales en una sola partición para evitar compartirlas entre
el entrenamiento y la prueba.

La prueba del flujo completo se ejecuta con `python -m unittest -v`.
