# 🎓 Alerta temprana de deserción estudiantil

**ACA 1 · Aprendizaje Automático** — Especialización en Inteligencia Artificial, Corporación Unificada Nacional de Educación Superior (CUN)
Docente: Dr. Hamilton Rivera Flor

**Integrantes:** Bonny Darhyll Brayan Galindo Santos · Andrés Alberto López Herrera · Camilo José Rodríguez Ayubi

Solución de Machine Learning clásico (pilar **Educación**) que clasifica a cada estudiante en *Deserción*, *Matriculado* o *Graduado* a partir de sus datos de ingreso, su situación económica y su rendimiento del primer año, con despliegue en Streamlit.

## Resultados (conjunto de prueba, 885 estudiantes)

| Modelo | Exactitud | F1 macro | ROC-AUC | Detección de deserción |
|---|---|---|---|---|
| Regresión logística (C = 0,1, pesos balanceados) | 74,0 % | 0,691 | 0,874 | 70,1 % (precisión 82,2 %) |

## Estructura

```
├── app.py                                   # App de Streamlit
├── preprocesamiento.py                      # Limpieza, diccionarios e ingeniería de características (compartido)
├── ACA1_Aprendizaje_Automatico_Desercion.ipynb  # Notebook con EDA, limpieza, modelado y evaluación
├── requirements.txt
├── modelo/
│   ├── modelo_anual.joblib                  # Modelo principal (datos del 1.er año)
│   ├── modelo_semestre1.joblib              # Alerta tras el 1.er semestre
│   ├── modelo_ingreso.joblib                # Alerta al ingreso
│   └── metadatos.json                       # Métricas que muestra la app
├── data/
│   ├── data.csv                             # Dataset original de la UCI (CC BY 4.0)
│   └── estudiantes_ejemplo.csv              # 25 estudiantes de prueba para la predicción por lotes
└── figuras/                                 # Gráficas generadas por el notebook
```

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para reproducir el modelo, abrir el notebook (funciona también en Google Colab) y ejecutar todas las celdas. Los modelos se guardaron con scikit-learn 1.9.1; si se reentrena con otra versión, actualizar `requirements.txt`.

## Datos

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict students' dropout and academic success* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89 — Licencia CC BY 4.0.
