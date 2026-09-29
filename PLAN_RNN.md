# Plan de implementación del módulo RNN

## Objetivo

Desarrollar con Keras un modelo recurrente para pronosticar una serie temporal multivariada. El proceso debe conservar el orden cronológico, justificar la celda recurrente elegida y producir resultados reproducibles para incorporarlos al informe final.

## 1. Definir el problema y los datos

- Seleccionar un dataset temporal multivariado. Opción recomendada: Jena Climate.
- Definir una variable objetivo, por ejemplo la temperatura futura.
- Elegir el horizonte de pronóstico y el tamaño de la ventana histórica.
- Documentar frecuencia de muestreo, variables utilizadas, valores faltantes y rango temporal.
- Reservar los datos cronológicamente: 70 % entrenamiento, 15 % validación y 15 % prueba.

## 2. Scripts a desarrollar

### `rnn_data.py`

- Cargar y limpiar el dataset.
- Ordenar y validar las fechas.
- Realizar la partición cronológica antes de cualquier ajuste estadístico.
- Calcular media y desviación solamente con entrenamiento.
- Normalizar validación y prueba con esos mismos valores.
- Construir ventanas deslizantes y datasets de Keras.
- Guardar metadatos del procesamiento en `results/rnn/data_summary.json`.

### `rnn_pipeline.py`

- Implementar un baseline persistente para comparar el aprendizaje real del modelo.
- Construir una GRU como modelo principal; comparar con LSTM si el tiempo de cómputo lo permite.
- Usar una capa recurrente, regularización Dropout y una capa Dense de salida.
- Entrenar con Adam y MAE o MSE como función de pérdida.
- Aplicar `EarlyStopping`, `ModelCheckpoint` y clipping de gradiente.
- Exportar el mejor modelo a `models/rnn_forecaster.keras`.
- Guardar configuración, historial y tiempos de entrenamiento en `results/rnn/`.

### `rnn_experiments.py`

- Ejecutar al menos tres configuraciones controladas.
- Variar una decisión relevante por vez: GRU/LSTM, cantidad de unidades o longitud de ventana.
- Registrar parámetros, MAE y RMSE de validación para cada experimento.
- Seleccionar el modelo exclusivamente con la partición de validación.
- Evaluar una sola vez el modelo elegido sobre prueba.
- Guardar la comparación en `results/rnn/experiment_summary.csv`.

### `build_rnn_report_assets.py`

- Leer los archivos de resultados sin volver a entrenar.
- Generar curvas de pérdida de entrenamiento y validación.
- Generar un gráfico de valores reales frente a predicciones.
- Generar una tabla comparativa entre baseline y modelos recurrentes.
- Guardar imágenes listas para el informe en `results/rnn/figures/`.

## 3. Criterios de evaluación

- MAE como métrica principal por su interpretación directa.
- RMSE como métrica secundaria para detectar errores grandes.
- Comparación obligatoria con el baseline persistente.
- Curvas de entrenamiento y validación para identificar underfitting u overfitting.
- Inspección de predicciones consecutivas para analizar estabilidad temporal.
- Registro de semilla, hiperparámetros, cantidad de épocas y mejor época.

## 4. Información que se agregará al informe

La sección RNN se redactará después de ejecutar los scripts y contendrá solamente resultados obtenidos:

1. Dataset, variable objetivo, ventana e horizonte de pronóstico.
2. Partición cronológica y normalización sin fuga de información.
3. Arquitectura final y justificación de GRU o LSTM respecto de la memoria temporal y los gradientes en BPTT.
4. Hiperparámetros, callbacks y criterio de selección.
5. Tabla breve de experimentos con MAE y RMSE de validación.
6. Resultado final en prueba y comparación con el baseline.
7. Curvas de pérdida y gráfico de predicciones.
8. Análisis de estabilidad, underfitting u overfitting y limitaciones.

## 5. Orden de ejecución

1. Ejecutar `rnn_data.py` y revisar el resumen del dataset.
2. Ejecutar `rnn_experiments.py` para comparar configuraciones.
3. Entrenar y exportar el modelo seleccionado mediante `rnn_pipeline.py`.
4. Ejecutar `build_rnn_report_assets.py`.
5. Verificar que todas las métricas y figuras correspondan al mismo experimento final.
6. Actualizar el generador del informe con esos resultados.
7. Renderizar el documento y comprobar que no supere las cinco páginas exigidas.

## Entregables esperados

- Scripts Python escritos en inglés.
- Modelo recurrente exportado en formato `.keras`.
- Resultados estructurados en JSON o CSV.
- Gráficos reproducibles para el informe.
- Informe final con decisiones justificadas y resultados reales del módulo CNN y RNN.
