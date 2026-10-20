# CNN y RNN con Keras

Este proyecto implementa dos módulos con TensorFlow/Keras: una CNN para clasificar
Fashion-MNIST y una RNN para pronosticar temperatura con Jena Climate. Los datasets
se descargan automáticamente y no están incluidos en la entrega.

## Preparacion

TensorFlow requiere una version compatible de Python. Se recomienda Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En Linux o macOS se pueden reemplazar los comandos anteriores por `python3` y
`source .venv/bin/activate`.

## Verificación rápida

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m py_compile cnn_pipeline.py cnn_experiments.py rnn_data.py rnn_pipeline.py rnn_experiments.py build_rnn_report_assets.py
```

## Descarga del dataset

```powershell
.\.venv\Scripts\python.exe cnn_pipeline.py --download-only
```

## Prueba corta

```powershell
.\.venv\Scripts\python.exe cnn_pipeline.py --epochs 1 --limit-train 2000
```

## Entrenamiento final

```powershell
.\.venv\Scripts\python.exe cnn_experiments.py --epochs 10
```

El comando compara tres arquitecturas con la misma division por lotes, elige la
menor perdida de validacion y evalua ese modelo una unica vez sobre test. Los
resultados se guardan en `results/cnn`, el modelo en
`models/cnn_fashion_mnist.keras` y sus pesos en
`models/cnn_fashion_mnist.weights.h5`.

## Modulo RNN

El modulo recurrente pronostica temperatura a partir de una serie temporal
multivariada. La particion es cronologica y la normalizacion se ajusta solamente
con los datos de entrenamiento.

Prueba rapida sin descarga:

```powershell
.\.venv\Scripts\python.exe rnn_experiments.py --synthetic --limit-rows 1200 --epochs 1
.\.venv\Scripts\python.exe build_rnn_report_assets.py --allow-synthetic
```

Entrenamiento con Jena Climate:

```powershell
.\.venv\Scripts\python.exe rnn_experiments.py --download --epochs 15
.\.venv\Scripts\python.exe build_rnn_report_assets.py
```

Los resultados se guardan en `results/rnn` y el modelo seleccionado en
`models/rnn_forecaster.keras`. El conjunto de prueba se evalua una unica vez,
despues de seleccionar el modelo por MAE de validacion.

## Artefactos incluidos

- `informe_emanuel_rebecchi.pdf`: informe final.
- `models/cnn_fashion_mnist.keras`: modelo CNN seleccionado.
- `models/cnn_fashion_mnist.weights.h5`: pesos de la CNN.
- `models/rnn_forecaster.keras`: modelo RNN seleccionado.
- `results/cnn` y `results/rnn`: métricas, historiales y gráficos obtenidos.

Los archivos `.venv`, `.cache`, `tmp` y los datasets descargados no forman parte
de la entrega porque se regeneran mediante los comandos anteriores.
