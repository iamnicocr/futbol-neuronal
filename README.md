# Proyecto: Futbol Neuronal

Aplicación para estimar victoria local, empate o victoria visitante en encuentros de selecciones nacionales

## Ejecutar en Windows

1. Instala Python 3.11 o superior
2. Descarga `results.csv` desde https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
3. Copia el archivo en `data/results.csv`
4. Abre `ejecutar_windows.bat`
5. En la aplicación pulsa **Cargar datos**, configura la red y selecciona **Entrenar modelo**

También puedes iniciar desde una terminal abierta en esta carpeta

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Cómo usar

- **Datos**, partidos, variables y distribución de resultados
- **Rendimiento**, accuracy, F1, matriz de confusión, comparación y descargas
- **Red neuronal**, arquitectura, matrices, activaciones y gradientes
- **Gráficas**, pérdida y accuracy por época, frontera de decisión
- **Pronósticos**, elige las dos selecciones y consulta las tres probabilidades

La configuración se cambia desde la barra izquierda, puedes elegir capas, neuronas, activación, épocas, tasa de aprendizaje, tamaño de lote, criterio y características

## Variables

Cada conjunto utiliza cinco entradas y el resultado `y` con tres etiquetas, local = 0, empate = 1, visitante = 2

| Forma y goles | Elo y forma |
|---|---|
| Diferencia de victorias en los últimos 5 | Diferencia Elo |
| Diferencia de goles a favor en los últimos 10 | Diferencia de victorias en los últimos 5 |
| Diferencia de goles en contra en los últimos 10 | Diferencia de goles netos en los últimos 10 |
| Sede neutral | Sede neutral |
| Partido no amistoso | Partido no amistoso |

Los partidos se ordenan por fecha, los historiales se actualizan después de cada jornada y el escalado se ajusta solo con entrenamiento

## Red y evaluación

Implementación con NumPy, activaciones ReLU y sigmoide, salida Softmax, entropía cruzada y backpropagation, sin Scikit-Learn ni SQLite

El conjunto de datos se divide cronológicamente en entrenamiento, validación y prueba, la selección de configuración y época se realiza usando validación

## Archivos principales

```text
app.py                    Interfaz
mp4/datos.py              Carga de partidos
mp4/caracteristicas.py    Variables históricas
mp4/red.py                Red y entrenamiento
mp4/experimentos.py       Comparación de configuraciones
mp4/evaluacion.py         Métricas y exportación
mp4/figuras.py            Gráficas
scripts/entrenar_cli.py   Entrenamiento por consola
notebooks/                Comprobaciones matemáticas
tests/                    Pruebas automatizadas
DECLARACION_IA.md         Registro del uso de IA
```

## Pruebas

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Para ejecutar por consola

```powershell
python scripts/entrenar_cli.py --data data/results.csv --modo elo --buscar
```

Los resultados se almacenan en `resultados`, puedes usar `--help` para ver las opciones
