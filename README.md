# Proyecto: Futbol Neuronal

Aplicación para estimar victoria local, empate o victoria visitante en encuentros de selecciones nacionales

## Ejecutar en Windows

1. Instala Python 3.11 o superior
2. Descarga `results.csv` desde https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
3. Copia el archivo en `data/results.csv`
4. Abre `ejecutar_windows.bat`
5. En la aplicación pulsa **Cargar datos**, configura la red y selecciona **Entrenar**

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

Desde el panel izquierdo puedes elegir las capas, neuronas, activación, épocas, tasa, lotes y métrica de selección

## Variables

Cada conjunto utiliza cinco entradas y el resultado `y` con tres etiquetas, local = 0, empate = 1, visitante = 2

| Forma y goles | Elo y forma |
|---|---|
| Diferencia de victorias en los últimos 5 | Diferencia Elo |
| Diferencia de goles a favor en los últimos 10 | Diferencia de victorias en los últimos 5 |
| Diferencia de goles en contra en los últimos 10 | Diferencia de goles netos en los últimos 10 |
| Sede neutral | Sede neutral |
| Partido no amistoso | Partido no amistoso |

**Elo y goles esperados** combina la diferencia Elo, goles esperados de cada selección, probabilidad previa de empate y tipo de partido

**Elo reforzado** conserva cinco entradas y utiliza Elo con margen de goles, forma reciente ponderada, diferencia de goles, sede neutral y tipo de partido

Los partidos se ordenan por fecha, los historiales se actualizan después de cada jornada y el escalado se ajusta solo con entrenamiento

## Filtro de confianza

En el panel lateral hay dos opciones, **Más partidos (80 %)** y **Más precisión (84 %)**

- El primer filtro permite seleccionar más encuentros y conserva una accuracy histórica superior al 90 % en el conjunto de prueba evaluado
- El segundo filtra más encuentros y alcanzó alrededor del 95 % en la prueba histórica
- Ambas cifras son **accuracy selectiva**, no accuracy sobre todos los partidos, que se informa por separado en Rendimiento
- El filtro se escogió según validación, no según el conjunto de prueba

## Red y evaluación

Red con NumPy, activaciones ReLU y sigmoide, salida Softmax, entropía cruzada y backpropagation, sin Scikit-Learn ni SQLite. La búsqueda compara SGD y Adam, varias ponderaciones para empates y ajustes de sesgo en validación

El conjunto de datos se divide cronológicamente en entrenamiento, validación y prueba. La configuración, época y ajustes de salida se seleccionan en validación. La prueba se evalúa al final

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
python scripts/entrenar_cli.py --data data/results.csv --modo auto --cantidad 18 --criterio equilibrio
```

Los resultados se almacenan en `resultados`, incluido `metricas.json` con matriz, accuracy, F1 y tiempos. Usa `--help` para ver las opciones


## Alta confianza

En **Rendimiento** está la opción **Evaluar alta confianza**, y en **Pronósticos** se puede consultar el mismo filtro para un enfrentamiento

El filtro reúne cuatro redes guardadas y selecciona encuentros cuando la mayor probabilidad promedio llega al 84 %, un valor fijado usando validación

En la prueba histórica de 4.933 partidos, seleccionó 315, acertó 300 y dejó 4.618 sin selección, lo que corresponde a **95,24 % de accuracy selectiva con 6,39 % de cobertura**

Esto **no es 95,24 % de accuracy general**, la red principal continúa alrededor del 60,23 % sobre todos los partidos. En la selección de alta confianza los modelos no eligieron empates

Para reproducir la evaluación sin abrir la aplicación

```powershell
python scripts/evaluar_confianza.py --data data/results.csv --resultados resultados
```

Se guardan `resultados/alta_confianza/metricas.json` y los CSV con cada predicción de validación y prueba

## Tiempos

Los tiempos aparecen junto a cada resultado y en **Tiempos de respuesta**, dentro del panel izquierdo

Se registran la carga del CSV, la preparación de variables, la búsqueda y cada entrenamiento, el modelo de referencia, la evaluación, los gráficos, la simulación de gradientes, el pronóstico y la preparación de las descargas

El panel permite exportar un CSV con la acción, la duración, la hora y el estado

Las mediciones corresponden al procesamiento en Python, no incluyen la conexión ni la presentación final del navegador

## Verificar modelos guardados

```powershell
python scripts/verificar_modelos.py --data data/results.csv
```

Los resultados de las tres selecciones nuevas están en `resultados/goles_esperados_*` y la comparación histórica en `resultados/comparacion_real.csv`
