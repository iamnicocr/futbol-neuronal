import io
import json
import zipfile
from pathlib import Path
import numpy as np
from .caracteristicas import construir_variables, escalar
from .red import RedNeuronal

UMBRAL = 0.84
PERFILES = {'Más partidos (80 %)': 0.80, 'Más precisión (84 %)': 0.84}
NOMBRES_MODELOS = ('reales_v3_equilibrio', 'reales_v3_f1', 'goles_esperados_equilibrio', 'goles_esperados_f1')


def cargar_modelos(carpeta):
    carpeta = Path(carpeta)
    modelos = []
    for nombre in NOMBRES_MODELOS:
        ruta = carpeta / nombre / 'modelo.zip'
        with zipfile.ZipFile(ruta) as archivo:
            info = json.loads(archivo.read('experimento.json'))
            with np.load(io.BytesIO(archivo.read('pesos_modelo.npz'))) as datos:
                estado = {k: datos[k].copy() for k in datos.files if k.startswith(('W', 'b'))}
                media = datos['media'].copy()
                desviacion = datos['desviacion'].copy()
        red = RedNeuronal(5, info['arquitectura'][1:-1], info['activacion_oculta'])
        red.cargar_pesos(estado)
        modelos.append({'nombre': nombre, 'modo': info['configuracion']['variables_elegidas'],
                        'red': red, 'media': media, 'sd': desviacion})
    return modelos


def predecir_conjunto(partidos, indices, modelos):
    cache = {}
    probabilidades = []
    for modelo in modelos:
        modo = modelo['modo']
        if modo not in cache:
            cache[modo], _ = construir_variables(partidos, modo)
        x = escalar(cache[modo][indices], modelo['media'], modelo['sd'])
        probabilidades.append(modelo['red'].predecir_probabilidad(x))
    return np.mean(probabilidades, axis=0)


def resumen(y, prob, umbral=UMBRAL):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape != (len(y), 3):
        raise ValueError('Dimensiones incompatibles')
    elegidos = np.max(prob, axis=1) >= umbral
    pred = np.argmax(prob, axis=1)
    n = int(elegidos.sum())
    aciertos = int(np.sum(pred[elegidos] == y[elegidos]))
    matriz = np.bincount(3 * y[elegidos] + pred[elegidos], minlength=9).reshape(3, 3)
    return {'total': len(y), 'seleccionados': n, 'aciertos': aciertos,
            'no_seleccionados': len(y) - n, 'cobertura': n / len(y) if len(y) else 0.,
            'accuracy_selectiva': aciertos / n if n else None,
            'accuracy_general_ensemble': float(np.mean(pred == y)) if len(y) else None,
            'matriz': matriz.tolist(), 'umbral': float(umbral),
            'indices_seleccionados': np.flatnonzero(elegidos).tolist()}



def limite_inferior_wilson(aciertos, total, z=1.2815515655446004):
    """Límite inferior unilateral del 90 %, calculado con un intervalo de Wilson."""
    if total <= 0:
        return 0.0
    proporcion = aciertos / total
    zz = z*z
    denominador = 1.0 + zz / total
    centro = proporcion + zz / (2.0 * total)
    margen = z * np.sqrt(proporcion*(1.0-proporcion)/total + zz/(4.0*total*total))
    return float((centro-margen)/denominador)


def seleccionar_umbral(y_validacion, prob_validacion, objetivo=.90, minimo=80):
    """Obtiene el umbral de máxima cobertura solo con etiquetas de validación."""
    y = np.asarray(y_validacion, dtype=int)
    prob = np.asarray(prob_validacion, dtype=float)
    if prob.shape != (len(y), 3) or len(y) == 0:
        raise ValueError('Se necesitan probabilidades de tres clases en validación')
    if not 0.5 < objetivo < 1.0 or minimo < 1:
        raise ValueError('Objetivo o mínimo de partidos no válido')
    confianza = np.max(prob, axis=1)
    orden = np.argsort(-confianza, kind='stable')
    valores = confianza[orden]
    correctos = (np.argmax(prob, axis=1)[orden] == y[orden]).astype(int)
    acumulados = np.cumsum(correctos)
    candidato = None
    for i in range(minimo-1, len(y)):
        if i < len(y)-1 and valores[i] == valores[i+1]:
            continue
        n = i+1
        aciertos = int(acumulados[i])
        limite = limite_inferior_wilson(aciertos, n)
        if limite >= objetivo:
            candidato = {
                'umbral': float(valores[i]), 'objetivo': float(objetivo),
                'val_seleccionados': n, 'val_total': len(y),
                'val_aciertos': aciertos, 'val_accuracy': aciertos/n,
                'val_cobertura': n/len(y), 'val_limite_inferior': limite,
            }
    if candidato is None:
        return None
    return candidato


def curva_cobertura(y, prob, desde=.65, hasta=.96, pasos=64):
    """Pares de cobertura y accuracy observados exclusivamente en validación."""
    puntos=[]
    for umbral in np.linspace(desde,hasta,pasos):
        r = resumen(y,prob,umbral)
        if r['seleccionados']:
            puntos.append({'umbral':float(umbral), 'cobertura':r['cobertura'],
                           'accuracy':r['accuracy_selectiva'],
                           'partidos':r['seleccionados']})
    return puntos

def historial_para_pronosticos(partidos, modelos):
    historiales = {}
    for modelo in modelos:
        modo = modelo['modo']
        if modo not in historiales:
            _, historiales[modo] = construir_variables(partidos, modo)
    return historiales


def predecir_enfrentamiento(modelos, historiales, local, visitante, neutral, torneo, umbral=UMBRAL):
    probabilidades = []
    for modelo in modelos:
        hist = historiales[modelo['modo']]
        if modelo['modo'] == 'Elo y goles esperados':
            valores = hist.obtener(local, visitante, neutral, torneo, fecha=None)
        else:
            valores = hist.obtener(local, visitante, neutral, torneo, modelo['modo'])
        x = escalar(valores.reshape(1, 5), modelo['media'], modelo['sd'])
        probabilidades.append(modelo['red'].predecir_probabilidad(x)[0])
    prob = np.mean(probabilidades, axis=0)
    return prob, bool(np.max(prob) >= umbral)
