import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp4.datos import leer_partidos
from mp4.caracteristicas import dividir_cronologicamente
from mp4.confianza import cargar_modelos, predecir_conjunto, resumen, UMBRAL, PERFILES, seleccionar_umbral


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/results.csv')
    parser.add_argument('--resultados', default='resultados')
    args = parser.parse_args()
    datos, _ = leer_partidos(args.data)
    indices, fechas = dividir_cronologicamente(datos)
    modelos = cargar_modelos(args.resultados)
    resultado = {'umbral': UMBRAL, 'perfiles': PERFILES, 'modelos': [m['nombre'] for m in modelos],
                 'fechas': fechas, 'advertencia': 'Accuracy selectiva solo para pronósticos que superan el umbral'}
    salida = Path(args.resultados) / 'alta_confianza'
    salida.mkdir(parents=True, exist_ok=True)
    probabilidades = {}
    etiquetas = {}
    for grupo in ('val', 'test'):
        ix = indices[grupo]
        probabilidades[grupo] = predecir_conjunto(datos, ix, modelos)
        etiquetas[grupo] = datos.y.to_numpy(int)[ix]
    ajustes = {str(int(objetivo*100)): seleccionar_umbral(etiquetas['val'], probabilidades['val'], objetivo)
               for objetivo in (.90,.95)}
    resultado['filtros_automaticos'] = ajustes
    resultado['prueba_automatica'] = {}
    for nombre, ajuste in ajustes.items():
        if ajuste is not None:
            evaluacion = resumen(etiquetas['test'], probabilidades['test'], ajuste['umbral'])
            resultado['prueba_automatica'][nombre] = {k:v for k,v in evaluacion.items() if k != 'indices_seleccionados'}
    for grupo in ('val', 'test'):
        ix = indices[grupo]
        p = probabilidades[grupo]
        y = etiquetas[grupo]
        r = resumen(y, p)
        resultado[grupo + '_perfiles'] = {k: {kk: vv for kk,vv in resumen(y,p,umbral).items() if kk != 'indices_seleccionados'} for k,umbral in PERFILES.items()}
        resultado[grupo] = {k:v for k,v in r.items() if k != 'indices_seleccionados'}
        df = datos.iloc[ix][['date','home_team','away_team','home_score','away_score','y']].copy()
        df['prob_local'] = p[:,0]
        df['prob_empate'] = p[:,1]
        df['prob_visitante'] = p[:,2]
        df['prediccion'] = np.argmax(p, axis=1)
        df['confianza'] = np.max(p,axis=1)
        df['alta_confianza'] = df['confianza'] >= UMBRAL
        for nombre, ajuste in ajustes.items():
            if ajuste is not None:
                df[f'selectivo_{nombre}'] = df['confianza'] >= ajuste['umbral']
        df['acierto'] = df['prediccion'] == df['y']
        df.to_csv(salida / f'predicciones_{grupo}.csv',index=False)
        print(f"{grupo}: {r['aciertos']} / {r['seleccionados']} = {r['accuracy_selectiva']:.2%} | cobertura {r['cobertura']:.2%}")
    for nombre, evaluacion in resultado['prueba_automatica'].items():
        print(f"Automático {nombre}%: {evaluacion['aciertos']} / {evaluacion['seleccionados']}"
              f" = {evaluacion['accuracy_selectiva']:.2%}, cobertura {evaluacion['cobertura']:.2%}")
    (salida/'metricas.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2),encoding='utf8')


if __name__ == '__main__':
    main()
