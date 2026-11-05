"""Proyecto Futbol Neuronal"""
from pathlib import Path
from time import perf_counter
from html import escape
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mp4.datos import leer_partidos, descargar_csv, DATASET_URL
from mp4.caracteristicas import (nombres, construir_variables, dividir_cronologicamente,
                                 ajustar_escalador, escalar)
from mp4.red import RedNeuronal, CLASES, entropia_multiclase, entrenar
from mp4.evaluacion import metricas_multiclase, exportar_experimento
from mp4.experimentos import candidatos, buscar
from mp4.figuras import (curvas, matriz_confusion, arquitectura, frontera_2d,
                         barras_probabilidad, estilo_grafica)
from mp4.tiempos import RegistroTiempos, mostrar_tiempo
from mp4.confianza import (cargar_modelos, predecir_conjunto, resumen,
                           historial_para_pronosticos, predecir_enfrentamiento, UMBRAL, PERFILES,
                           seleccionar_umbral, curva_cobertura)

st.set_page_config(page_title='Proyecto: Futbol Neuronal', page_icon='⚽', layout='wide',
                   initial_sidebar_state='expanded')
css = Path(__file__).parent/'assets'/'estilo.css'
st.markdown(f'<style>{css.read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)
if 'reloj' not in st.session_state:
    st.session_state['reloj'] = RegistroTiempos()
reloj = st.session_state['reloj']


def chip(nombre, prefijo=None):
    segundos = reloj.ultimo(nombre)
    if segundos is not None:
        etiqueta = escape(prefijo or nombre)
        st.markdown(f'<span class="timer-chip">{etiqueta}  {mostrar_tiempo(segundos)}</span>',
                    unsafe_allow_html=True)


def titulo(numero, texto):
    st.markdown(f'<div class="section-heading"><b>{numero}</b><span>{escape(texto)}</span></div>',
                unsafe_allow_html=True)


def grafica(nombre, figura):
    with reloj.medir(nombre):
        fig = figura() if callable(figura) else figura
        st.plotly_chart(fig, use_container_width=True)
    chip(nombre, 'Gráfico')


st.markdown('''
<div class="masthead">
  <div class="overline">ANÁLISIS DE SELECCIONES &nbsp; / &nbsp; ESTUDIO 01</div>
  <h1>Futbol<br>Neuronal</h1>
  <div class="subline">PROYECTO &nbsp; / &nbsp; 03 RESULTADOS &nbsp; / &nbsp; 05 VARIABLES</div>
  <div class="tab-id">FN</div>
</div>
''', unsafe_allow_html=True)

st.sidebar.markdown('<div class="control-cap">FN / PANEL DE CONTROL</div>', unsafe_allow_html=True)
st.sidebar.header('Archivo')
archivo = st.sidebar.file_uploader('Seleccionar results.csv', type=['csv'])
ruta = Path(__file__).resolve().parent/'data'/'results.csv'
with st.sidebar.expander('Obtener archivo'):
    st.markdown(f'[Abrir dataset]({DATASET_URL})')
    if st.button('Descargar CSV', key='descargar_csv', use_container_width=True):
        try:
            with reloj.medir('Descarga CSV'):
                descargar_csv(ruta)
            st.success('Archivo listo, selecciona Cargar datos')
            chip('Descarga CSV')
        except Exception as exc:
            st.error(f'No se pudo descargar: {exc}')
if ruta.is_file():
    st.sidebar.caption('Archivo local disponible')
if st.sidebar.button('Cargar datos', key='cargar_datos', type='primary', use_container_width=True):
    try:
        origen = io.BytesIO(archivo.getvalue()) if archivo is not None else ruta
        with st.spinner('Leyendo partidos'):
            with reloj.medir('Carga y limpieza'):
                datos, info = leer_partidos(origen)
        st.session_state['datos'] = (datos, info)
        st.session_state.pop('modelo', None)
        st.session_state.pop('prueba', None)
        st.session_state.pop('alta_confianza_prueba', None)
        st.session_state.pop('alta_confianza_probabilidades', None)
        st.session_state.pop('alta_confianza_etiquetas', None)
        st.session_state.pop('alta_confianza_probabilidades_val', None)
        st.session_state.pop('alta_confianza_etiquetas_val', None)
        st.session_state.pop('historiales_alta_confianza', None)
        st.session_state.pop('ultimo_pronostico_selectivo', None)
        st.sidebar.success(f'{len(datos):,} partidos listos')
        chip('Carga y limpieza')
    except Exception as exc:
        st.sidebar.error(f'No se pudo cargar el archivo: {exc}')

st.sidebar.divider()
st.sidebar.header('Modelo')
disponible = 'datos' in st.session_state
if disponible:
    datos, info = st.session_state['datos']
    minimo, maximo = int(datos.date.dt.year.min()), int(datos.date.dt.year.max())
    anio = st.sidebar.slider('Año inicial', minimo, maximo, min(1990, maximo))
else:
    anio = 1990
modo = st.sidebar.selectbox('Variables', ['Forma y goles', 'Elo y forma', 'Elo reforzado', 'Elo y goles esperados'], index=2)
capas = st.sidebar.slider('Capas ocultas', 1, 2, 1)
neuronas1 = st.sidebar.slider('Neuronas en capa 1', 4, 8, 8)
neuronas2 = st.sidebar.slider('Neuronas en capa 2', 4, 8, 6, disabled=capas != 2)
config_ocultas = (neuronas1, neuronas2) if capas == 2 else (neuronas1,)
activacion = st.sidebar.selectbox('Activación oculta', ['ReLU', 'Sigmoide'])
epocas = st.sidebar.slider('Épocas máximas', 20, 500, 200, step=10)
tasa = st.sidebar.select_slider('Tasa de aprendizaje', options=[.001,.003,.005,.01,.02,.03,.05,.08,.1,.2], value=.03)
lote = st.sidebar.select_slider('Tamaño del lote', options=[128,256,512,1024,2048], value=512)
paciencia = st.sidebar.slider('Paciencia', 0, 70, 30, step=5, help='0 desactiva la parada anticipada')
criterio = st.sidebar.selectbox('Elegir por', ['accuracy','equilibrio', 'f1_macro', 'loss'],
    format_func=lambda k: {'equilibrio':'Accuracy y F1', 'accuracy':'Accuracy', 'f1_macro':'F1 macro', 'loss':'Pérdida'}[k])
semilla = st.sidebar.number_input('Semilla', min_value=0, max_value=999999, value=42)
st.sidebar.divider()
probar_varias = st.sidebar.checkbox('Probar otras arquitecturas', value=True, help='Demora más')
max_candidatos = st.sidebar.slider('Configuraciones', 2, 18, 18, disabled=not probar_varias)
ambos = st.sidebar.checkbox('Comparar las cuatro selecciones de variables', value=True, disabled=not probar_varias)
st.sidebar.divider()
OPCIONES_CONFIANZA = ['Automático 90 %', 'Automático 95 %', *PERFILES]
perfil_confianza = st.sidebar.selectbox('Filtro de confianza', OPCIONES_CONFIANZA, index=0)
objetivo_auto = .95 if perfil_confianza == 'Automático 95 %' else .90
usa_auto = perfil_confianza.startswith('Automático')
umbral_confianza = PERFILES.get(perfil_confianza, UMBRAL)


def umbral_validado(datos, estado):
    if not usa_auto:
        return umbral_confianza, None
    if 'alta_confianza_probabilidades_val' not in st.session_state:
        with reloj.medir('Calibración de confianza'):
            modelos = cargar_modelos(Path(__file__).parent / 'resultados')
            ix_val = estado['indices']['val']
            st.session_state['alta_confianza_probabilidades_val'] = predecir_conjunto(datos, ix_val, modelos)
            st.session_state['alta_confianza_etiquetas_val'] = estado['y'][ix_val]
    ajuste = seleccionar_umbral(st.session_state['alta_confianza_etiquetas_val'],
                               st.session_state['alta_confianza_probabilidades_val'], objetivo_auto)
    if ajuste is None:
        raise ValueError('No se encontró un umbral que alcance el objetivo en validación')
    return ajuste['umbral'], ajuste


if disponible and st.sidebar.button('Entrenar', type='primary', use_container_width=True):
    inicio_total = perf_counter()
    try:
        with st.spinner('Preparando partidos'):
            with reloj.medir('División cronológica'):
                indices, fechas = dividir_cronologicamente(datos, int(anio))
                y = datos['y'].to_numpy(dtype=int)
            modos = ['Forma y goles', 'Elo y forma', 'Elo reforzado', 'Elo y goles esperados'] if (ambos and probar_varias) else [modo]
            conjuntos = {}
            for nombre in modos:
                with reloj.medir(f'Variables · {nombre}'):
                    raw, hist = construir_variables(datos, nombre)
                    mu, sd = ajustar_escalador(raw[indices['train']])
                    conjuntos[nombre] = {'raw':raw, 'x':escalar(raw,mu,sd),
                                         'hist':hist, 'media':mu, 'sd':sd}
        opciones = candidatos(capas, config_ocultas, activacion, tasa, semilla, probar_varias)
        opciones = opciones[:max_candidatos] if probar_varias else opciones
        progreso = st.sidebar.progress(0, text='Entrenando')
        final, tabla, mejor_clave = None, [], (-float('inf'), -float('inf'))
        for j, (estrategia, grupo) in enumerate(conjuntos.items()):
            x = grupo['x']
            with reloj.medir(f'Búsqueda · {estrategia}'):
                mejor, candidatos_tab = buscar(x[indices['train']],y[indices['train']],
                    x[indices['val']], y[indices['val']], opciones, epocas, lote, paciencia, criterio)
            for fila in candidatos_tab:
                fila['caracteristicas'] = estrategia
                tabla.append(fila)
                reloj.guardar(f"Red {fila['arquitectura']} · {estrategia} · #{fila['prueba']}",
                             fila['segundos'])
            red, fit, val, info_mejor = mejor
            score = (.7*val['accuracy']+.3*val['f1_macro']) if criterio=='equilibrio' else (val['accuracy'] if criterio=='accuracy' else val['f1_macro'] if criterio=='f1_macro' else -val['cross_entropy'])
            clave = (round(score,8), round(val['f1_macro'],8), -val['cross_entropy'])
            if clave > mejor_clave:
                final = {'red':red, 'fit':fit, 'val':val, 'estrategia':estrategia,
                         'meta':info_mejor, 'grupo':grupo}
                mejor_clave = clave
            progreso.progress((j+1)/len(conjuntos), text=f'Entrenamiento {j+1}/{len(conjuntos)}')
        grupo = final['grupo']
        x = grupo['x']
        with reloj.medir('Modelo de referencia'):
            base = RedNeuronal(5, (), activacion, semilla)
            fit_base = entrenar(base, x[indices['train']], y[indices['train']],
                               x[indices['val']], y[indices['val']], epocas, tasa,
                               lote, paciencia, criterio, semilla)
            val_base = metricas_multiclase(y[indices['val']], base.predecir_probabilidad(x[indices['val']]))
        configuracion = {'anio':anio, 'modo_inicial':modo, 'capas_iniciales':capas,
            'neuronas_iniciales':config_ocultas, 'activacion_inicial':activacion,
            'epocas_max':epocas, 'tasa_inicial':tasa, 'lote':lote, 'paciencia':paciencia,
            'criterio':criterio, 'semilla':int(semilla), 'busqueda':probar_varias,
            'ambas_estrategias':ambos}
        st.session_state['modelo'] = {'mejor':final,
            'baseline':{'red':base, 'fit':fit_base, 'val':val_base},
            'tabla':tabla, 'y':y, 'indices':indices, 'fechas':fechas,
            'config':configuracion, 'conjuntos':conjuntos}
        st.session_state.pop('prueba', None)
        st.session_state.pop('alta_confianza_prueba', None)
        st.session_state.pop('alta_confianza_probabilidades', None)
        st.session_state.pop('alta_confianza_etiquetas', None)
        st.session_state.pop('alta_confianza_probabilidades_val', None)
        st.session_state.pop('alta_confianza_etiquetas_val', None)
        reloj.guardar('Entrenamiento completo', perf_counter()-inicio_total)
        progreso.empty()
        st.sidebar.success('Modelo entrenado')
        chip('Entrenamiento completo')
    except Exception as exc:
        reloj.guardar('Entrenamiento completo', perf_counter()-inicio_total, correcto=False)
        st.sidebar.error(f'Error de entrenamiento: {exc}')

with st.sidebar.expander('Tiempos de respuesta', expanded=False):
    if reloj.eventos:
        for evento in reversed(reloj.eventos[-12:]):
            nombre = escape(evento['accion'])
            valor = mostrar_tiempo(evento['segundos'])
            st.markdown(f'<div class="sidebar-time"><span>{nombre}</span><strong>{valor}</strong></div>',
                        unsafe_allow_html=True)
        registros = pd.DataFrame(reloj.eventos)
        st.download_button('Exportar tiempos CSV', registros.to_csv(index=False).encode('utf-8'),
                           'tiempos_respuesta.csv', 'text/csv', key='tiempos_csv')
    else:
        st.caption('Los tiempos aparecerán al ejecutar acciones')

if not disponible:
    st.info('Selecciona results.csv y pulsa Cargar datos')
    st.stop()

datos, info = st.session_state['datos']
estado = st.session_state.get('modelo')
ultimo = mostrar_tiempo(reloj.ultimo('Entrenamiento completo'))
st.markdown(f'''<div class="status-strip">
<div><i class="dot"></i><span class="label">{len(datos):,} PARTIDOS</span></div>
<div><i class="dot"></i><span class="label">{info['equipos']} SELECCIONES</span></div>
<div><i class="dot"></i><span class="label">{'MODELO LISTO' if estado else 'SIN ENTRENAR'}</span></div>
<div><i class="dot"></i><span class="label">ÚLTIMO ENTRENAMIENTO  {ultimo}</span></div>
</div>''', unsafe_allow_html=True)

paginas = st.tabs(['01  Datos', '02  Rendimiento', '03  Red neuronal',
                   '04  Gráficas', '05  Pronósticos'])

with paginas[0]:
    titulo('01 / DATOS', 'Partidos registrados')
    a,b,c,d = st.columns(4)
    a.metric('Partidos', f'{len(datos):,}')
    b.metric('Selecciones', info['equipos'])
    c.metric('Desde', info['primer_partido'])
    d.metric('Hasta', info['ultimo_partido'])
    st.caption('Sin registros descartados' if info['descartados']==0 else f"Descartados {info['descartados']}")
    with reloj.medir('Distribución de resultados'):
        conteos = datos['y'].value_counts().reindex([0,1,2], fill_value=0)
        fig = go.Figure(go.Bar(x=list(CLASES),y=conteos.values,
                         marker_color=['#E7603D','#D0A044','#178579']))
        fig.update_layout(title='Resultados por clase',height=330,yaxis_title='Partidos')
        estilo_grafica(fig)
    grafica('Render distribución', fig)
    chip('Distribución de resultados', 'Datos')
    if estado:
        m = estado['mejor']; grupo = m['grupo']; cols = nombres(m['estrategia'])
        with reloj.medir('Tabla de variables'):
            tabla = pd.DataFrame(grupo['raw'], columns=cols)
            tabla['y'] = estado['y']
            vista = tabla.tail(35).round(4)
        titulo('01A / VARIABLES', 'Entradas del modelo')
        st.caption('5 entradas, columna y para el resultado real')
        st.dataframe(vista, hide_index=True, use_container_width=True)
        chip('Tabla de variables', 'Tabla')
        ss = estado['indices']; fechas = estado['fechas']
        a,b,c = st.columns(3)
        a.metric('Entrenamiento',f"{len(ss['train']):,}")
        b.metric('Validación',f"{len(ss['val']):,}")
        c.metric('Prueba',f"{len(ss['test']):,}")
        st.caption(f"Validación desde {fechas['validacion_desde']}  /  Prueba desde {fechas['prueba_desde']}")
    else:
        st.dataframe(datos[['date','home_team','away_team','home_score','away_score','neutral','y']].tail(35),
                     hide_index=True,use_container_width=True)
        st.dataframe(pd.DataFrame({'Variable':nombres(modo),
                          'Tipo':['Continua','Continua','Continua',
                                  'Continua' if modo=='Elo y goles esperados' else 'Binaria','Binaria']}),
                          hide_index=True,use_container_width=True)

with paginas[1]:
    titulo('02 / RESULTADOS', 'Rendimiento')
    if not estado:
        st.info('Configura los parámetros y pulsa Entrenar')
    else:
        mejor = estado['mejor']; val = mejor['val']; base_val = estado['baseline']['val']
        a,b,c,d = st.columns(4)
        a.metric('Accuracy',f"{val['accuracy']:.2%}",delta=f"{(val['accuracy']-base_val['accuracy'])*100:+.2f} pp")
        b.metric('F1 macro',f"{val['f1_macro']:.3f}")
        c.metric('Pérdida',f"{val['cross_entropy']:.4f}")
        d.metric('Parámetros',mejor['red'].parametros)
        st.caption(f"Validación / {' → '.join(map(str,mejor['red'].dimensiones))} / {mejor['red'].tipo} / η {mejor['meta']['tasa']}")
        grafica('Matriz de validación',lambda: matriz_confusion(val['matriz'],'Validación'))
        a,b = st.columns(2)
        a.metric('Red elegida',mostrar_tiempo(mejor['fit']['segundos']))
        b.metric('Entrenamiento total',mostrar_tiempo(reloj.ultimo('Entrenamiento completo')))
        titulo('02A / COMPARATIVA','Modelos evaluados')
        with reloj.medir('Tabla de modelos'):
            tabla = pd.DataFrame(estado['tabla']).sort_values(['accuracy_val','f1_macro_val'],ascending=False)
            tabla_vista = tabla.style.format({'accuracy_val':'{:.3%}','f1_macro_val':'{:.4f}',
                        'loss_val':'{:.4f}','segundos':'{:.2f}'})
        st.dataframe(tabla_vista, hide_index=True, use_container_width=True)
        chip('Tabla de modelos', 'Comparación')
        st.caption(f"Modelo lineal / accuracy {base_val['accuracy']:.2%} / F1 {base_val['f1_macro']:.4f} / {estado['baseline']['red'].parametros} parámetros")
        titulo('02B / TEST','Evaluación final')
        if 'prueba' not in st.session_state:
            if st.button('Evaluar en prueba',type='primary',key='evaluar_test'):
                with reloj.medir('Evaluación final'):
                    x = mejor['grupo']['x']; ix = estado['indices']['test']; yt = estado['y'][ix]
                    pred = mejor['red'].predecir_probabilidad(x[ix])
                    pb = estado['baseline']['red'].predecir_probabilidad(x[ix])
                    st.session_state['prueba'] = {
                        'mlp':metricas_multiclase(yt,pred),
                        'base':metricas_multiclase(yt,pb),
                        'mayoritaria':float(np.mean(yt==np.bincount(estado['y'][estado['indices']['train']],minlength=3).argmax()))}
                st.rerun()
        if 'prueba' in st.session_state:
            r = st.session_state['prueba']; test = r['mlp']
            a,b,c,d = st.columns(4)
            a.metric('Accuracy test',f"{test['accuracy']:.2%}")
            b.metric('F1 macro test',f"{test['f1_macro']:.3f}")
            c.metric('Accuracy lineal',f"{r['base']['accuracy']:.2%}")
            d.metric('Clase mayoritaria',f"{r['mayoritaria']:.2%}")
            chip('Evaluación final')
            grafica('Matriz de prueba',lambda: matriz_confusion(test['matriz'],'Prueba'))
            st.dataframe(pd.DataFrame(test['por_clase']).assign(nombre=list(CLASES)),
                         hide_index=True,use_container_width=True)
        with st.expander('Alta confianza'):
            st.caption('Precisión y cobertura del conjunto seleccionado, por separado de la accuracy general')
            if st.button('Evaluar alta confianza', key='evaluar_alta_confianza'):
                try:
                    with reloj.medir('Evaluación alta confianza'):
                        umbral_actual, ajuste = umbral_validado(datos, estado)
                        modelos_ac = cargar_modelos(Path(__file__).parent / 'resultados')
                        ix = estado['indices']['test']
                        probabilidades_ac = predecir_conjunto(datos, ix, modelos_ac)
                        st.session_state['alta_confianza_probabilidades'] = probabilidades_ac
                        st.session_state['alta_confianza_etiquetas'] = estado['y'][ix]
                    st.rerun()
                except Exception as exc:
                    st.error(f'No se pudo evaluar: {exc}')
            if 'alta_confianza_probabilidades' in st.session_state:
                try:
                    umbral_actual, ajuste = umbral_validado(datos, estado)
                    r_ac = resumen(st.session_state['alta_confianza_etiquetas'],
                                   st.session_state['alta_confianza_probabilidades'], umbral_actual)
                    a,b,c = st.columns(3)
                    a.metric('Accuracy selectiva', f"{r_ac['accuracy_selectiva']:.2%}" if r_ac['accuracy_selectiva'] is not None else '—')
                    b.metric('Cobertura', f"{r_ac['cobertura']:.2%}")
                    c.metric('Partidos seleccionados', f"{r_ac['seleccionados']:,} / {r_ac['total']:,}")
                    st.caption(f"{r_ac['aciertos']} aciertos, {r_ac['no_seleccionados']:,} sin selección · Umbral {umbral_actual:.1%}")
                    st.caption(f"Accuracy general de las cuatro redes: {r_ac['accuracy_general_ensemble']:.2%}")
                    if ajuste is not None:
                        st.caption(f"Validación · {ajuste['val_accuracy']:.2%} accuracy · {ajuste['val_cobertura']:.2%} cobertura · objetivo {objetivo_auto:.0%}")
                        puntos = curva_cobertura(st.session_state['alta_confianza_etiquetas_val'],
                                               st.session_state['alta_confianza_probabilidades_val'])
                        figura = go.Figure()
                        figura.add_trace(go.Scatter(x=[p['cobertura']*100 for p in puntos],
                                  y=[p['accuracy']*100 for p in puntos],
                                  mode='lines', name='Validación', line=dict(color='#178579',width=3)))
                        figura.add_trace(go.Scatter(x=[r_ac['cobertura']*100],
                                  y=[r_ac['accuracy_selectiva']*100] if r_ac['accuracy_selectiva'] is not None else [0],
                                  mode='markers', name='Prueba', marker=dict(size=12,color='#E7603D')))
                        figura.update_layout(xaxis_title='Cobertura (%)',yaxis_title='Accuracy selectiva (%)',
                                             height=300,margin=dict(l=15,r=15,t=20,b=15))
                        grafica('Precisión y cobertura',figura)
                    st.dataframe(pd.DataFrame(r_ac['matriz'], index=CLASES, columns=CLASES), use_container_width=True)
                    chip('Evaluación alta confianza')
                except Exception as exc:
                    st.error(f'No se pudo mostrar: {exc}')
        with st.expander('Descargar resultados'):
            with reloj.medir('Preparar CSV de métricas'):
                csv = pd.DataFrame(estado['tabla']).to_csv(index=False).encode('utf-8')
            st.download_button('Comparación CSV',csv,'resultados_validacion.csv','text/csv',key='exportar_metricas')
            chip('Preparar CSV de métricas','Preparación CSV')
            with reloj.medir('Preparar modelo ZIP'):
                met = {'validacion_red':val,'validacion_lineal':base_val}
                if 'prueba' in st.session_state:
                    met['prueba_final'] = st.session_state['prueba']
                cfg = {**estado['config'],'modelo':mejor['meta'],'caracteristicas':mejor['estrategia']}
                paquete = exportar_experimento(mejor['red'],mejor['grupo']['media'],
                        mejor['grupo']['sd'],nombres(mejor['estrategia']),cfg,estado['fechas'],met)
            st.download_button('Modelo ZIP',paquete,'futbol_neuronal_modelo.zip','application/zip',key='exportar_modelo')
            chip('Preparar modelo ZIP','Preparación ZIP')

with paginas[2]:
    titulo('03 / LABORATORIO','Operaciones MLP: entrada a gradientes')
    if not estado:
        st.info('Entrena un modelo para consultar sus operaciones')
    else:
        m = estado['mejor']; red = m['red']; grupo = m['grupo']
        ss = estado['indices']['val']
        indice = st.slider('Selecciona un ejemplo de validación',0,len(ss)-1,0)
        with reloj.medir('Cálculo de neuronas'):
            pos = ss[indice]; x = grupo['x'][pos:pos+1]; y = np.array([estado['y'][pos]])
            partido = datos.iloc[pos]; cache = red.forward(x); prob = cache['prob'][0]
            perdida = entropia_multiclase(y,prob.reshape(1,3))
        st.write(f"**{partido['date'].date()}**  /  {partido['home_team']} {partido['home_score']}–{partido['away_score']} {partido['away_team']}  /  y={int(y[0])}")
        a,b,c = st.columns(3)
        a.metric('Clase de cálculo',CLASES[np.argmax(prob)])
        b.metric('Probabilidad real',f'{prob[int(y[0])]:.4f}')
        c.metric('Pérdida',f'{perdida:.6f}')
        chip('Cálculo de neuronas','Forward')
        grafica('Dibujo de arquitectura',lambda: arquitectura(red,nombres(m['estrategia'])))
        st.latex(r'Z^{[l]}=A^{[l-1]}W^{[l]}+b^{[l]},\quad A^{[l]}=g(Z^{[l]})')
        st.dataframe(pd.DataFrame({'Entrada':nombres(m['estrategia']),
            'Valor bruto':grupo['raw'][pos], 'Valor estandarizado':x[0]}).round(6),
            hide_index=True,use_container_width=True)
        for k in range(len(red.W)):
            nombre = f'Capa {k+1} / '+('Softmax' if k==len(red.W)-1 else red.tipo)
            with st.expander(nombre,expanded=k==len(red.W)-1):
                st.caption(f"W {red.W[k].shape}  /  Z {cache['z'][k].shape}")
                st.dataframe(pd.DataFrame(red.W[k]).round(6),use_container_width=True)
                st.write('Sesgo',red.b[k].round(6).tolist())
                st.write('Z',cache['z'][k].round(6).tolist())
                st.write('Activación',cache['a'][k+1].round(6).tolist())
        st.latex(r'p_k=\frac{e^{z_k}}{\sum_{j=0}^{2}e^{z_j}},\quad L=-\sum_{k=0}^{2}y_k\ln(p_k)')
        st.latex(r'\delta^{[L]}=p-\operatorname{onehot}(y),\quad \frac{\partial L}{\partial W^{[l]}}=(A^{[l-1]})^T\delta^{[l]}')
        with reloj.medir('Simulación de gradientes'):
            grad = red.gradientes(y,cache)
            clon = RedNeuronal(5,red.ocultas,red.tipo)
            clon.cargar_pesos(red.pesos())
            clon.paso_gradiente(grad,m['meta']['tasa'])
            despues = entropia_multiclase(y,clon.predecir_probabilidad(x))
        a,b,c = st.columns(3)
        a.metric('η',m['meta']['tasa'])
        b.metric('Pérdida inicial',f'{perdida:.6f}')
        c.metric('Después del paso',f'{despues:.6f}',delta=f'{(despues-perdida):+.6f}',delta_color='inverse')
        chip('Simulación de gradientes','Gradientes')
        with st.expander('Pesos y derivadas'):
            for k in range(len(red.W)):
                w = red.W[k]; g = grad[f'W{k+1}']
                st.markdown(f'**W{k+1}**')
                st.dataframe(pd.DataFrame({'Índice':[str(i) for i in np.ndindex(w.shape)],
                    'Antes':w.ravel(),'Gradiente':g.ravel(),'Después':clon.W[k].ravel()}).round(8),
                    hide_index=True,use_container_width=True)
                st.caption(f"Sesgo b{k+1}  {red.b[k].round(6).tolist()} → {clon.b[k].round(6).tolist()}")

with paginas[3]:
    titulo('04 / GEOMETRÍA','Curvas y regiones')
    if not estado:
        st.info('Entrena un modelo para ver las gráficas')
    else:
        m = estado['mejor']; fit = m['fit']
        grafica('Curvas de pérdida',lambda: curvas({'MLP':fit,'Lineal':estado['baseline']['fit']},'perdida'))
        grafica('Curvas de accuracy',lambda: curvas({'MLP':fit,'Lineal':estado['baseline']['fit']},'accuracy'))
        col1,col2 = st.columns(2)
        cols = nombres(m['estrategia'])
        i = col1.selectbox('Eje X',range(5),format_func=lambda n:cols[n],index=0)
        j = col2.selectbox('Eje Y',[k for k in range(5) if k!=i],format_func=lambda n:cols[n],index=0)
        ep = sorted(fit['instantaneas'])
        opciones = ['Modelo elegido'] + [f'Época {k}' for k in ep]
        elegido = st.select_slider('Época de la frontera',options=opciones,value='Modelo elegido')
        pesos = None if elegido=='Modelo elegido' else fit['instantaneas'][int(elegido.split()[-1])]
        ix = estado['indices']['val']
        with reloj.medir('Calcular frontera 2D'):
            figura = frontera_2d(m['red'],m['grupo']['x'][ix],estado['y'][ix],(i,j),cols,pesos)
        grafica('Render frontera',figura)
        chip('Calcular frontera 2D','Frontera')
        st.caption('Sección bidimensional del modelo de 5 entradas, las otras 3 se fijan en la mediana')
        st.caption(f"Épocas {fit['epocas_ejecutadas']}  /  Mejor {fit['mejor_epoca']}  /  Entrenamiento {mostrar_tiempo(fit['segundos'])}")

with paginas[4]:
    titulo('05 / PRONÓSTICO','Nuevo enfrentamiento')
    if not estado:
        st.info('Entrena un modelo para realizar pronósticos')
    else:
        m = estado['mejor']; hist = m['grupo']['hist']; grupo = m['grupo']
        equipos = sorted(set(datos.home_team).union(set(datos.away_team)))
        a,b = st.columns(2)
        local = a.selectbox('Local',equipos,index=equipos.index('Colombia') if 'Colombia' in equipos else 0)
        visita = b.selectbox('Visitante',equipos,index=equipos.index('Brazil') if 'Brazil' in equipos else min(1,len(equipos)-1))
        a,b = st.columns(2)
        neutral = a.checkbox('Sede neutral',value=True)
        oficial = b.checkbox('Torneo oficial',value=True)
        if local==visita:
            st.error('Selecciona selecciones diferentes')
        else:
            with reloj.medir('Pronóstico'):
                raw = hist.obtener(local,visita,neutral,'Competición oficial' if oficial else 'Friendly',m['estrategia'])
                x = escalar(raw.reshape(1,-1),grupo['media'],grupo['sd'])
                prob = m['red'].predecir_probabilidad(x)[0]
            st.markdown(f'<div class="match-result">{escape(CLASES[np.argmax(prob)])}<strong>{prob.max():.1%}</strong></div>',unsafe_allow_html=True)
            grafica('Gráfico de pronóstico',lambda: barras_probabilidad(prob))
            chip('Pronóstico','Tiempo de respuesta')
            st.dataframe(pd.DataFrame({'Variable':nombres(m['estrategia']),
                         'Valor':raw,'Estandarizado':x[0]}).round(4),hide_index=True,use_container_width=True)
            st.caption('Estadísticas hasta el último partido del archivo')
            with st.expander('Alta confianza'):
                if st.button('Calcular pronóstico selectivo', key='pronostico_confianza'):
                    try:
                        with reloj.medir('Pronóstico alta confianza'):
                            modelos_ac = cargar_modelos(Path(__file__).parent / 'resultados')
                            if 'historiales_alta_confianza' not in st.session_state:
                                st.session_state['historiales_alta_confianza'] = historial_para_pronosticos(datos, modelos_ac)
                            umbral_actual, ajuste = umbral_validado(datos, estado)
                            p_ac, seguro = predecir_enfrentamiento(modelos_ac, st.session_state['historiales_alta_confianza'],
                                local, visita, neutral, 'Competición oficial' if oficial else 'Friendly', umbral_actual)
                            st.session_state['ultimo_pronostico_selectivo'] = {
                                'partido': (local, visita, neutral, oficial), 'probabilidades': p_ac,
                                'seleccionado': seguro, 'umbral': umbral_actual, 'perfil': perfil_confianza}
                    except Exception as exc:
                        st.error(f'No se pudo calcular: {exc}')
                r_ac = st.session_state.get('ultimo_pronostico_selectivo')
                if r_ac and r_ac['partido'] == (local, visita, neutral, oficial) and r_ac['perfil'] == perfil_confianza:
                    p_ac = r_ac['probabilidades']
                    if np.max(p_ac) >= r_ac['umbral']:
                        st.success(f"Alta confianza, {CLASES[int(np.argmax(p_ac))]} · {max(p_ac):.1%}")
                    else:
                        st.info('Sin selección de alta confianza para este encuentro')
                    st.dataframe(pd.DataFrame({'Resultado': CLASES, 'Estimación': [f'{v:.1%}' for v in p_ac]}),
                                 hide_index=True, use_container_width=True)
                    chip('Pronóstico alta confianza')
                st.caption('Resultados históricos y cobertura del filtro disponibles en Rendimiento')

