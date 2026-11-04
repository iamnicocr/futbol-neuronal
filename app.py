"""Interfaz del proyecto Fútbol Neuronal"""
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mp4.datos import leer_partidos, descargar_csv, DATASET_URL
from mp4.caracteristicas import (nombres, construir_variables, dividir_cronologicamente,
                                 ajustar_escalador, escalar)
from mp4.red import RedNeuronal, CLASES, entropia_multiclase
from mp4.evaluacion import metricas_multiclase, exportar_experimento
from mp4.experimentos import candidatos, buscar
from mp4.figuras import (curvas, matriz_confusion, arquitectura, frontera_2d,
                         barras_probabilidad)

st.set_page_config(page_title='Proyecto: Futbol Neuronal', page_icon='⚽',layout='wide')
st.markdown('''<style>
.block-container{padding-top:1.5rem;max-width:1400px}
.hero{border:1px solid rgba(102,167,255,.28);border-radius:16px;padding:22px 25px;margin-bottom:18px;background:linear-gradient(115deg,rgba(19,78,104,.4),rgba(17,92,76,.20))}
.hero h1{margin:.2em 0;font-size:2.2rem;letter-spacing:-.03em}
.hero p{margin:0;opacity:.8}
[data-testid="stMetric"]{padding:12px 16px;border:1px solid rgba(131,160,190,.25);border-radius:12px}
</style>''',unsafe_allow_html=True)
st.markdown('''<div class="hero"><h1>⚽ Proyecto: Futbol Neuronal</h1><p>3 clases · 5 características</p></div>''',unsafe_allow_html=True)

st.sidebar.header('Datos')
file=st.sidebar.file_uploader('Selecciona results.csv',type=['csv'])
ruta=Path(__file__).resolve().parent/'data'/'results.csv'
with st.sidebar.expander('Descargar archivo'):
    st.markdown(f'[Descargar results.csv]({DATASET_URL})')
    if st.button('Descargar del GitHub del autor'):
        try:
            descargar_csv(ruta)
            st.success('Archivo descargado, pulsa Cargar datos')
        except Exception as ex:
            st.error(f'No fue posible descargarlo: {ex}')
if ruta.is_file():
    st.sidebar.caption('✓ Archivo disponible')
if st.sidebar.button('📥 Cargar datos',use_container_width=True,type='primary'):
    try:
        origen=io.BytesIO(file.getvalue()) if file is not None else ruta
        with st.spinner('Cargando partidos...'):
            datos,info=leer_partidos(origen)
        st.session_state['datos']=(datos,info)
        st.session_state.pop('modelo',None)
        st.session_state.pop('prueba',None)
        st.sidebar.success(f'{len(datos):,} partidos válidos')
    except Exception as exc:
        st.sidebar.error(f'No se pudo cargar results.csv: {exc}')

st.sidebar.divider()
st.sidebar.header('Configuración')
disponible='datos' in st.session_state
if disponible:
    datos,info=st.session_state['datos']
    minimo=int(datos.date.dt.year.min()); maximo=int(datos.date.dt.year.max())
    anio=st.sidebar.slider('Año inicial',minimo,maximo,min(1990,maximo),help='Los partidos anteriores se usan para calcular estadísticas históricas')
else:
    anio=1990
modo=st.sidebar.selectbox('Características',['Forma y goles','Elo y forma'])
st.sidebar.caption('Redes de 5 entradas, sin suma de conjuntos')
capas=st.sidebar.slider('Capas ocultas',1,2,1)
neuronas1=st.sidebar.slider('Neuronas · capa oculta 1',4,8,8)
neuronas2=st.sidebar.slider('Neuronas · capa oculta 2',4,8,6,disabled=capas!=2)
config_ocultas=(neuronas1,neuronas2) if capas==2 else (neuronas1,)
act=st.sidebar.selectbox('Función de activación oculta',['ReLU','Sigmoide'])
epocas=st.sidebar.slider('Épocas máximas',20,500,150,step=10)
tasa=st.sidebar.select_slider('Tasa de aprendizaje η',options=[.001,.003,.005,.01,.02,.03,.05,.08,.1,.2],value=.03)
lote=st.sidebar.select_slider('Partidos por mini-lote',options=[128,256,512,1024,2048],value=512)
paciencia=st.sidebar.slider('Paciencia de parada',0,70,30,step=5,help='0 desactiva la parada anticipada')
criterio=st.sidebar.selectbox('Seleccionar por',options=['accuracy','f1_macro','loss'],format_func=lambda x:{'accuracy':'Mayor accuracy de validación','f1_macro':'Mayor F1 macro de validación','loss':'Menor pérdida de validación'}[x])
semilla=st.sidebar.number_input('Semilla',min_value=0,max_value=999999,value=42)
st.sidebar.divider()
busqueda=st.sidebar.checkbox('Probar varias arquitecturas',value=True,help='Compara configuraciones con validación, demora más')
max_candidatos=st.sidebar.slider('Configuraciones por conjunto',2,8,4,disabled=not busqueda)
ambos=st.sidebar.checkbox('Probar ambos conjuntos de variables',value=False,disabled=not busqueda)

if disponible and st.sidebar.button('▶ Entrenar modelo',use_container_width=True,type='primary'):
    try:
        with st.spinner('Preparando características...'):
            indices,fechas=dividir_cronologicamente(datos,int(anio))
            y=datos['y'].to_numpy(dtype=int)
            modos=['Forma y goles','Elo y forma'] if ambos else [modo]
            conjuntos={}
            for nombre in modos:
                raw,hist=construir_variables(datos,nombre)
                mu,sd=ajustar_escalador(raw[indices['train']])
                conjuntos[nombre]={'raw':raw,'x':escalar(raw,mu,sd),'hist':hist,'media':mu,'sd':sd}
        opciones=candidatos(capas,config_ocultas,act,tasa,semilla,busqueda)
        opciones=opciones[:max_candidatos] if busqueda else opciones
        barra=st.sidebar.progress(0,'Entrenando...')
        final=None;tabla=[];mejor_clave=(-float('inf'),-float('inf'))
        for j, (estrategia,grupo) in enumerate(conjuntos.items()):
            x=grupo['x']
            mejor,candidatos_tab=buscar(x[indices['train']],y[indices['train']],
                 x[indices['val']],y[indices['val']],opciones,epocas,lote,paciencia,criterio)
            for fila in candidatos_tab:
                fila['caracteristicas']=estrategia
                tabla.append(fila)
            red,fit,val,info_mejor=mejor
            score=val['accuracy'] if criterio=='accuracy' else val['f1_macro'] if criterio=='f1_macro' else -val['cross_entropy']
            clave=(round(score,8),-val['cross_entropy'])
            if clave>mejor_clave:
                final={'red':red,'fit':fit,'val':val,'estrategia':estrategia,
                       'meta':info_mejor,'grupo':grupo}
                mejor_clave=clave
            barra.progress((j+1)/len(conjuntos),f'Conjunto {j+1}/{len(conjuntos)}')
        # Referencia lineal para comparar los resultados
        from mp4.red import entrenar
        grupo=final['grupo']; x=grupo['x']
        base=RedNeuronal(5,(),act,semilla)
        fit_base=entrenar(base,x[indices['train']],y[indices['train']],
                     x[indices['val']],y[indices['val']],epocas,tasa,lote,paciencia,criterio,semilla)
        val_base=metricas_multiclase(y[indices['val']],base.predecir_probabilidad(x[indices['val']]))
        config={'anio':anio,'modo_inicial':modo,'capas_iniciales':capas,
                'neuronas_iniciales':config_ocultas,'activacion_inicial':act,'epocas_max':epocas,
                'tasa_inicial':tasa,'lote':lote,'paciencia':paciencia,
                'criterio':criterio,'semilla':int(semilla),'busqueda':busqueda,'ambas_estrategias':ambos}
        st.session_state['modelo']={'mejor':final,'baseline':{'red':base,'fit':fit_base,'val':val_base},
                                   'tabla':tabla,'y':y,'indices':indices,'fechas':fechas,
                                   'config':config,'conjuntos':conjuntos}
        st.session_state.pop('prueba',None)
        barra.empty();st.sidebar.success('Entrenamiento finalizado')
    except Exception as exc:
        st.sidebar.error(f'Error durante el entrenamiento: {exc}')

if not disponible:
    st.info('Carga el archivo results.csv desde el panel izquierdo para comenzar')
    st.stop()

datos,info=st.session_state['datos']
estado=st.session_state.get('modelo')
paginas=st.tabs(['📊 Datos','📈 Rendimiento','🧠 Red neuronal',
                 '📉 Gráficas','⚽ Pronósticos'])

with paginas[0]:
    st.subheader('Partidos')
    a,b,c,d=st.columns(4)
    a.metric('Partidos válidos',f'{len(datos):,}')
    b.metric('Selecciones',info['equipos'])
    c.metric('Primera fecha',info['primer_partido'])
    d.metric('Última fecha',info['ultimo_partido'])
    st.caption(('Sin registros descartados' if info['descartados']==0 else f"Registros descartados: {info['descartados']}") + ', los marcadores no se usan como entradas')
    conteos=datos['y'].value_counts().reindex([0,1,2],fill_value=0)
    fig=go.Figure(go.Bar(x=list(CLASES),y=conteos.values,marker_color=['#28c9a7','#eeb45d','#e47592']))
    fig.update_layout(title='Distribución de resultados',height=320,yaxis_title='Partidos')
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('**5 entradas**, la sexta columna `y` contiene el resultado real')
    if estado:
        nombre=estado['mejor']['estrategia'];grupo=estado['mejor']['grupo'];cols=nombres(nombre)
        st.caption('Características: '+nombre)
        tabla=pd.DataFrame(grupo['raw'],columns=cols)
        tabla['y']=estado['y']
        st.dataframe(tabla.tail(35).round(4),hide_index=True,use_container_width=True)
        ss=estado['indices'];f=estado['fechas']
        a,b,c=st.columns(3)
        a.metric('Entrenamiento',f"{len(ss['train']):,}")
        b.metric('Validación',f"{len(ss['val']):,}")
        c.metric('Prueba final',f"{len(ss['test']):,}")
        st.caption(f"Validación desde {f['validacion_desde']} · Prueba desde {f['prueba_desde']}")
    else:
        st.dataframe(datos[['date','home_team','away_team','home_score','away_score','neutral','y']].tail(35),hide_index=True,use_container_width=True)
        st.dataframe(pd.DataFrame({'Variable':nombres(modo),'Tipo':['Continua','Continua','Continua','Binaria','Binaria']}),hide_index=True,use_container_width=True)

with paginas[1]:
    st.subheader('Resultados del modelo')
    st.caption('Resultados de validación')
    if not estado:
        st.info('Configura y entrena el modelo desde el panel izquierdo')
    else:
        mejor=estado['mejor'];v=mejor['val'];bs=estado['baseline']['val']
        a,b,c,d=st.columns(4)
        a.metric('Accuracy validación',f"{v['accuracy']:.2%}",delta=f"{(v['accuracy']-bs['accuracy'])*100:+.2f} pp vs modelo lineal")
        b.metric('F1 macro validación',f"{v['f1_macro']:.3f}")
        c.metric('Cross-entropy validación',f"{v['cross_entropy']:.4f}")
        d.metric('Parámetros',mejor['red'].parametros)
        st.markdown('**Configuración:** '+str(mejor['meta']['arquitectura'])+
             f" · {mejor['red'].tipo} · η={mejor['meta']['tasa']} · época {mejor['fit']['mejor_epoca']}")
        st.plotly_chart(matriz_confusion(v['matriz'],'Matriz de confusión · VALIDACIÓN'),use_container_width=True)
        st.caption(f"Tiempo total: {sum(f['segundos'] for f in estado['tabla']) + estado['baseline']['fit']['segundos']:.2f} s · Modelo elegido: {mejor['fit']['segundos']:.2f} s")
        st.subheader('Comparación de modelos')
        tabla=pd.DataFrame(estado['tabla']).sort_values(['accuracy_val','f1_macro_val'],ascending=False)
        st.dataframe(tabla.style.format({'accuracy_val':'{:.3%}','f1_macro_val':'{:.4f}',
                       'loss_val':'{:.4f}','segundos':'{:.2f}'}),hide_index=True,use_container_width=True)
        st.markdown('**Modelo lineal de referencia**')
        st.write(f"Accuracy de validación: **{bs['accuracy']:.2%}** · F1 macro: **{bs['f1_macro']:.4f}** · parámetros: **{estado['baseline']['red'].parametros}**")
        st.divider()
        st.subheader('Evaluación final')
        if 'prueba' not in st.session_state:
            if st.button('Evaluar modelo en prueba',type='primary'):
                x=mejor['grupo']['x']; ix=estado['indices']['test']; yt=estado['y'][ix]
                pred=mejor['red'].predecir_probabilidad(x[ix]); pred_base=estado['baseline']['red'].predecir_probabilidad(x[ix])
                st.session_state['prueba']={'mlp':metricas_multiclase(yt,pred),
                    'base':metricas_multiclase(yt,pred_base),
                    'mayoritaria':float(np.mean(yt==np.bincount(estado['y'][estado['indices']['train']],minlength=3).argmax()))}
                st.rerun()
        else:
            resultado=st.session_state['prueba']; t=resultado['mlp']
            a,b,c,d=st.columns(4)
            a.metric('Accuracy test final',f"{t['accuracy']:.2%}")
            b.metric('F1 macro test',f"{t['f1_macro']:.3f}")
            c.metric('Accuracy baseline lineal',f"{resultado['base']['accuracy']:.2%}")
            d.metric('Clase mayoritaria',f"{resultado['mayoritaria']:.2%}")
            st.plotly_chart(matriz_confusion(t['matriz'],'Matriz de confusión · PRUEBA FINAL'),use_container_width=True)
            st.dataframe(pd.DataFrame(t['por_clase']).assign(nombre=list(CLASES)),hide_index=True,use_container_width=True)
            st.caption('Conjunto de prueba independiente')

        with st.expander('Descargar resultados'):
            tabla_exportar=pd.DataFrame(estado['tabla'])
            st.download_button('Descargar comparaciones CSV',tabla_exportar.to_csv(index=False).encode('utf-8'),
                               file_name='resultados_validacion.csv',mime='text/csv')
            met={'validacion_red':mejor['val'],'validacion_lineal':estado['baseline']['val']}
            if 'prueba' in st.session_state:
                met['prueba_final']=st.session_state['prueba']
            cfg={**estado['config'],'modelo':mejor['meta'],'caracteristicas':mejor['estrategia']}
            paquete=exportar_experimento(mejor['red'],mejor['grupo']['media'],mejor['grupo']['sd'],
                       nombres(mejor['estrategia']),cfg,estado['fechas'],met)
            st.download_button('Descargar modelo ZIP',paquete,file_name='futbol_neuronal_modelo.zip',mime='application/zip')

with paginas[2]:
    st.subheader('Operaciones MLP: 1 Entrada-Gradientes')
    if not estado:
        st.info('Entrena un modelo desde el panel izquierdo')
    else:
        mejor=estado['mejor'];r=mejor['red'];grupo=mejor['grupo']
        ss=estado['indices']['val'];indice=st.slider('Seleccione ejemplo de validación a estudiar',0,len(ss)-1,0)
        pos=ss[indice];x=grupo['x'][pos:pos+1];y=np.array([estado['y'][pos]])
        dfila=datos.iloc[pos]
        cache=r.forward(x);prob=cache['prob'][0]
        st.write(f"**Partido:** {dfila['date'].date()} · {dfila['home_team']} {dfila['home_score']}–{dfila['away_score']} {dfila['away_team']} · **y={int(y[0])} ({CLASES[int(y[0])]})**")
        a,b,c=st.columns(3)
        a.metric('Clase de cálculo',CLASES[np.argmax(prob)])
        b.metric('Probabilidad de la clase real',f'{prob[int(y[0])]:.4f}')
        c.metric('Pérdida del ejemplo',f'{entropia_multiclase(y,prob.reshape(1,3)):.6f}')
        st.plotly_chart(arquitectura(r,nombres(mejor['estrategia'])),use_container_width=True)
        st.latex(r'Z^{[l]}=A^{[l-1]}W^{[l]}+b^{[l]},\quad A^{[l]}=g(Z^{[l]})')
        st.dataframe(pd.DataFrame({'Entrada':nombres(mejor['estrategia']),
            'Valor bruto':grupo['raw'][pos],'Valor estandarizado':x[0]}).round(6),
            hide_index=True,use_container_width=True)
        for k in range(len(r.W)):
            titulo=f"Capa {k+1} · "+('Salida Softmax' if k==len(r.W)-1 else f'Oculta {r.tipo}')
            with st.expander(titulo,expanded=(k==len(r.W)-1)):
                st.markdown(f"**Forma matricial:** A anterior {cache['a'][k].shape}, W {r.W[k].shape}, b {r.b[k].shape}, Z {cache['z'][k].shape}")
                st.dataframe(pd.DataFrame(r.W[k]).round(6),use_container_width=True)
                st.write('**Sesgo b**',r.b[k].round(6).tolist())
                st.write('**Z (combinación afín)**',cache['z'][k].round(6).tolist())
                st.write('**A (activación)**',cache['a'][k+1].round(6).tolist())
        st.latex(r'p_k=\frac{e^{z_k}}{\sum_{j=0}^{2} e^{z_j}},\quad L=-\sum_{k=0}^{2}y_k\ln(p_k)')
        st.latex(r'\delta^{[L]}=p-\operatorname{onehot}(y),\quad \frac{\partial L}{\partial W^{[l]}}=(A^{[l-1]})^T\delta^{[l]}')
        st.latex(r'\delta^{[l-1]}=(\delta^{[l]} W^{[l]T})\odot g\prime(Z^{[l-1]}),\quad W\leftarrow W-\eta\nabla_W L')
        grad=r.gradientes(y,cache)
        clon=RedNeuronal(5,r.ocultas,r.tipo);clon.cargar_pesos(r.pesos())
        clon.paso_gradiente(grad,mejor['meta']['tasa'])
        perdida_antes=entropia_multiclase(y,cache['prob'])
        perdida_despues=entropia_multiclase(y,clon.predecir_probabilidad(x))
        a,b,c=st.columns(3)
        a.metric('η del candidato',mejor['meta']['tasa'])
        b.metric('Pérdida antes del paso',f'{perdida_antes:.6f}')
        c.metric('Pérdida después del paso',f'{perdida_despues:.6f}',delta=f'{(perdida_despues-perdida_antes):+.6f}',delta_color='inverse')
        st.caption('Simulación de un paso desde los pesos entrenados, una actualización individual puede aumentar la pérdida')
        with st.expander('Ver todos los pesos, gradientes y pesos actualizados'):
            for k in range(len(r.W)):
                g=grad[f'W{k+1}'];w=r.W[k];nuevo=clon.W[k]
                st.markdown(f'**Matriz W{k+1}**')
                st.dataframe(pd.DataFrame({'Elemento':[str(i) for i in np.ndindex(w.shape)],
                    'Antes':w.ravel(),'Gradiente':g.ravel(),'Después':nuevo.ravel()}).round(8),hide_index=True,use_container_width=True)
                st.markdown(f'**Sesgos b{k+1}** · '+str(r.b[k].round(6).tolist())+' → '+str(clon.b[k].round(6).tolist()))

with paginas[3]:
    st.subheader('Curvas de entrenamiento y frontera de decisión')
    if not estado:
        st.info('Entrena un modelo desde el panel izquierdo')
    else:
        m=estado['mejor'];fit=m['fit']
        st.plotly_chart(curvas({'Red neuronal':fit,'Modelo lineal':estado['baseline']['fit']},'perdida'),use_container_width=True)
        st.plotly_chart(curvas({'Red neuronal':fit,'Modelo lineal':estado['baseline']['fit']},'accuracy'),use_container_width=True)
        a,b=st.columns(2)
        cols=nombres(m['estrategia'])
        i=a.selectbox('Variable del eje X',range(5),format_func=lambda n:cols[n],index=0)
        j=b.selectbox('Variable del eje Y',[k for k in range(5) if k!=i],format_func=lambda n:cols[n],index=0)
        ep=list(sorted(fit['instantaneas']))
        opciones=['Modelo elegido']+[f'Época {k}' for k in ep]
        elegido=st.select_slider('Evolución de la frontera',options=opciones,value='Modelo elegido')
        pesos=None if elegido=='Modelo elegido' else fit['instantaneas'][int(elegido.split()[-1])]
        ix=estado['indices']['val']
        st.plotly_chart(frontera_2d(m['red'],m['grupo']['x'][ix],estado['y'][ix],(i,j),cols,pesos),use_container_width=True)
        st.caption('**Sección bidimensional** del modelo de 5 entradas, no representa la superficie de 5 dimensiones, las restantes se fijan en la mediana')
        st.markdown(f"Parada anticipada: **{'Sí' if fit['parada_anticipada'] else 'No'}** · Épocas realizadas: **{fit['epocas_ejecutadas']}** · Época seleccionada: **{fit['mejor_epoca']}** · Tiempo de entrenamiento: **{fit['segundos']:.2f} s**")

with paginas[4]:
    st.subheader('Pronóstico de partido')
    if not estado:
        st.info('Entrena un modelo desde el panel izquierdo')
    else:
        m=estado['mejor'];hist=m['grupo']['hist'];grupo=m['grupo']
        equipos=sorted(set(datos.home_team).union(set(datos.away_team)))
        a,b=st.columns(2)
        local=a.selectbox('Selección local',equipos,index=equipos.index('Colombia') if 'Colombia' in equipos else 0)
        visitante=b.selectbox('Selección visitante',equipos,index=equipos.index('Brazil') if 'Brazil' in equipos else min(1,len(equipos)-1))
        a,b=st.columns(2)
        neutral=a.checkbox('Sede neutral',value=True)
        oficial=b.checkbox('Torneo oficial',value=True)
        if local==visitante:
            st.error('Elige selecciones diferentes.')
        else:
            raw=hist.obtener(local,visitante,neutral,'Competición oficial' if oficial else 'Friendly',m['estrategia'])
            x=escalar(raw.reshape(1,-1),grupo['media'],grupo['sd'])
            prob=m['red'].predecir_probabilidad(x)[0]
            st.plotly_chart(barras_probabilidad(prob),use_container_width=True)
            st.success(f'Pronóstico: **{CLASES[np.argmax(prob)]}** · **{prob.max():.2%}**')
            st.dataframe(pd.DataFrame({'Variable':nombres(m['estrategia']),'Valor prepartido':raw,
                        'Valor estandarizado':x[0]}).round(4),hide_index=True,use_container_width=True)
            st.caption('Estadísticas hasta el último partido del archivo')
