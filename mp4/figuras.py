"""Gráficas de la red neuronal"""
import numpy as np
import plotly.graph_objects as go
from .red import RedNeuronal, CLASES

TONOS=['#E7603D','#D0A044','#178579']


def estilo_grafica(fig):
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='#F1EEE5',plot_bgcolor='#F1EEE5',
        font=dict(family='Trebuchet MS, Arial, sans-serif',color='#182A2C',size=12),
        title_font=dict(family='Georgia, Times New Roman, serif',color='#182A2C',size=20),
        xaxis=dict(gridcolor='#D8D5CA',zerolinecolor='#182A2C'),
        yaxis=dict(gridcolor='#D8D5CA',zerolinecolor='#182A2C'),
        hoverlabel=dict(bgcolor='#182A2C',font_color='#F1EEE5'),
        legend=dict(bgcolor='rgba(0,0,0,0)'))
    return fig



def curvas(modelos,campo='perdida'):
    fig=go.Figure()
    claves=(('perdida_entrenamiento','Entrenamiento'),('perdida_validacion','Validación')) if campo=='perdida' else (
        ('exactitud_entrenamiento','Entrenamiento'),('exactitud_validacion','Validación'))
    for nombre,fit in modelos.items():
        hist=fit['historial']
        for clave,etiqueta in claves:
            fig.add_trace(go.Scatter(x=[e['epoca'] for e in hist],y=[e[clave] for e in hist],
               mode='lines',name=f'{nombre} · {etiqueta}',
               line=dict(width=2.5,dash='dot' if etiqueta=='Entrenamiento' else 'solid')))
    fig.update_layout(title='Entropía cruzada multiclase vs épocas' if campo=='perdida' else 'Accuracy vs épocas',
                      xaxis_title='Época',yaxis_title='Pérdida' if campo=='perdida' else 'Accuracy',
                      height=395,legend=dict(orientation='h',y=-.25),margin=dict(l=30,r=10,t=55,b=10))
    return estilo_grafica(fig)


def matriz_confusion(m,titulo='Matriz de confusión · prueba'):
    z=np.asarray(m,dtype=int)
    fig=go.Figure(go.Heatmap(z=z,x=['Local (0)','Empate (1)','Visitante (2)'],
            y=['Local (0)','Empate (1)','Visitante (2)'],colorscale=[[0,'#EEE9DC'],[0.42,'#EFAD72'],[1,'#A43F2B']],
            text=z,texttemplate='%{text}',textfont=dict(size=18),showscale=False))
    fig.update_layout(title=titulo,xaxis_title='Predicción',yaxis_title='Real',height=425,
                      margin=dict(l=20,r=10,t=55,b=45))
    return estilo_grafica(fig)


def arquitectura(red,caracteristicas):
    """Dibuja las conexiones y sus pesos"""
    capas=red.dimensiones
    fig=go.Figure()
    ys=[]
    for n in capas:
        ys.append(np.linspace((n-1)/2,-(n-1)/2,n))
    for k,w in enumerate(red.W):
        a=[];b=[];hover=[]
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                val=float(w[i,j]);color='#178579' if val>=0 else '#E7603D'
                fig.add_trace(go.Scatter(x=[k,k+1],y=[ys[k][i],ys[k+1][j]],mode='lines',
                    line=dict(width=.7+1.5*abs(val)/(np.max(abs(w))+1e-12),color=color),
                    showlegend=False,hoverinfo='skip'))
                a.append(k+.5);b.append((ys[k][i]+ys[k+1][j])/2)
                hover.append(f'W{k+1}[{i},{j}] = {val:+.6f}')
        fig.add_trace(go.Scatter(x=a,y=b,mode='markers',marker=dict(size=9,color='rgba(0,0,0,0)'),
                        text=hover,hovertemplate='%{text}<extra></extra>',showlegend=False))
    for k,n in enumerate(capas):
        etiquetas=caracteristicas if k==0 else list(CLASES) if k==len(capas)-1 else [f'h{k}.{i+1}' for i in range(n)]
        fig.add_trace(go.Scatter(x=[k]*n,y=ys[k],mode='markers+text',text=etiquetas,textposition='middle right',
              textfont=dict(size=10),marker=dict(size=20,color='#182A2C' if k==0 else '#E7603D' if k==len(capas)-1 else '#D0A044',line=dict(color='#F1EEE5',width=1)),
              name=f'Capa {k}',hovertemplate='%{text}<extra></extra>'))
    fig.update_layout(title='Arquitectura: '+' → '.join(map(str,capas))+' · '+str(red.parametros)+' parámetros',
        xaxis=dict(visible=False,range=[-.28,len(capas)-.30]),yaxis=dict(visible=False),
        height=490,margin=dict(l=80,r=110,t=55,b=10),showlegend=False)
    return estilo_grafica(fig)


def frontera_2d(red,x,y,indices,nombres,estado=None,semilla=42):
    """Sección en dos variables con las demás en la mediana"""
    if estado is not None:
        clon=RedNeuronal(5,red.ocultas,red.tipo)
        clon.cargar_pesos(estado)
    else:
        clon=red
    i,j=indices
    rng=np.random.default_rng(semilla)
    ids=np.sort(rng.choice(len(y),min(len(y),600),replace=False))
    lo=np.percentile(x[:,[i,j]],1,axis=0);hi=np.percentile(x[:,[i,j]],99,axis=0)
    hi=np.maximum(hi,lo+.2)
    xx=np.linspace(lo[0]-.4,hi[0]+.4,80)
    yy=np.linspace(lo[1]-.4,hi[1]+.4,80)
    X,Y=np.meshgrid(xx,yy)
    base=np.tile(np.median(x,axis=0),(X.size,1))
    base[:,i]=X.ravel();base[:,j]=Y.ravel()
    p=clon.predecir_probabilidad(base)
    etiqueta=np.argmax(p,axis=1).reshape(X.shape)
    fig=go.Figure()
    fig.add_trace(go.Heatmap(x=xx,y=yy,z=etiqueta,zmin=-.5,zmax=2.5,
       colorscale=[[0,TONOS[0]],[1/3,TONOS[0]],[1/3+.001,TONOS[1]],
                   [2/3,TONOS[1]],[2/3+.001,TONOS[2]],[1,TONOS[2]]],opacity=.35,showscale=False,
       hovertemplate='Región predicha: %{z}<extra></extra>'))
    for clase in range(3):
        mask=np.asarray(y)[ids]==clase
        fig.add_trace(go.Scatter(x=x[ids][mask,i],y=x[ids][mask,j],mode='markers',name=CLASES[clase],
          marker=dict(color=TONOS[clase],size=6,opacity=.75,line=dict(color='#182A2C',width=.45))))
    fig.update_layout(title='Frontera de decisión en 2D',
       xaxis_title=nombres[i]+' (estandarizada)',yaxis_title=nombres[j]+' (estandarizada)',height=480,
       margin=dict(l=20,r=15,t=50,b=30))
    return estilo_grafica(fig)


def barras_probabilidad(prob):
    fig = go.Figure(go.Bar(x=list(CLASES),y=[float(v)*100 for v in prob],
        marker_color=TONOS,text=[f'{v*100:.1f}%' for v in prob],textposition='outside')).update_layout(
        height=340,yaxis_title='Probabilidad estimada (%)',yaxis_range=[0,105],margin=dict(l=10,r=10,t=20,b=25))
    return estilo_grafica(fig)
