"""Métricas y exportación del modelo"""
import io
import json
import zipfile
import numpy as np
from .red import entropia_multiclase


def metricas_multiclase(y,p):
    y=np.asarray(y,dtype=int).ravel()
    p=np.asarray(p,dtype=float)
    if p.shape!=(len(y),3):
        raise ValueError('La matriz de probabilidades debe ser (N,3)')
    pred=np.argmax(p,axis=1)
    cm=np.zeros((3,3),dtype=int)
    np.add.at(cm,(y,pred),1)
    per=[]
    for k in range(3):
        tp=cm[k,k];real=cm[k,:].sum();predicho=cm[:,k].sum()
        prec=float(tp/predicho) if predicho else 0.
        rec=float(tp/real) if real else 0.
        f1=float(2*prec*rec/(prec+rec)) if (prec+rec) else 0.
        per.append({'clase':k,'precision':prec,'recall':rec,'f1':f1,'soporte':int(real)})
    n=len(y)
    return {'accuracy':float(np.trace(cm)/n),'f1_macro':float(np.mean([d['f1'] for d in per])),
            'f1_ponderado':float(sum(d['f1']*d['soporte'] for d in per)/n),
            'cross_entropy':entropia_multiclase(y,p),
            'matriz':cm.tolist(),'por_clase':per}


def exportar_experimento(red,mu,sd,caracteristicas,config,limites,metricas):
    a=io.BytesIO(); pesos=io.BytesIO()
    np.savez_compressed(pesos,**red.pesos(),media=mu,desviacion=sd)
    meta={'version':2,'arquitectura':red.dimensiones,'activacion_oculta':red.tipo,
          'caracteristicas':caracteristicas,'configuracion':config,
          'divisiones':limites,'evaluacion':metricas,
          'aviso':'Clases 0=local, 1=empate, 2=visitante. No se guarda SQLite.'}
    with zipfile.ZipFile(a,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('pesos_modelo.npz',pesos.getvalue())
        z.writestr('experimento.json',json.dumps(meta,ensure_ascii=False,indent=2))
    return a.getvalue()
