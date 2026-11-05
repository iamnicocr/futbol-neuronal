"""Configuraciones y selección sobre validación"""
from .red import RedNeuronal, entrenar, puntuar
from .evaluacion import metricas_multiclase


def candidatos(capas,neuronas,activacion,tasa,semilla,automatico=False):
    tamanos=tuple(neuronas) if isinstance(neuronas,(tuple,list)) else (int(neuronas),)*capas
    if len(tamanos)!=capas:
        raise ValueError('Las neuronas no coinciden con las capas')
    s=int(semilla)
    original=(tamanos,activacion,float(tasa),s,1.0,'SGD')
    if not automatico:return [original]
    opciones=[original,
        ((8,), 'ReLU', .004, s, 1.0,'Adam'),
        ((8,), 'ReLU', .004, s, 1.35,'Adam'),
        ((8,8), 'ReLU', .003, s, 1.0,'Adam'),
        ((8,8), 'ReLU', .003, s, 1.4,'Adam'),
        ((6,6), 'ReLU', .006, s+11, 1.2,'Adam'),
        ((8,), 'Sigmoide', .006, s, 1.4,'Adam'),
        ((6,), 'Sigmoide', .007, s+11, 1.0,'Adam'),
        ((4,8), 'ReLU', .005, s+23, 1.6,'Adam'),
        ((8,8), 'ReLU', .006, s+11, 1.7,'Adam'),
        ((8,), 'ReLU', .008, s+23, 1.75,'Adam'),
        ((8,), 'ReLU', .05, s+11, 1.0,'SGD'),
        ((6,6), 'ReLU', .08, s+23, 1.5,'SGD'),
        ((8,), 'Sigmoide', .08, s+11, 1.0,'SGD'),
        ((8,8), 'Sigmoide', .006, s, 1.2,'Adam'),
        ((4,), 'ReLU', .01, s+42, 1.0,'Adam'),
        ((8,), 'ReLU', .003, s+42, 1.2,'Adam'),
        ((8,8), 'ReLU', .003, s+42, 1.3,'Adam'),
    ]
    return list(dict.fromkeys(opciones))


def buscar(x_train,y_train,x_val,y_val,opciones,epocas,lote,paciencia,criterio):
    resultados=[];mejor=None;clave_mejor=(-float('inf'),)*3
    for numero,opcion in enumerate(opciones,1):
        ocultas,act,eta,semilla,*extras=opcion
        peso_empate,optim=extras if extras else (1.,'SGD')
        red=RedNeuronal(5,ocultas,act,semilla)
        fit=entrenar(red,x_train,y_train,x_val,y_val,epocas,eta,lote,paciencia,criterio,semilla,
                     optimizador=optim,peso_empate=peso_empate,regularizacion=.0001 if optim=='Adam' else 0.)
        ev=metricas_multiclase(y_val,red.predecir_probabilidad(x_val))
        fila={'prueba':numero,'arquitectura':' → '.join(map(str,red.dimensiones)),
              'capas':len(ocultas),'neuronas':str(ocultas),'activacion':act,
              'optimizador':optim,'peso_empate':peso_empate,'tasa':eta,
              'sesgo_empate':fit['calibracion']['ajuste_empate'],
              'semilla':semilla,'epocas_utilizadas':fit['epocas_ejecutadas'],
              'accuracy_val':ev['accuracy'],'f1_macro_val':ev['f1_macro'],
              'loss_val':ev['cross_entropy'],'segundos':fit['segundos'],
              'parametros':red.parametros}
        resultados.append(fila)
        score,acc,f1=puntuar(y_val,red.predecir_probabilidad(x_val),criterio)
        clave=(round(score,8),round(f1,8),-ev['cross_entropy'])
        if clave>clave_mejor:
            clave_mejor=clave;mejor=(red,fit,ev,fila)
    return mejor,resultados
