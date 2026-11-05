"""Red neuronal de tres salidas"""
import time
import numpy as np

CLASES = ('Victoria local', 'Empate', 'Victoria visitante')


def sigmoide(z):
    z = np.asarray(z, dtype=float)
    salida = np.empty_like(z)
    pos = z >= 0
    salida[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    salida[~pos] = ez / (1 + ez)
    return salida


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def entropia_multiclase(y, probabilidades):
    y = np.asarray(y, dtype=int).ravel()
    p = np.asarray(probabilidades, dtype=float)
    if len(y) != len(p) or p.shape[1] != 3:
        raise ValueError('Se esperan 3 probabilidades por partido')
    return float(np.mean(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))))


def activar(z, tipo):
    return np.maximum(z, 0) if tipo == 'ReLU' else sigmoide(z)


def derivada(z, a, tipo):
    return (z > 0).astype(float) if tipo == 'ReLU' else a * (1-a)


class RedNeuronal:
    def __init__(self, entradas=5, ocultas=(6,), activacion_oculta='ReLU', semilla=42):
        self.entradas = int(entradas)
        self.ocultas = tuple(int(n) for n in ocultas)
        self.tipo = activacion_oculta
        self.salidas = 3
        if self.entradas != 5 or len(self.ocultas) > 2 or any(not 4 <= n <= 8 for n in self.ocultas):
            raise ValueError('Se requieren 5 entradas, 0–2 capas ocultas y 4–8 neuronas por capa')
        if self.tipo not in ('ReLU', 'Sigmoide'):
            raise ValueError('La activación oculta debe ser ReLU o Sigmoide')
        rng = np.random.default_rng(int(semilla))
        self.dimensiones = [self.entradas, *self.ocultas, self.salidas]
        self.W = []
        self.b = []
        for n_in, n_out in zip(self.dimensiones[:-1],self.dimensiones[1:]):
            factor = np.sqrt((2 if self.tipo == 'ReLU' else 1) / n_in)
            self.W.append(rng.normal(0, factor, (n_in,n_out)))
            self.b.append(np.zeros((1,n_out)))

    @property
    def parametros(self):
        return sum(w.size+b.size for w,b in zip(self.W,self.b))

    def pesos(self):
        return {**{f'W{i+1}': w.copy() for i,w in enumerate(self.W)},
                **{f'b{i+1}': b.copy() for i,b in enumerate(self.b)}}

    def cargar_pesos(self, estado):
        for i in range(len(self.W)):
            for pref, destino in (('W',self.W),('b',self.b)):
                nuevo = np.asarray(estado[f'{pref}{i+1}'], dtype=float)
                if nuevo.shape != destino[i].shape:
                    raise ValueError('Dimensiones incompatibles al cargar parámetros')
                destino[i] = nuevo.copy()

    def forward(self, x):
        x = np.asarray(x,dtype=float).reshape(-1,self.entradas)
        a = [x]
        z = []
        for i,(wi,bi) in enumerate(zip(self.W,self.b)):
            zi = a[-1] @ wi + bi
            z.append(zi)
            a.append(softmax(zi) if i==len(self.W)-1 else activar(zi,self.tipo))
        return {'x':x, 'a':a, 'z':z, 'prob':a[-1]}

    def gradientes(self, y, cache, pesos_clase=None):
        y = np.asarray(y,dtype=int).ravel()
        n = len(y)
        delta = cache['prob'].copy()
        delta[np.arange(n),y] -= 1
        if pesos_clase is None:
            delta /= n
        else:
            factores = np.asarray(pesos_clase, dtype=float)[y]
            delta *= (factores / np.sum(factores))[:, None]
        grad = {}
        for i in range(len(self.W)-1,-1,-1):
            grad[f'W{i+1}'] = cache['a'][i].T @ delta
            grad[f'b{i+1}'] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * derivada(cache['z'][i-1], cache['a'][i],self.tipo)
        return grad

    def paso_gradiente(self, grad, tasa):
        for i in range(len(self.W)):
            self.W[i] -= float(tasa)*grad[f'W{i+1}']
            self.b[i] -= float(tasa)*grad[f'b{i+1}']

    def predecir_probabilidad(self, x):
        return self.forward(x)['prob']


def evaluar_rapido(y, p):
    pre = np.argmax(p,axis=1)
    return float(np.mean(pre == y))


def puntuar(y, probabilidades, criterio):
    """Las métricas se calculan solo sobre las etiquetas disponibles."""
    pred = np.argmax(probabilidades, axis=1)
    acc = float(np.mean(pred == y))
    if criterio == 'loss':
        return -entropia_multiclase(y, probabilidades), acc, 0.
    cm = np.bincount(3*np.asarray(y, dtype=int)+pred, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=1) + cm.sum(axis=0)
    f1 = float(np.mean(np.divide(2*tp, denom, out=np.zeros(3), where=denom>0)))
    if criterio == 'f1_macro':
        return f1, acc, f1
    if criterio == 'equilibrio':
        return .7*acc + .3*f1, acc, f1
    return acc, acc, f1


def calibrar_salida(red, x_val, y_val, criterio):
    """Ajusta exclusivamente los sesgos de salida usando validación."""
    p = red.predecir_probabilidad(x_val)
    logits = np.log(np.clip(p, 1e-12, 1))
    original = puntuar(y_val, p, criterio)
    mejor = (original[0], original[1], original[2], -entropia_multiclase(y_val,p), 0.)
    ajuste = (0., 0.)
    for local in (-.30,-.15,0.,.15,.30):
        for empate in (-.70,-.50,-.30,-.15,0.,.15,.30,.45,.60,.75,.90,1.05,1.20):
            # La regularización del ajuste favorece cambios pequeños en empates de score.
            cambios = np.array([local, empate, 0.])
            z = logits + cambios
            pred = np.argmax(z,axis=1)
            acc = float(np.mean(pred==y_val))
            if criterio == 'loss':
                q = softmax(z)
                score = -entropia_multiclase(y_val,q)
                f1 = 0.
                loss = -score
            else:
                cm = np.bincount(3*y_val+pred,minlength=9).reshape(3,3)
                tp = np.diag(cm).astype(float)
                denom = cm.sum(axis=0)+cm.sum(axis=1)
                f1 = float(np.mean(np.divide(2*tp,denom,out=np.zeros(3),where=denom>0)))
                score = f1 if criterio=='f1_macro' else (.7*acc+.3*f1 if criterio=='equilibrio' else acc)
                loss = entropia_multiclase(y_val,softmax(z))
            clave=(round(score, 7), round(acc,7), round(f1,7), -loss, -(local*local+empate*empate))
            if clave > mejor:
                mejor=clave
                ajuste=(local,empate)
    red.b[-1][0,0] += ajuste[0]
    red.b[-1][0,1] += ajuste[1]
    return {'ajuste_local':float(ajuste[0]),'ajuste_empate':float(ajuste[1])}


def entrenar(red, x_train, y_train, x_val, y_val, epocas=120, tasa=0.05,
             lote=512, paciencia=25, criterio='accuracy', semilla=42,
             guardar_epocas=(0, 10, 25, 50, 100), optimizador='SGD',
             peso_empate=1., regularizacion=0.):
    if not 1 <= epocas <= 1000 or not 0 < tasa <= 2 or not 16 <= lote <= 8192:
        raise ValueError('Revisar épocas, tasa y tamaño del lote')
    if criterio not in ('accuracy','loss','f1_macro','equilibrio'):
        raise ValueError('Criterio desconocido')
    if optimizador not in ('SGD','Adam') or not .5 <= peso_empate <= 3.:
        raise ValueError('Optimizador o ponderación inválidos')
    x_train=np.asarray(x_train,dtype=float); y_train=np.asarray(y_train,dtype=int)
    x_val=np.asarray(x_val,dtype=float); y_val=np.asarray(y_val,dtype=int)
    rng=np.random.default_rng(int(semilla))
    pesos_clase=np.array([1.,peso_empate,1.]) if peso_empate!=1 else None
    mejor_clave=(-float('inf'),-float('inf'),-float('inf'))
    mejor_estado=red.pesos();mejor_epoca=0;sin_mejora=0
    instantaneas={0:red.pesos()};historial=[]
    momentos={k:[np.zeros_like(w) for w in lista] for k,lista in (('W',red.W),('b',red.b))}
    cuadrados={k:[np.zeros_like(w) for w in lista] for k,lista in (('W',red.W),('b',red.b))}
    pasos=0
    inicio=time.perf_counter()
    for epoca in range(1,int(epocas)+1):
        orden=rng.permutation(len(y_train))
        for comienzo in range(0,len(orden),int(lote)):
            ids=orden[comienzo:comienzo+int(lote)]
            cache=red.forward(x_train[ids]);grad=red.gradientes(y_train[ids],cache,pesos_clase)
            if regularizacion:
                for k,w in enumerate(red.W):
                    grad[f'W{k+1}'] += regularizacion*w
            if optimizador=='Adam':
                pasos += 1
                for k in range(len(red.W)):
                    for pref, matrices in (('W',red.W),('b',red.b)):
                        g=grad[f'{pref}{k+1}']
                        norm=np.linalg.norm(g)
                        if norm>5.:g=g*(5./norm)
                        m=momentos[pref][k];v=cuadrados[pref][k]
                        m *= .9;m += .1*g
                        v *= .999;v += .001*g*g
                        matrices[k] -= tasa*(m/(1-.9**pasos))/(np.sqrt(v/(1-.999**pasos))+1e-8)
            else:
                red.paso_gradiente(grad,tasa)
        ptr=red.predecir_probabilidad(x_train)
        pval=red.predecir_probabilidad(x_val)
        atr=evaluar_rapido(y_train,ptr);ava=evaluar_rapido(y_val,pval)
        ltr=entropia_multiclase(y_train,ptr);lva=entropia_multiclase(y_val,pval)
        score,_,f1=puntuar(y_val,pval,criterio)
        clave=(round(score,8),round(f1,8),-lva)
        historial.append({'epoca':epoca,'perdida_entrenamiento':ltr,
                          'perdida_validacion':lva,'exactitud_entrenamiento':atr,
                          'exactitud_validacion':ava})
        if not np.isfinite(ltr+lva):
            raise FloatingPointError('Pérdida no finita, reduce la tasa')
        if clave>mejor_clave:
            mejor_clave=clave;mejor_estado=red.pesos();mejor_epoca=epoca;sin_mejora=0
        else:sin_mejora+=1
        if epoca in guardar_epocas:
            instantaneas[epoca]=red.pesos()
        if paciencia and sin_mejora>=int(paciencia) and epoca>=max(20,int(paciencia)):
            break
    red.cargar_pesos(mejor_estado)
    calibracion=calibrar_salida(red,x_val,y_val,criterio)
    segundos=time.perf_counter()-inicio
    pval=red.predecir_probabilidad(x_val)
    return {'historial':historial,'mejor_epoca':mejor_epoca,
            'epocas_ejecutadas':len(historial),'perdida_validacion':entropia_multiclase(y_val,pval),
            'accuracy_validacion':evaluar_rapido(y_val,pval),
            'segundos':segundos,'instantaneas':instantaneas,
            'optimizador':optimizador,'peso_empate':peso_empate,'calibracion':calibracion,
            'parada_anticipada':len(historial)<epocas}
