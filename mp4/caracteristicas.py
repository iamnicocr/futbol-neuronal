"""Características calculadas con los resultados anteriores a cada partido"""
from collections import defaultdict, deque
from math import exp, log1p
import numpy as np

FEATURES_FORMA = ['diff_form','diff_gf','diff_ga','is_neutral','is_tournament']
FEATURES_ELO = ['diff_elo','diff_form','diff_goles_neto','is_neutral','is_tournament']
FEATURES_GOLES = ['diff_elo','goles_previstos_local','goles_previstos_visitante','prob_empate_previa','is_tournament']
PESOS5 = {n: tuple(exp(-.23*(n-j-1)) for j in range(n)) for n in range(1,6)}
PESOS10 = {n: tuple(exp(-.12*(n-j-1)) for j in range(n)) for n in range(1,11)}


def nombres(modo):
    if modo=='Forma y goles':
        return FEATURES_FORMA.copy()
    if modo=='Elo y goles esperados':
        return FEATURES_GOLES.copy()
    if modo in ('Elo y forma','Elo reforzado'):
        return FEATURES_ELO.copy()
    raise ValueError('Estrategia desconocida')


class Historial:
    def __init__(self):
        self.partidos=defaultdict(lambda:deque(maxlen=10))
        self.elo=defaultdict(lambda:1500.0)
        self.elo_reFORZADO=defaultdict(lambda:1500.0)
        self.total=defaultdict(int)

    def obtener(self, local, visitante, neutral=False, torneo='Friendly', modo='Forma y goles'):
        def calculos(equipo):
            h=list(self.partidos[equipo])
            ult5=h[-5:]; ult10=h[-10:]
            winrate=sum(v[0] for v in ult5)/len(ult5) if ult5 else 0.0
            gf=sum(v[1] for v in ult10)/len(ult10) if ult10 else 0.0
            ga=sum(v[2] for v in ult10)/len(ult10) if ult10 else 0.0
            return winrate,gf,ga
        fl,gfl,gal=calculos(local);fv,gfv,gav=calculos(visitante)
        n=float(bool(neutral)); t=float(str(torneo).strip().lower()!='friendly')
        if modo=='Forma y goles':
            return np.array([fl-fv,gfl-gfv,gal-gav,n,t],dtype=float)
        if modo=='Elo y forma':
            return np.array([self.elo[local]-self.elo[visitante],fl-fv,
                             (gfl-gal)-(gfv-gav),n,t],dtype=float)
        if modo=='Elo reforzado':
            def rendimiento(equipo):
                partidos=list(self.partidos[equipo])
                ult5=partidos[-5:]
                ult10=partidos[-10:]
                if not ult5:return .5,0.
                pf=PESOS5[len(ult5)]
                suma=0.
                for peso,v in zip(pf,ult5):
                    puntos=max(0.,min(1.,v[3]+.12*(v[4]-1500.)/400.))
                    suma+=peso*puntos
                forma=(suma+.75)/(sum(pf)+1.5)
                pg=PESOS10[len(ult10)]
                suma=0.
                for peso,v in zip(pg,ult10):
                    dif=max(-4,min(4,v[1]-v[2]))
                    suma+=peso*dif
                goles=suma/(sum(pg)+2.)
                return forma,goles
            forma_l,goles_l=rendimiento(local)
            forma_v,goles_v=rendimiento(visitante)
            return np.array([self.elo_reFORZADO[local]-self.elo_reFORZADO[visitante],
                             forma_l-forma_v,goles_l-goles_v,n,t],dtype=float)
        raise ValueError('Estrategia desconocida')

    def incorporar(self,local,visitante,gl,gv,neutral=False):
        gl=int(gl);gv=int(gv)
        pl=1.0 if gl>gv else 0.0;pv=1.0 if gl<gv else 0.0
        rl=self.elo[local];rv=self.elo[visitante]
        al=self.elo_reFORZADO[local];av=self.elo_reFORZADO[visitante]
        resultado=1.0 if gl>gv else 0.5 if gl==gv else 0.0
        self.partidos[local].append((pl,gl,gv,resultado,av))
        self.partidos[visitante].append((pv,gv,gl,1.-resultado,al))
        ventaja=0.0 if neutral else 60.0
        e_local=1/(1+10**((rv-rl-ventaja)/400))
        resultado=1.0 if gl>gv else 0.5 if gl==gv else 0.0
        cambio=20*(resultado-e_local)
        self.elo[local]=rl+cambio;self.elo[visitante]=rv-cambio
        esperado=1./(1.+10**((av-al-(0 if neutral else 70))/400.))
        margen=1.+.35*log1p(abs(gl-gv))
        movimiento=24.*margen*(resultado-esperado)
        self.elo_reFORZADO[local]=al+movimiento
        self.elo_reFORZADO[visitante]=av-movimiento
        self.total[local]+=1;self.total[visitante]+=1


def construir_variables(partidos, modo='Forma y goles'):
    if modo=='Elo y goles esperados':
        from .goles_previos import HistorialGoles
        hist=HistorialGoles()
    else:
        hist=Historial()
    x=np.empty((len(partidos),5),dtype=float)
    fecha_actual=None
    pendientes=[]
    for pos,fila in enumerate(partidos.itertuples(index=False)):
        if fecha_actual is not None and fila.date!=fecha_actual:
            for partido in pendientes:
                hist.incorporar(partido.home_team,partido.away_team,
                    partido.home_score,partido.away_score,partido.neutral)
            pendientes.clear()
        fecha_actual=fila.date
        if modo=='Elo y goles esperados':
            x[pos]=hist.obtener(fila.home_team,fila.away_team,
                                fila.neutral,fila.tournament,fecha=fila.date)
        else:
            x[pos]=hist.obtener(fila.home_team,fila.away_team,
                                fila.neutral,fila.tournament,modo)
        pendientes.append(fila)
    for partido in pendientes:
        hist.incorporar(partido.home_team,partido.away_team,
            partido.home_score,partido.away_score,partido.neutral)
    return x,hist


def dividir_cronologicamente(partidos,min_year=1990,frac_train=.70,frac_val=.15):
    if not (0<frac_train<1 and 0<frac_val<1 and frac_train+frac_val<1):
        raise ValueError('Fracciones incorrectas')
    sub=partidos.loc[partidos.date.dt.year>=min_year]
    if len(sub)<100 or sub.date.nunique()<15:
        raise ValueError('No hay partidos suficientes en ese período')
    c1=sub['date'].iloc[int(len(sub)*frac_train)]
    c2=sub['date'].iloc[int(len(sub)*(frac_train+frac_val))]
    m=partidos.date.dt.year>=min_year
    train=np.flatnonzero((m & (partidos.date<c1)).to_numpy())
    val=np.flatnonzero((m & (partidos.date>=c1)&(partidos.date<c2)).to_numpy())
    test=np.flatnonzero((m & (partidos.date>=c2)).to_numpy())
    if min(map(len,(train,val,test)))<10:
        raise ValueError('No hay suficientes partidos por subconjunto')
    return {'train':train,'val':val,'test':test}, {'validacion_desde':str(c1.date()),'prueba_desde':str(c2.date())}


def ajustar_escalador(x):
    x=np.asarray(x,dtype=float)
    mu=x.mean(axis=0);sd=x.std(axis=0)
    return mu,np.where(sd<1e-10,1.,sd)


def escalar(x,mu,sd):
    return np.clip((np.asarray(x,dtype=float)-mu)/sd,-8.,8.)
