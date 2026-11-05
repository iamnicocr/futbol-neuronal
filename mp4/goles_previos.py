from collections import defaultdict
from math import exp, log
import numpy as np
from .caracteristicas import Historial


class HistorialGoles(Historial):
    def __init__(self):
        super().__init__()
        self.ataque = defaultdict(float)
        self.defensa = defaultdict(float)
        self.ultima_fecha = {}
        self.fecha_actual = None

    def _preparar(self, equipo, fecha):
        if fecha is None:
            return
        anterior = self.ultima_fecha.get(equipo)
        if anterior is not None:
            dias = max(0, (fecha - anterior).days)
            factor = exp(-0.002 * dias)
            self.ataque[equipo] *= factor
            self.defensa[equipo] *= factor
        self.ultima_fecha[equipo] = fecha
        self.fecha_actual = fecha

    def _goles_esperados(self, local, visitante, neutral):
        ventaja = 0.0 if neutral else 0.10
        local_goles = exp(log(1.42) + ventaja + self.ataque[local] - self.defensa[visitante])
        visitante_goles = exp(log(1.12) + self.ataque[visitante] - self.defensa[local])
        return min(5.0, max(0.2, local_goles)), min(5.0, max(0.2, visitante_goles))

    def obtener(self, local, visitante, neutral=False, torneo='Friendly', modo=None, fecha=None):
        self._preparar(local, fecha)
        self._preparar(visitante, fecha)
        gl, gv = self._goles_esperados(local, visitante, neutral)
        pl = np.empty(9)
        pv = np.empty(9)
        pl[0] = exp(-gl)
        pv[0] = exp(-gv)
        for i in range(1, 9):
            pl[i] = pl[i-1] * gl / i
            pv[i] = pv[i-1] * gv / i
        empate_prob = float(pl @ pv) / float(pl.sum() * pv.sum())
        elo = self.elo_reFORZADO[local] - self.elo_reFORZADO[visitante]
        elo += 0 if neutral else 65
        oficial = float(str(torneo).strip().lower() != 'friendly')
        return np.array([elo, gl, gv, empate_prob, oficial], dtype=float)

    def incorporar(self, local, visitante, gl, gv, neutral=False):
        previsto_local, previsto_visitante = self._goles_esperados(local, visitante, neutral)
        dif_local = min(6.0, max(-6.0, gl - previsto_local))
        dif_visitante = min(6.0, max(-6.0, gv - previsto_visitante))
        tasa = 0.025
        self.ataque[local] = min(1.5, max(-1.5, self.ataque[local] + tasa*dif_local))
        self.defensa[visitante] = min(1.5, max(-1.5, self.defensa[visitante] - tasa*dif_local))
        self.ataque[visitante] = min(1.5, max(-1.5, self.ataque[visitante] + tasa*dif_visitante))
        self.defensa[local] = min(1.5, max(-1.5, self.defensa[local] - tasa*dif_visitante))
        super().incorporar(local, visitante, gl, gv, neutral)
