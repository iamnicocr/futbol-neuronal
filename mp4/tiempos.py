"""Medición de tiempos de procesamiento"""
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter


def mostrar_tiempo(segundos):
    if segundos is None:
        return '—'
    if segundos < 0.001:
        return f'{segundos * 1000:.2f} ms'
    if segundos < 1:
        return f'{segundos * 1000:.1f} ms'
    if segundos < 60:
        return f'{segundos:.2f} s'
    minutos, resto = divmod(segundos, 60)
    return f'{int(minutos)} min {resto:04.1f} s'


class RegistroTiempos:
    def __init__(self, limite=80):
        self.limite = limite
        self.eventos = []
        self.ultimos = {}

    @contextmanager
    def medir(self, nombre):
        inicio = perf_counter()
        correcto = False
        try:
            yield
            correcto = True
        finally:
            segundos = perf_counter() - inicio
            evento = {'accion': nombre, 'segundos': segundos,
                      'hora': datetime.now().strftime('%H:%M:%S'),
                      'estado': 'OK' if correcto else 'Error'}
            self.eventos.append(evento)
            self.eventos = self.eventos[-self.limite:]
            self.ultimos[nombre] = evento

    def guardar(self, nombre, segundos, correcto=True):
        evento = {'accion': nombre, 'segundos': float(segundos),
                  'hora': datetime.now().strftime('%H:%M:%S'),
                  'estado': 'OK' if correcto else 'Error'}
        self.eventos.append(evento)
        self.eventos = self.eventos[-self.limite:]
        self.ultimos[nombre] = evento

    def ultimo(self, nombre):
        return self.ultimos.get(nombre, {}).get('segundos')

    def fila(self, nombre):
        return self.ultimos.get(nombre)
