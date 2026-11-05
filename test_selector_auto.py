import numpy as np
from mp4.confianza import seleccionar_umbral, limite_inferior_wilson, resumen


def test_umbral_se_calcula_solo_con_validacion():
    y = np.array([0]*200)
    prob = np.array([[.98,.01,.01]]*100+[[.45,.50,.05]]*100)
    r = seleccionar_umbral(y,prob,.9,minimo=40)
    assert r is not None
    assert r['val_seleccionados'] == 100
    assert r['val_accuracy'] == 1.0
    assert r['umbral'] == .98


def test_rechaza_casos_sin_suficiente_evidencia():
    y = np.array([1]*130)
    prob = np.tile([.98,.01,.01],(130,1))
    assert seleccionar_umbral(y,prob,.90,minimo=80) is None


def test_wilson_no_equivale_a_accuracy_bruta():
    assert limite_inferior_wilson(90,100)<.9
    assert limite_inferior_wilson(200,200)>.9


def test_la_prueba_no_se_usa_para_elegir_umbral():
    y = np.array([0]*120+[1]*120)
    prob = np.array([[.96,.02,.02]]*120+[[.2,.7,.1]]*120)
    r = seleccionar_umbral(y,prob,.9,minimo=80)
    evaluacion = resumen(np.array([1,2]),np.array([[.96,.02,.02],[.01,.02,.97]]),r['umbral'])
    assert evaluacion['aciertos'] == 1
    assert r['val_accuracy'] == 1.0
