"""Carga y limpieza de resultados de fútbol"""
from pathlib import Path
import io
import numpy as np
import pandas as pd

DATASET_URL = 'https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017'
RAW_URL = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
COLUMNAS = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament', 'city', 'country', 'neutral']


def leer_partidos(origen):
    """Acepta una ruta local, un fichero subido o un buffer CSV."""
    df = pd.read_csv(origen)
    ausentes = [c for c in COLUMNAS if c not in df.columns]
    if ausentes:
        raise ValueError('Faltan columnas del results.csv oficial: ' + ', '.join(ausentes))
    n_original = len(df)
    df = df[COLUMNAS].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
    for col in ('home_team', 'away_team', 'tournament'):
        df[col] = df[col].astype('string').str.strip()
    # Excluir encuentros sin marcador
    df = df.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'])
    df = df[(df['home_team'] != '') & (df['away_team'] != '')]
    df = df[(df['home_score'] >= 0) & (df['away_score'] >= 0)]
    df = df[(df['home_score'] % 1 == 0) & (df['away_score'] % 1 == 0)]
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    df['tournament'] = df['tournament'].fillna('Desconocido')
    
    df['neutral'] = df['neutral'].astype('string').str.lower().isin(['true', '1', 'yes', 'sí', 'si'])
    df = df.sort_values(['date', 'home_team', 'away_team'], kind='stable').reset_index(drop=True)
    if df.empty:
        raise ValueError('No hay partidos válidos con resultado completo')
    df['y'] = np.select([df['home_score'] > df['away_score'],df['home_score'] == df['away_score']], [0,1],default=2).astype(int)
    informe = dict(total_original=n_original, total_limpio=len(df), descartados=n_original-len(df),
                   primer_partido=str(df['date'].min().date()), ultimo_partido=str(df['date'].max().date()),
                   equipos=int(pd.unique(pd.concat([df['home_team'], df['away_team']])).size))
    return df, informe


def descargar_csv(destino):
    """Descarga results.csv"""
    import urllib.request
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(RAW_URL, destino)
    return destino
