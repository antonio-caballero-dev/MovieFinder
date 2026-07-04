GENRES_IMDB = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Sci-Fi",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western"
}

GENRES_IMDB_INVERTED = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Sci-Fi": 878,
    "TV Movie": 10770,
    "Thriller": 53,
    "War": 10752,
    "Western": 37
}

import requests
import json
import time
import pandas as pd
import os
import argparse
from tqdm import tqdm

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY") # Remember to add the API_KEY in the .env file

# Configuración de la API

BASE_URL = "https://api.themoviedb.org/3/movie/top_rated"
MAX_PELICULAS = 10000  # Número total de películas a descargar
PELICULAS_POR_PAGINA = 20  # TMDB devuelve 20 películas por página
paginas_a_descargar = (MAX_PELICULAS // PELICULAS_POR_PAGINA) + 1


PATH_SAVE = "peliculas.csv"

import concurrent.futures


def obtener_peliculas(n=MAX_PELICULAS, idioma="es-ES"):
    peliculas = []
    for pagina in tqdm(range(1, paginas_a_descargar + 1), desc="Descargando películas"):
        url = f"{BASE_URL}?api_key={API_KEY}&language={idioma}&page={pagina}"
        respuesta = requests.get(url)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            peliculas.extend(datos["results"])
        else:
            print(f"⚠ Error en la petición: {respuesta.status_code}")
            break

        if len(peliculas) >= n:
            break

    return peliculas[:n]




if __name__=="__main__":
        
    
    # Obtener las películas más populares
    peliculas = obtener_peliculas(MAX_PELICULAS, "en-EN")

    # Crear un DataFrame y guardar en un archivo CSV
    df_peliculas = pd.DataFrame(peliculas)

    df_peliculas.to_csv(PATH_SAVE, index=False, encoding="utf-8")

    print(f"✅ Se han guardado {len(peliculas)} películas en {PATH_SAVE}")