import pandas as pd
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from typing import List
from typing import Union
import psutil
from functools import wraps

# Cargar el dataset "peliculas"
peliculas = pd.read_csv('datasets/peliculas.csv')

# Reemplazar valores NaN por cadenas vacías en las columnas relevantes
peliculas['franquicia'].fillna('', inplace=True)
peliculas['genres'].fillna('', inplace=True)
peliculas['idioma'].fillna('', inplace=True)
peliculas['eslogan'].fillna('', inplace=True)

# Concatenamos las características relevantes en un solo campo para el análisis TF-IDF
peliculas['combined_features'] = peliculas['franquicia'] + ' ' + peliculas['genres']

# Creamos una matriz TF-IDF con las características de todas las películas
tfidf = TfidfVectorizer(stop_words='english')
features_matrix = tfidf.fit_transform(peliculas['combined_features'].fillna(''))

# Calcular la similitud de coseno entre las características de todas las películas
cosine_sim = linear_kernel(features_matrix, features_matrix)

indices = pd.Series(peliculas.index, index=peliculas['title']).drop_duplicates()

# Crear la instancia de FastAPI
app = FastAPI()

@app.get('/recomendacion/caracteristicas/{titulo}', response_model=Union[List[str], dict])
def recommend_movies_by_features(titulo: str):
    try:
        idx = indices[indices.index == titulo].iloc[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Eliminar puntuaciones de baja similitud para reducir el uso de memoria
        sim_scores = [item for item in sim_scores if item[1] > 0.1]
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        movie_indices = [i[0] for i in sim_scores]
        movie_titles = peliculas.iloc[movie_indices]['title'].tolist()
        return movie_titles
    except KeyError:
        return {'message': f'No se encuentra la película "{titulo}" en el conjunto de datos'}
    except IndexError:
        return {'message': f'No se pudo obtener la recomendación para la película "{titulo}"'}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)