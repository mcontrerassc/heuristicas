import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Point
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import math 

# ------------ Abrir SHP con datos de Población y Hogares  ------------

gdf = gpd.read_file("base_buenos_aires.shp")

if gdf.crs.is_geographic:
    gdf = gdf.to_crs(epsg=5347) 

gdf["codigo_gob"] = gdf["codigo_gob"].fillna(0).astype(int)

# print(gdf.columns) 
# ['cod_jurisd', 'jurisdicci', 'codigo_gob', 'categoria', 'gob_local',
# 'viviendas', 'poblacion', 'viv_part', 'p_viv_part', 'viv_colec',
# 'p_viv_cole', 'p_calle', 'geometry']

# ------------ Abrir Matriz de Adyacencia ------------

matriz_adyacencia = pd.read_csv("matriz_adyacencia_buenos aires.csv", index_col=0)

matriz_adyacencia.index = matriz_adyacencia.index.astype(float).astype(int)
matriz_adyacencia.columns = matriz_adyacencia.columns.astype(float).astype(int)

G = nx.from_pandas_adjacency(matriz_adyacencia)

# print(G)
# print(list(G.nodes)[:10])

# ------------ Funciones ------------

def polsby_popper(gdf, codigos_gob, plot=False):
    subset = gdf[gdf["codigo_gob"].isin(codigos_gob)]

    if subset.empty:
        return np.nan

    geom_union = unary_union(subset.geometry).buffer(0)

    if geom_union.is_empty or geom_union.area == 0:
        return np.nan

    area = geom_union.area
    perimetro = geom_union.length

    if plot: 

        fig, ax = plt.subplots(figsize=(8, 8))

        gdf.plot(ax=ax, color='lightgrey', edgecolor='white')

        subset.plot(ax=ax, color='skyblue', edgecolor='black')

        subset.boundary.plot(ax=ax, color='red', linewidth=2)
        
        ax.set_title(f"Distrito con códigos: {lista_codigos}")
        ax.legend()

        plt.show()

    return (4 * np.pi * area) / (perimetro ** 2)

def reock_score(gdf, codigos_gob, plot=False):
    subset = gdf[gdf["codigo_gob"].isin(codigos_gob)]

    if subset.empty:
        return np.nan

    geom_union = unary_union(subset.geometry).buffer(0)

    if geom_union.is_empty or geom_union.area == 0:
        return np.nan

    centroid = geom_union.centroid
    polygons = [geom_union] if geom_union.geom_type == "Polygon" else geom_union.geoms
    max_dist = max(
        max(centroid.distance(Point(c)) for c in poly.exterior.coords)
        for poly in polygons
    )

    area_circle = np.pi * max_dist**2
    area = geom_union.area

    if plot: 
        plot_reock(gdf, lista_codigos)

    return area / area_circle if area_circle > 0 else np.nan

def plot_reock(gdf, lista_codigos): 
    subset = gdf[gdf["codigo_gob"].isin(lista_codigos)]
    geom_union = unary_union(subset.geometry).buffer(0)
    
    centroid = geom_union.centroid
    
    polygons = [geom_union] if geom_union.geom_type == "Polygon" else geom_union.geoms
    max_dist = max(
        max(centroid.distance(Point(c)) for c in poly.exterior.coords)
        for poly in polygons
    )

    circle_reock = Point(centroid.x, centroid.y).buffer(max_dist)

    fig, ax = plt.subplots(figsize=(8, 8))

    gdf.plot(ax=ax, color='lightgrey', edgecolor='white')

    subset.plot(ax=ax, color='skyblue', edgecolor='black')
    
    ax.plot(centroid.x, centroid.y, 'ro', markersize=8, label='Centroide')
    
    gpd.GeoSeries(circle_reock).boundary.plot(ax=ax, color='red', linestyle='--', label='Círculo Reock')
    
    ax.set_title(f"Distrito con códigos: {lista_codigos}")
    ax.legend()

    plt.show()

def es_adyacente_a_lista(G, código, codigos_gob): 
    for n in codigos_gob:
        if G.has_edge(código, n):
            return True
    return False

def desviacion(gdf, lista_codigos, variable):
    poblacion = gdf[gdf['codigo_gob'].isin(lista_codigos)][variable].sum()

    media = gdf[variable].sum() / len(gdf)

    desviacion = (abs(poblacion - media)/media)**2
    return desviacion

def desviacion_total(gdf, lista_distrito, variable='poblacion'): 
    desv_tot = 0
    for lista_codigos in lista_distrito: 
        desv_tot += desviacion(gdf, lista_codigos, variable)
    
    return math.sqrt(desv_tot)


# ------------ Ejemplo ------------
lista_codigos = [60756, 60805]

# reock = reock_score(gdf, lista_codigos, plot=True)
# print("Índice Reock:", reock)

# reock = polsby_popper(gdf, lista_codigos, plot=True)
# print("Índice Reock:", reock)

# print(es_adyacente_a_lista(G, 60749, lista_codigos)) # debe dar true 
# print(es_adyacente_a_lista(G, 60616, lista_codigos)) # debe dar false

# print(desviacion(gdf, lista_codigos))
