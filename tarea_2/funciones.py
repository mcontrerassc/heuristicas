import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Point
import matplotlib.pyplot as plt


gdf = gpd.read_file("base_buenos_aires.shp")

if gdf.crs.is_geographic:
    gdf = gdf.to_crs(epsg=5347) 

gdf["codigo_gob"] = gdf["codigo_gob"].fillna(0).astype(int)

# print(gdf.columns) 
# ['cod_jurisd', 'jurisdicci', 'codigo_gob', 'categoria', 'gob_local',
# 'viviendas', 'poblacion', 'viv_part', 'p_viv_part', 'viv_colec',
# 'p_viv_cole', 'p_calle', 'geometry']
#print(gdf.head())

def polsby_popper(gdf, codigos_gob, plot=False):
    subset = gdf[gdf["codigo_gob"].isin(codigos_gob)]

    if subset.empty:
        print("No se encontraron geometrías para los códigos indicados.")
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
        print("⚠️ No se encontraron geometrías para los códigos indicados.")
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


# ------------ Ejemplo ------------
lista_codigos = [60756, 60805]

# reock = reock_score(gdf, lista_codigos, plot=True)
# print("Índice Reock:", reock)

# reock = polsby_popper(gdf, lista_codigos, plot=True)
# print("Índice Reock:", reock)

