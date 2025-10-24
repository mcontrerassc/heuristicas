import geopandas as gpd
import matplotlib.pyplot as plt

# Leer el shapefile resultante
gdf = gpd.read_file(
    r"C:\Users\noefa\Desktop\Facultad\2025\segundo\Heuristicas\ProyectoFinal\base_la_pampa.shp",
    encoding='latin1'  # <- especifica la codificación correcta
)
print(gdf.head())
# Graficar (básico)
gdf.plot(edgecolor="black", facecolor="lightblue", figsize=(10, 8))
plt.title("Mapa base_completa.shp", fontsize=14)
plt.show()
