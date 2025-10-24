import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx

# provincias_path = "C:\\Users\\noefa\\Desktop\\Facultad\\2025\\segundo\\Heuristicas\\ProyectoFinal\\provincias\\provincias.shp"


# def agregar_regiones_a_grid(grid, regiones_file, tipo_de_region):
#     nom_por_tipo = {"provincia": "NAM", "dpto": "in1"}
#     nom = nom_por_tipo.get(tipo_de_region)

#     if nom is None:
#         raise ValueError("tipo_de_region debe ser 'provincia' o 'dpto'.")
#     else: 
#         print(f"Agregando {tipo_de_region} usando campo '{nom}' de {regiones_file} a {grid}...")

#     #print("Cargando grilla y regiones...")
#     regiones = gpd.read_file(regiones_file)
#     #print("Columnas del shapefile de regiones:", regiones.columns)

#     # === Asegurar CRS coincidentes ===
#     if grid.crs != regiones.crs:
#         #print("Reproyectando regiones al CRS del grid...")
#         regiones = regiones.to_crs(grid.crs)

#     # Eliminar geometrías nulas, vacías o inválidas
#     if (~regiones.is_valid).sum() > 0:
#         geom_invalida = regiones[~regiones.is_valid]
#         print(geom_invalida)
#         regiones = regiones[regiones.is_valid]
#         print("Geometrías inválidas en regiones:", (~regiones.is_valid).sum())

#     # Hacemos overlay para obtener geometrías de intersección
#     intersec = gpd.overlay(grid, regiones[[nom, 'geometry']], how='intersection', keep_geom_type=False)
#     intersec = intersec[intersec.geometry.type.isin(['Polygon', 'MultiPolygon'])]

#     #print("Reproyectando intersecciones a EPSG:5347 para cálculo de área...")
#     intersec = intersec.to_crs("EPSG:5347")

#     #print("Calculando área de intersección...")
#     intersec['area_inter'] = intersec.geometry.area

#     #print("Seleccionando región dominante por celda...")
    
#     # Asociar cada intersección a una celda original
#     intersec['grid_index'] = intersec.index

#     # Para cada celda, nos quedamos con la región que tiene mayor área de intersección
#     idx_max_area = (
#         intersec.groupby('grid_index')['area_inter']
#         .idxmax()
#     )

#     region_dominante = intersec.loc[idx_max_area, ['grid_index', nom]].set_index('grid_index')

#     # Asignamos el nombre de la región dominante al GeoDataFrame original
#     grid[tipo_de_region] = grid.index.map(region_dominante[nom])


#     return grid

# df = pd.read_excel(
#     "C:\\Users\\noefa\\Desktop\\Facultad\\2025\\segundo\\Heuristicas\\ProyectoFinal\\c2022_tp_gobierno_local_c1.xlsx",
#     sheet_name=1,   # segunda hoja
#     header=3,       # fila 4 → encabezados
#     skiprows=[4]    # salta la fila 5 (la línea divisoria "Total del país" etc.)
# )

# # Shapefile de gobiernos locales
# gdf = gpd.read_file("C:\\Users\\noefa\\Desktop\\Facultad\\2025\\segundo\\Heuristicas\\ProyectoFinal\\municipio\\municipioPolygon.shp")

# print(gdf.head())
# # plt.title("Mapa de Municipios", fontsize=14)
# # plt.show()

# # Index(['gid', 'fna', 'gna', 'nam', 'in1', 'fdc', 'sag', 'geometry'], dtype='object')

# df.rename(columns=lambda x: x.strip(), inplace=True)
# gdf.rename(columns=lambda x: x.strip(), inplace=True)

# df["Código de gobierno local"] = pd.to_numeric(df["Código de gobierno local"], errors="coerce")
# gdf["in1"] = pd.to_numeric(gdf["in1"], errors="coerce")

# df = df.dropna(subset=["Código de gobierno local"])
# gdf = gdf.dropna(subset=["in1"])

# print("Cantidad de gobiernos locales:", len(gdf))

# gdf_merged = gdf.merge(
#     df,
#     how="inner",  
#     left_on="in1",
#     right_on="Código de gobierno local"
# )

# column_map = {
#     "gid": "gid",
#     "fna": "fna",
#     "gna": "gna",
#     "nam": "nam",
#     "in1": "in1",
#     "fdc": "fdc",
#     "sag": "sag",
#     "geometry": "geometry",
#     "Código de jurisdicción": "cod_jurisdiccion",
#     "Jurisdicción": "jurisdiccion",
#     "Código de gobierno local": "codigo_gob_local",
#     "Categoría": "categoria",
#     "Gobierno local": "gob_local",
#     "Viviendas": "viviendas",
#     "Población": "poblacion",
#     "Viviendas\nparticulares": "viv_part",
#     "Población en\nviviendas\nparticulares": "p_viv_part",
#     "Viviendas\ncolectivas": "viv_colec",
#     "Población\nen\nviviendas\ncolectivas": "p_viv_colec",
#     "Población\nen situación\nde calle": "p_calle"
# }

# gdf_merged = gdf_merged.rename(columns=column_map). drop(columns=["gid", "fna", "gna", "nam", "in1", "fdc", "sag"])

# print("Columnas del dataset final:", gdf_merged.columns)
# print(f"Cantidad de gobiernos locales en el dataset final: {len(gdf_merged)}")
# print(f"Cantidad de gobiernos locales perdidos: {len(gdf) - len(gdf_merged)}")

# # gdf_merged = agregar_regiones_a_grid(
# #     gdf_merged,
# #     provincias_path,
# #     tipo_de_region="provincia"
# # )

# print("provincias representadas:", gdf_merged['jurisdiccion'].unique())

# gdf_merged.columns = gdf_merged.columns.str.slice(0, 10)
# print ("Columnas del dataset final (cortadas a 10 caracteres):", gdf_merged.columns)

#gdf_merged.to_file("C:\\Users\\noefa\\Desktop\\Facultad\\2025\\segundo\\Heuristicas\\ProyectoFinal\\base_completa.shp")

# Buenos Aires, Santa Fe, La pampa
gdf_merged = gpd.read_file("C:\\Users\\noefa\\Desktop\\Facultad\\2025\\segundo\\Heuristicas\\ProyectoFinal\\base_completa.shp")
provincias_path = "C:\\Users\\noefa\\Desktop\\Facultad\\2025\\segundo\\Heuristicas\\ProyectoFinal\\provincias\\provincias.shp"
provincias = gpd.read_file(provincias_path)
print(provincias["NAM"].unique())

while True:
    nombre_provincia = input("Ingrese el nombre de la provincia que desea visualizar (o 'salir' para terminar): ").strip()

    if nombre_provincia.lower() == 'salir':
        break
    provincia_seleccionada = provincias[provincias['NAM'].str.lower() == nombre_provincia.lower()]
    if provincia_seleccionada.empty:
        print(f"No se encontró la provincia '{nombre_provincia}'. Intente nuevamente.")
        continue

    # --------------- Graficar Provincia y Gobiernos Locales -------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    provincia_seleccionada.plot(ax=ax, color='lightblue', edgecolor='black')
    gdf_provincia = gdf_merged[gdf_merged['jurisdicci'].str.lower() == nombre_provincia.lower()]
    gdf_provincia.plot(ax=ax, color='orange', edgecolor='red', alpha=0.5)
    plt.title(f"Provincia de {nombre_provincia}", fontsize=16)
    plt.show()

    # --------------- Guardar Shapefile de Gobiernos Locales -------------------------------
    guardar = input(f"¿Desea guardar el shapefile de los gobiernos locales de {nombre_provincia}? (s/n): ").strip().lower()
    if guardar == 's':
        if gdf_provincia.empty:
            print(f"No se encontraron gobiernos locales para la provincia '{nombre_provincia}'.")
            continue
        output_path = f"C:\\Users\\noefa\\Desktop\\Facultad\\2025\\segundo\\Heuristicas\\ProyectoFinal\\base_{nombre_provincia.lower().replace(' ', '_')}.shp"
        gdf_provincia.to_file(output_path)
        print(f"Shapefile de gobiernos locales de {nombre_provincia} guardado en: {output_path}")

    # --------------- Generar Grafo -------------------------------

    gdf = gdf_provincia

    col_nodo = "codigo_gob"
    assert col_nodo in gdf.columns, f"No existe la columna '{col_nodo}' en el shapefile."

    G = nx.Graph()
    for _, row in gdf.iterrows():
        print(row[col_nodo])
        G.add_node(row[col_nodo])

    gdf["geometry"] = gdf["geometry"].buffer(0)

    # --------------- Calcular Adyacencias -------------------------------
    for i, geom_i in enumerate(gdf.geometry):
        posibles_vecinos = list(gdf.sindex.intersection(geom_i.bounds))
        
        for j in posibles_vecinos:
            # Usamos índice posicional con iloc
            geom_j = gdf.iloc[j].geometry if j < len(gdf) else None
            if geom_j is None or i >= j:
                continue

            if geom_i.touches(geom_j):  # comparten borde
                G.add_edge(gdf.iloc[i][col_nodo], gdf.iloc[j][col_nodo])


    # --------------- Visualizar Grafo -------------------------------
    gdf["centroid"] = gdf.geometry.centroid
    pos = {row[col_nodo]: (row["centroid"].x, row["centroid"].y) for _, row in gdf.iterrows()}

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, facecolor="lightgrey", edgecolor="white")

    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="red", linewidth=0.8, alpha=0.6)

    ax.scatter(
        [p[0] for p in pos.values()],
        [p[1] for p in pos.values()],
        color="blue",
        s=10,
        zorder=5
    )

    ax.set_title("Grafo de adyacencia entre gobiernos locales", fontsize=14)
    ax.axis("off")
    plt.show()

    guardar = input(f"¿Desea guardar la matriz de adyacencia de los gobiernos locales de {nombre_provincia}? (s/n): ").strip().lower()
    if guardar == 's':
        adj_matrix = nx.to_pandas_adjacency(G, dtype=int)
        adj_matrix.to_csv(f"matriz_adyacencia_{nombre_provincia.lower()}.csv")
