import geopandas as gpd
import requests
from PIL import Image as PILImage
from PIL import ImageChops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from IPython.display import Image as IPyImage
import matplotlib.animation as animation
from scipy.ndimage import label
from matplotlib import cm
import matplotlib.lines as mlines
from pyproj import Transformer
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from folium import Map, Rectangle, LayerControl
from folium.raster_layers import ImageOverlay


# FUNKCJE DO WYKRESÓW
def calculate_land_use_trend(class_percentage_per_year, area_name, show_plot=True,show_percent=True):
    trend = []

    for year, counts in class_percentage_per_year.items():
        row = {"Year": year}
        for class_label, percent in counts.items():
            row[class_label] = percent
        trend.append(row)

    trend_df = pd.DataFrame(trend).set_index("Year").fillna(0).sort_index(axis=1)

    if show_plot:
        ax = trend_df.plot(kind='bar', figsize=(12, 6), colormap='viridis')

        # Dodaj wartości liczbowo nad każdym 
        if show_percent:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', fontsize=8, label_type='edge',rotation=45)

        plt.title(f"Zmiany użytkowania terenu w {area_name.replace('_', ' ').title()}")
        plt.xlabel("Rok")
        plt.ylabel("Udział (%)")
        plt.xticks(rotation=45)
        plt.legend(title="Typ terenu", bbox_to_anchor=(1.05, 1), loc='upper left')
        #plt.tight_layout()
        ax.grid(True)
        plt.show()
    


    return trend_df

def PrintImageForYear(year,images_dict,legend_dict_per_year,area_name,level_name,ax_map=None,ax_legend=None):
    if ((ax_map == None) or (ax_legend == None)):
        fig, (ax_map, ax_legend) = plt.subplots(ncols=2, figsize=(10, 5),gridspec_kw={'width_ratios': [3, 1]})
    pil_img = images_dict[year]
    year_legend = legend_dict_per_year[year]

    # Mapa
    ax_map.imshow(pil_img)
    ax_map.axis("off")
    title = area_name.replace("_", " ").title()
    ax_map.set_title(f"Mapa użytkowania terenu poziom {level_name} - {title} ({year})")

    # Legenda
    ax_legend.axis("off")
    ax_legend.set_title("Legenda", fontsize=12, fontweight='bold')

    legend_patches = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markersize=10,
            markerfacecolor=np.array(rgb) / 255.0,
            label=label
        )
        for rgb, label in year_legend.items()
    ]

    ax_legend.legend(
        handles=legend_patches,
        loc="upper left",
        fontsize=9,
        frameon=False
    )


def MapAnimationByYears(images_dict, legend_dict_per_year, area_name, level_name, save_as_gif=True, fps=1):
    years = list(images_dict.keys())
    fig, (ax_map, ax_legend) = plt.subplots(ncols=2, figsize=(10, 5),gridspec_kw={'width_ratios': [3, 1]})

    def update(frame):
        ax_map.clear()
        ax_legend.clear()
        year = years[frame]
        PrintImageForYear(year,images_dict,legend_dict_per_year,area_name,level_name,ax_map=ax_map,ax_legend=ax_legend)

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(years),
        interval=int(1000 / fps),
        repeat=True,
        blit=False
    )

    gif_filename = f"animacja_mapy_{level_name}.gif"
    if save_as_gif:
        ani.save(gif_filename, writer="pillow", fps=fps)
        plt.close(fig)
        return IPyImage(filename=gif_filename)
    else:
        return ani
    
def seperate_classes (level_classes, years, dfs,class_lvl,area_name, color_dict):
    results = []

    for level_type in level_classes:
        for year in years:
            df = dfs[year]

            height = df['y'].max() + 1
            width = df['x'].max() + 1

            mask = np.zeros((height, width), dtype=np.uint8)
            x = df['x'].to_numpy()
            y = df['y'].to_numpy()
            classes = df[class_lvl].to_numpy()
            mask[y, x] = (classes == level_type).astype(np.uint8)

            labeled, num_features = label(mask)
            results.append({
                "year": year,
                "class": level_type,
                "count": num_features
            })

    data = pd.DataFrame(results)
    df_islands_pivot = data.pivot(index='year', columns='class', values='count')

    # Tworzenie mapy kolorów
    class_to_rgb = {v: k for k, v in color_dict.items()}
    colors = [tuple(x / 255 for x in class_to_rgb.get(level_type, (0, 0, 0))) for level_type in df_islands_pivot.columns]

    df_islands_pivot.plot(kind='line', figsize=(10, 6), marker='o', color=colors)
    plt.title(f"Liczba wysp w {area_name.replace('_', ' ').title()} w latach {years[0]}-{years[-1]}")
    plt.xlabel("Rok")
    plt.ylabel("Liczba wysp")
    plt.xticks(years, rotation=45)
    plt.legend(title="Typ terenu", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    
def plot_multiple_land_use_change_areas( dfs_dict, bbox_dict, years, class_column="class_lvl_3", min_changes=2, cmap_name="brg" ):

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # Wycentrowanie mapy na pierwsze miasto
    first_city = list(bbox_dict.keys())[0]
    x0, y0, x1, y1 = bbox_dict[first_city]
    lon0, lat0 = transformer.transform(x0, y0)
    lon1, lat1 = transformer.transform(x1, y1)
    center_lat = (lat0 + lat1) / 2
    center_lon = (lon0 + lon1) / 2

    m = Map(location=[center_lat, center_lon], zoom_start=6)

    for city, dfs in dfs_dict.items():
        bbox = bbox_dict[city]
        x_min, y_min, x_max, y_max = bbox
        h, w = dfs[years[0]]['y'].max()+1, dfs[years[0]]['x'].max()+1
        count = np.zeros((h, w), dtype=int)
        for i in range(len(years)-1):
            df1, df2 = dfs[years[i]], dfs[years[i+1]]
            changed = df1[class_column].values != df2[class_column].values
            count[df1.loc[changed, 'y'], df1.loc[changed, 'x']] += 1
        masked = np.where(count >= min_changes, count, np.nan)

        # Heatmapa dla danych 
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad((1, 1, 1, 0))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(masked, cmap=cmap, interpolation='nearest', vmin=min_changes)
        ax.axis('off')
        plt.tight_layout()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name, transparent=True, bbox_inches='tight', pad_inches=0)
            image_path = tmp.name
        plt.close(fig)

        # Transformacja BBOX
        lon_min, lat_min = transformer.transform(x_min, y_min)
        lon_max, lat_max = transformer.transform(x_max, y_max)

        # Rectangle + Overlay
        Rectangle(bounds=[(lat_min, lon_min), (lat_max, lon_max)], color="blue",weight=0.5, fill=True, fill_opacity=0.1, tooltip=city).add_to(m)

        ImageOverlay(
            name=f"Heatmap: {city}",
            image=image_path,
            bounds=[(lat_min, lon_min), (lat_max, lon_max)],
            opacity=1,
            zindex=1,
        ).add_to(m)

    LayerControl().add_to(m)
    return m

def plot_cities_land_use_change_map( city_names, city_objects, bbox_source,  years, class_column="class_lvl_3", min_changes=1,cmap_name="brg"):
    
    dfs_dict = {city: city_objects[city].dfs for city in city_names}
    bbox_dict = {city: bbox_source[city] for city in city_names}
    
    return plot_multiple_land_use_change_areas(
        dfs_dict=dfs_dict,
        bbox_dict=bbox_dict,
        years=years,
        class_column=class_column,
        min_changes=min_changes,
        cmap_name=cmap_name
    )

def plot_land_use_change_map(dfs, years, class_column="class_lvl_1", cmap_name="tab20"):
    
    # Przygotowanie danych
    base = dfs[years[0]].loc[:, ['x', 'y', class_column]].copy()
    base.columns = ['x', 'y', 'class_base']
    base['class_mid'] = dfs[years[1]][class_column].values
    base['class_final'] = dfs[years[2]][class_column].values

    base['join'] = base['class_base'] + ' -> ' + base['class_final']
    base['join6'] = base['class_base'] + ' -> ' + base['class_mid']
    base['join18'] = base['class_mid'] + ' -> ' + base['class_final']

    filters = {
        f'{years[0]} → {years[2]}': base[base['class_base'] != base['class_final']].copy(),
        f'{years[0]} → {years[1]}': base[base['class_base'] != base['class_mid']].copy(),
        f'{years[1]} → {years[2]}': base[base['class_mid'] != base['class_final']].copy()
    }

    # Kolory
    unique_values = pd.unique(pd.concat([df[key] for key, df in zip(['join', 'join6', 'join18'], filters.values())]))
    n = len(unique_values)
    cmap = cm.get_cmap(cmap_name, n)
    color_map = {val: tuple((np.array(cmap(i)[:3]) * 255).astype(int)) for i, val in enumerate(unique_values)}

    def make_image(df, join_col):
        df['color'] = df[join_col].map(color_map)
        width = df['x'].max() + 1
        height = df['y'].max() + 1
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[df['y'], df['x']] = np.stack(df['color'])
        return PILImage.fromarray(img, mode='RGB')

    # Generowanie obrazów dla każdej zmiany
    titles = list(filters.keys())
    joins = ['join', 'join6', 'join18']
    images = [make_image(df, join) for df, join in zip(filters.values(), joins)]

    # Rysowanie obrazów
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    for ax, title, img in zip(axs.flat[:3], titles, images):
        ax.imshow(img)
        ax.set_title(f"Zmiana: {title}")
        ax.axis("off")

    # Tworzenie legendy
    ax_legend = axs[1, 1]
    ax_legend.axis("off")
    ax_legend.set_title("Legenda", fontsize=12, fontweight='bold')

    legend_patches = [
        mlines.Line2D([0], [0], marker='o', color='w', markersize=10,
                      markerfacecolor=np.array(color_map[label]) / 255.0, label=label)
        for label in unique_values
    ]
    ax_legend.legend(handles=legend_patches, loc="upper left", fontsize=8, frameon=False)

    plt.tight_layout()
    plt.show()

def plot_land_use_change_matrix_matplotlib(dfs, year_from, year_to, class_column='class_lvl_1', figsize=(10, 6), cmap="Blues"):
    df1 = dfs[year_from][class_column]
    df2 = dfs[year_to][class_column]

    change_df = pd.DataFrame({f"{year_from}": df1, f"{year_to}": df2})
    change_matrix = pd.crosstab(change_df[f"{year_from}"], change_df[f"{year_to}"]).astype(float)

    # Normalizacja wierszy do procentów
    change_matrix = change_matrix.div(change_matrix.sum(axis=1), axis=0) * 100

    row_labels = change_matrix.index.tolist()
    col_labels = change_matrix.columns.tolist()
    data = change_matrix.values

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(data, cmap=cmap)

    # Etykiety osi
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    # Adnotacje
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = f"{data[i, j]:.1f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

    ax.set_title(f"Zmiany użytkowania terenu: {year_from} → {year_to}")
    ax.set_xlabel(f"Nowa klasa ({year_to})")
    ax.set_ylabel(f"Stara klasa ({year_from})")

    fig.colorbar(cax, ax=ax, label="% zmiany")
    plt.tight_layout()
    plt.show()


def calculate_land_use_trend_for_cities(class_percentage_per_city, area_names, show_plot=True):
   
    first_city_data = class_percentage_per_city[0]
    first_year = sorted(first_city_data.keys())[0]
    classes = sorted(first_city_data[first_year].index)
    
    city_dfs = []
    for city_data in class_percentage_per_city:
        trend = []
        for year, series in city_data.items():
            row = {"Year": year}
            row.update(series.to_dict())
            trend.append(row)
        df = pd.DataFrame(trend).set_index("Year").fillna(0).sort_index()
        city_dfs.append(df)
    
    if show_plot:
        fig, axs = plt.subplots(len(classes), 1, figsize=(10, 4*len(classes)), sharex=True)
        if len(classes) == 1:
            axs = [axs]
        
        for i, class_label in enumerate(classes):
            ax = axs[i]
            for df, area_name in zip(city_dfs, area_names):
                if class_label in df.columns:
                    ax.plot(df.index, df[class_label], marker='o', label=area_name)
            ax.set_title(f"Zmiana udziału klasy: {class_label}")
            ax.set_ylabel("Udział (%)")
            ax.grid(True)
            ax.legend()
        
        axs[-1].set_xlabel("Rok")
        plt.suptitle("Trendy użytkowania terenu w 3 miastach")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
