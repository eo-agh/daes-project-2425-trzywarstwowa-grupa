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

level_3_legend = {
    (230, 0, 77): "111 - Zwarta zabudowa miejska",
    (255, 0, 0): "112 - Rozproszona zabudowa miejska",
    (204, 77, 242): "121 - Obszary przemysłowe i handlowe",
    (204, 0, 0): "122 - Sieci drogowe i kolejowe oraz powiązane tereny",
    (230, 204, 204): "123 - Obszary portowe",
    (230, 204, 230): "124 - Lotniska",
    (166, 0, 204): "131 - Tereny wydobycia surowców",
    (166, 77, 0): "132 - Wysypiska",
    (255, 77, 255): "133 - Place budowy",
    (255, 166, 255): "141 - Zielone obszary miejskie",
    (255, 230, 255): "142 - Obiekty sportowe i rekreacyjne",
    (255, 255, 168): "211 - Grunty orne nieirygowane",
    (255, 255, 0): "212 - Grunty orne nawadniane",
    (230, 230, 0): "213 - Pola ryżowe",
    (230, 128, 0): "221 - Winnice",
    (242, 166, 77): "222 - Sady i plantacje owoców jagodowych",
    (230, 166, 0): "223 - Gaje oliwne",
    (230, 230, 77): "231 - Pastwiska",
    (255, 230, 166): "241 - Uprawy jednoroczne powiązane z uprawami trwałymi",
    (255, 230, 77): "242 - Złożone wzorce upraw",
    (230, 204, 77): "243 - Obszary rolnicze z naturalną roślinnością",
    (242, 204, 166): "244 - Obszary agro-leśne",
    (128, 255, 0): "311 - Lasy liściaste",
    (0, 166, 0): "312 - Lasy iglaste",
    (77, 255, 0): "313 - Lasy mieszane",
    (204, 242, 77): "321 - Naturalne trawiaste obszary",
    (166, 255, 128): "322 - Wrzośce i wrzosowiska",
    (166, 230, 77): "323 - Roślinność twardolistna",
    (166, 242, 0): "324 - Przejściowe formy lasu i zarośli",
    (230, 230, 230): "331 - Plaże, wydmy, piaski",
    (204, 204, 204): "332 - Skały lite",
    (204, 255, 204): "333 - Obszary słabo roślinne",
    (0, 0, 0): "334 - Obszary spalone",
    (166, 230, 204): "335 - Lodowce i wieczne śniegi",
    (166, 166, 255): "411 - Bagna śródlądowe",
    (77, 77, 255): "412 - Torfowiska",
    (204, 204, 255): "421 - Słone bagna",
    (230, 230, 255): "422 - Saliny",
    (166, 166, 230): "423 - Płaskie wybrzeża zalewowe",
    (0, 204, 242): "511 - Rzeki",
    (128, 242, 230): "512 - Zbiorniki wodne",
    (0, 255, 166): "521 - Laguny przybrzeżne",
    (166, 255, 230): "522 - Estuaria",
    (230, 242, 255): "523 - Morza i oceany",
    (230, 242, 255): "000 - Brak danych"
}

level_3_legend_rgb = {v.split(" - ")[0]: k for k, v in level_3_legend.items()}

level_2_legend = {
    (230, 0, 77): "11 - Zabudowa miejska",
    (204, 77, 242): "12 - Obiekty przemysłowe, handlowe i transportowe",
    (166, 0, 204): "13 - Tereny górnicze, hałdy i place budowy",
    (255, 166, 255): "14 - Sztuczna roślinność nie-rolnicza",
    (255, 255, 168): "21 - Grunty orne",
    (230, 128, 0): "22 - Uprawy trwałe",
    (230, 230, 77): "23 - Użytki zielone",
    (255, 230, 166): "24 - Złożone obszary rolnicze",
    (128, 255, 0): "31 - Lasy",
    (204, 242, 77): "32 - Zarośla i roślinność trawiasta",
    (230, 230, 230): "33 - Tereny odkryte z małą lub żadną roślinnością",
    (166, 166, 255): "41 - Śródlądowe obszary podmokłe",
    (204, 204, 255): "42 - Nadmorskie obszary podmokłe",
    (0, 204, 242): "51 - Wody śródlądowe",
    (0, 255, 166): "52 - Wody morskie",
    (230, 242, 255): "00 - Brak danych"
}

level_2_legend_rgb = {v.split(" - ")[0]: k for k, v in level_2_legend.items()}
level_2_legend_desc = {v.split(" - ")[0]: v for v in level_2_legend.values()}

level_1_legend = {
    (230, 0, 77): "1 - Sztuczne powierzchnie",
    (255, 255, 168): "2 - Użytki rolne",
    (128, 255, 0): "3 - Lasy i tereny półnaturalne",
    (166, 166, 255): "4 - Obszary podmokłe",
    (0, 204, 242): "5 - Ciała wodne",
    (230, 242, 255): "0 - Brak danych"
}

level_1_legend_rgb = {v.split(" - ")[0]: k for k, v in level_1_legend.items()}
level_1_legend_desc = {v.split(" - ")[0]: v for v in level_1_legend.values()}

POLSKA_OBSZARY_BBOX = {
    'Warszawa': [2310981, 6808884, 2378018, 6875921],
    'Gdańsk': [2035020, 7200851, 2102057, 7267889],
    'Wrocław': [1861305, 6606825, 1928343, 6673863],
    'Poznań': [1851565, 6840188, 1918603, 6907225],
    'Rzeszów': [2412727, 6417097, 2479764, 6484135],
    'Lublin': [2478595, 6632117, 2545632, 6699154],
    'Szczecin': [1586491, 7029137, 1653528, 7096174],
    'Katowice': [2084613, 6458760, 2151650, 6525797],
    'Białystok': [2545620, 6974026, 2612657, 7041063],
    'Łódź': [2132313, 6723183, 2199350, 6790220]
}

EUROPA_OBSZARY_BBOX = {
    'Berlin': [1460889, 6858493, 1527927, 6925530],
    'Paryż': [226634, 6217370, 293672, 6284407],
    'Rzym': [1354356, 5112504, 1421394, 5179542],
    'Amsterdam': [512336, 6830972, 579373, 6898010],
    'Barcelona': [205595, 5036896, 272633, 5103934],
    'Praga': [1571797, 6428011, 1638834, 6495048],
    'Wiedeń': [1789204, 6108043, 1856241, 6175080],
    'Oslo': [1163410, 8347075, 1230447, 8414112],
    'Madryt': [-445823, 4893178, -378786, 4960215],
    'Lizbona': [-1051279, 4650536, -984242, 4717573],
    'Kopenhaga': [1365578, 7460686, 1432615, 7527723],
    'Bruksela': [450910, 6561337, 517947, 6628374]
}

def print_available_bounding_boxes():
    print("EUROPA_OBSZARY_BBOX")
    for city, _ in EUROPA_OBSZARY_BBOX.items():
        print(city)
    print()
    print("POLSKA_OBSZARY_BBOX")
    for city, _ in POLSKA_OBSZARY_BBOX.items():
        print(city)

class AreaManager:
    def __init__(self, area_name: str, api_url: str = "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC{}_WM/MapServer/export", years: list = [1990,2000,2006,2012,2018]):
        self.area_name = area_name
        if area_name in EUROPA_OBSZARY_BBOX.keys():
            bbox_values = EUROPA_OBSZARY_BBOX[area_name]
        elif area_name in POLSKA_OBSZARY_BBOX.keys():
            bbox_values = POLSKA_OBSZARY_BBOX[area_name]
        else:
            print("Niepoprawny obszar! Nie da się utworzyć klasy. Sprawdź dostępne obszary przy użyciu print_available_bounding_boxes()")
            return
        bbox = f"{bbox_values[0]},{bbox_values[1]},{bbox_values[2]},{bbox_values[3]}"
        width_meters = bbox_values[2] - bbox_values[0]
        height_meters = bbox_values[3] - bbox_values[1]
        aspect_ratio = width_meters / height_meters

        # Ustalmy bazową szerokość - jeśli bounding box nie jest kwadratem, to trzeba dopasować odpowiednio rozdzielczość
        base_width = 2048
        base_height = int(base_width / aspect_ratio) 
                        
        params = {
            "bbox": bbox,
            "bboxSR": "102100",     # EPSG:3857 (Web Mercator)
            "imageSR": "102100",
            "size": f"{base_width},{base_height}",    # rozdzielczość
            "format": "png",        # 'jpg', 'tiff'
            "f": "image"
        }
        
        self.images_lvl_3 = {}

        # ściągamy plik dla każdego dostępnego roku
        for year in years:
            print(api_url.format(year))
            response = requests.get(api_url.format(year), params=params)
            print(f"Obraz dla roku {year} wczytany")
            image = PILImage.open(BytesIO(response.content))
            self.images_lvl_3[year] = image
        print("Pobrano wszystkie obrazy.")
    
    def create_analytical_df(self):
        self.dfs = {}  

        for year, img in self.images_lvl_3.items():
            img = img.convert("RGB")
            img_array = np.array(img)

            height, width, _ = img_array.shape

            pixels = img_array.reshape(-1, 3)
            rows = np.repeat(np.arange(height), width)
            cols = np.tile(np.arange(width), height)

            df = pd.DataFrame({
                'x': cols,
                'y': rows,
                'rgb': list(zip(pixels[:, 0], pixels[:, 1], pixels[:, 2]))
            })

            self.dfs[year] = df  
            print(f"DataFrame dla roku {year} utworzony.")
        print(f"Obliczono wszystkie DataFrame.")
    
    def count_classes(self):
        self.class_lvl_3_counts_per_year = {}
        self.class_lvl_2_counts_per_year = {}
        self.class_lvl_1_counts_per_year = {}

        for year, df in self.dfs.items():
            # lvl 3
            df["class_lvl_3"] = df["rgb"].map(level_3_legend).fillna("000 - Brak danych")

            # lvl 2
            df["class_lvl_2"] = df["class_lvl_3"].str[:2]
            df["rgb_lvl_2"] = df["class_lvl_2"].map(level_2_legend_rgb)
            df["class_lvl_2"] = df["class_lvl_2"].map(level_2_legend_desc)

            # lvl 1
            df["class_lvl_1"] = df["class_lvl_3"].str[0]
            df["rgb_lvl_1"] = df["class_lvl_1"].map(level_1_legend_rgb)
            df["class_lvl_1"] = df["class_lvl_1"].map(level_1_legend_desc)
            
            class_lvl_3_counts = df["class_lvl_3"].value_counts()
            class_lvl_2_counts = df["class_lvl_2"].value_counts()
            class_lvl_1_counts = df["class_lvl_1"].value_counts()

            self.class_lvl_3_counts_per_year[year] = class_lvl_3_counts
            self.class_lvl_2_counts_per_year[year] = class_lvl_2_counts
            self.class_lvl_1_counts_per_year[year] = class_lvl_1_counts

            print(f"Zliczenia klas dla roku {year} utworzone.")
        print(f"Obliczono zliczenia dla każdego roku.")

    def calculate_percentage_of_classes(self):
        self.class_lvl_3_percentage_per_year = {}
        self.class_lvl_2_percentage_per_year = {}
        self.class_lvl_1_percentage_per_year = {}

        for year, _ in self.dfs.items():   
            total = self.class_lvl_3_counts_per_year[year].sum()
            self.class_lvl_3_percentage_per_year[year] = (self.class_lvl_3_counts_per_year[year] / total) * 100

            total = self.class_lvl_2_counts_per_year[year].sum()
            self.class_lvl_2_percentage_per_year[year] = (self.class_lvl_2_counts_per_year[year] / total) * 100

            total = self.class_lvl_1_counts_per_year[year].sum()
            self.class_lvl_1_percentage_per_year[year] = (self.class_lvl_1_counts_per_year[year] / total) * 100

        print("Obliczono procentowy udział klas.")

    def determine_unique_categories(self):
        self.unique_categories_lvl_3_per_year = {}
        self.unique_categories_lvl_2_per_year = {}
        self.unique_categories_lvl_1_per_year = {}

        for year, counts in self.class_lvl_3_counts_per_year.items():
                categories = counts.index
                self.unique_categories_lvl_3_per_year[year] = {level_3_legend_rgb[el.split(" - ")[0]]: el for el in categories}

        for year, counts in self.class_lvl_2_counts_per_year.items():
                categories = counts.index
                self.unique_categories_lvl_2_per_year[year] = {level_2_legend_rgb[el.split(" - ")[0]]: el for el in categories}

        for year, counts in self.class_lvl_1_counts_per_year.items():
                categories = counts.index
                self.unique_categories_lvl_1_per_year[year] = {level_1_legend_rgb[el.split(" - ")[0]]: el for el in categories}

        print("Wyznaczono unikalne kategorie.")

    def generate_images_from_dfs(self,rgb_column):
        images = {}

        for year, df in self.dfs.items():
            width = df['x'].max() + 1
            height = df['y'].max() + 1

            image_array = np.zeros((height, width, 3), dtype=np.uint8)
            rgb = np.stack(df[rgb_column].to_numpy())
            x = df['x'].to_numpy()
            y = df['y'].to_numpy()

            image_array[y, x] = rgb
            image = PILImage.fromarray(image_array, mode='RGB')
            images[year] = image

            print(f'Obraz dla roku {year} ({rgb_column}) utworzony.')

        return images
    
    def generate_lvl_2_and_1_images(self):
        self.images_lvl_1 = self.generate_images_from_dfs(rgb_column='rgb_lvl_1')
        self.images_lvl_2 = self.generate_images_from_dfs(rgb_column='rgb_lvl_2')

    def prepare_analytical_data(self):
        print("-----TWORZENIE DATAFRAME-----\n")
        self.create_analytical_df()
        print("\n------ZLICZENIA KLAS------\n")
        self.count_classes()
        print("\n------ZLICZENIA PROCENTÓW------\n")
        self.calculate_percentage_of_classes()
        print("\n------UNIKALNE KLASY------\n")
        self.determine_unique_categories()
        print("\n------OBRAZY DLA NIŻSZYCH POZIOMÓW------\n")
        self.generate_lvl_2_and_1_images()