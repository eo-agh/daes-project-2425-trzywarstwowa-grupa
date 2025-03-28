## Cel
Celem projektu jest analiza zmian w użytkownaniu wybranych obszarów Europy na przestrzeni czasu na podstawie sklasyfikowanych danych rastrowych.
## Metody
Dane źródłowe pochodzą z zbiórów CORINE Land Cover. Dostępne zbiory danych pochodzą z lat: 1990, 2000, 2006, 2012, 2018. Dane pobierane są za pomocą REST API udostępnianego przez serwis Copernicus. Dane są sklasyfikowane według użytkowania terenu z podziałem na 3 poziomy szczegółowości. Projekt zakłada analizę na każdym z poziomów. Będziemy badać różnice w procentowych udziale danego użytkowania terenu na przestrzeni lat. Ostatnim etapem będzie wizualizacja badań.
## Techniczna implementacja
Python, Jupyter Notebook, 
Biblioteki pythona: requests, NumPy, pandas, PIL