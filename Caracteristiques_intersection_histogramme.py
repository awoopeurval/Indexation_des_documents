from __future__ import print_function, division
from tabulate import tabulate
import cv2 as cv
import numpy as np

# Spécifiez directement les chemins des images ici
image_path1 = 'F:\github.com_extension\indexation_document\coil-100\obj1__5.png'
image_path2 = 'F:\github.com_extension\indexation_document\coil-100\obj2__5.png'
image_path3 = 'F:\github.com_extension\indexation_document\coil-100\obj3__5.png'

src_base = cv.imread(cv.samples.findFile(image_path1))
src_test1 = cv.imread(cv.samples.findFile(image_path2))
src_test2 = cv.imread(cv.samples.findFile(image_path3))

if src_base is None or src_test1 is None or src_test2 is None:
    print('Could not open or find the images!')
    exit(0)

# Convertir en HSV
hsv_base = cv.cvtColor(src_base, cv.COLOR_BGR2HSV)
hsv_test1 = cv.cvtColor(src_test1, cv.COLOR_BGR2HSV)
hsv_test2 = cv.cvtColor(src_test2, cv.COLOR_BGR2HSV)

# Convertir la moitié inférieure de l'image de base en HSV
hsv_half_down = hsv_base[hsv_base.shape[0]//2:, :]

# Utiliser 50 bins pour la teinte et 60 pour la saturation
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]

# La teinte varie de 0 à 179, la saturation de 0 à 255
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges  # concaténer les listes

# Utiliser les canaux 0 et 1
channels = [0, 1]

# Calculer les histogrammes pour les images HSV
hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

hist_half_down = cv.calcHist([hsv_half_down], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_half_down, hist_half_down, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

hist_test1 = cv.calcHist([hsv_test1], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

hist_test2 = cv.calcHist([hsv_test2], channels, None, histSize, ranges, accumulate=False)
cv.normalize(hist_test2, hist_test2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

# Appliquer les méthodes de comparaison des histogrammes
results = []

# Appliquer les méthodes de comparaison des histogrammes
for compare_method in range(4):
    base_base = cv.compareHist(hist_base, hist_base, compare_method)
    base_half = cv.compareHist(hist_base, hist_half_down, compare_method)
    base_test1 = cv.compareHist(hist_base, hist_test1, compare_method)
    base_test2 = cv.compareHist(hist_base, hist_test2, compare_method)


    results.append([compare_method, base_base, base_half, base_test1, base_test2])
# Afficher les résultats sous forme de tableau
headers = ["Method", "Perfect", "Base-Half", "Base-Test(1)", "Base-Test(2)"]
print(tabulate(results, headers=headers, tablefmt="grid")) 
