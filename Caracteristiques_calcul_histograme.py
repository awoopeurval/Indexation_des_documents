from __future__ import print_function, division
import cv2 as cv
import numpy as np

# Spécifiez directement le chemin de l'image ici
image_path = 'F:\github.com_extension\indexation_document\coil-100\obj1__0.png'

src = cv.imread(cv.samples.findFile(image_path))
if src is None:
    print('Could not open or find the image:', image_path)
    exit(0)

bgr_planes = cv.split(src)

# Taille de l'histogramme avant réduction
histSize = 256
histRange = (0, 256)  # la borne supérieure est exclusive
accumulate = False

b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

# Réduction de l'histogramme à 32 valeurs
M = 32
factor = 256 // M

reduced_b_hist = np.zeros((M,))
reduced_g_hist = np.zeros((M,))
reduced_r_hist = np.zeros((M,))

for i in range(M):
    reduced_b_hist[i] = np.sum(b_hist[i * factor:(i + 1) * factor])
    reduced_g_hist[i] = np.sum(g_hist[i * factor:(i + 1) * factor])
    reduced_r_hist[i] = np.sum(r_hist[i * factor:(i + 1) * factor])

# Normalisation des histogrammes réduits pour les afficher
hist_h = 400
cv.normalize(reduced_b_hist, reduced_b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(reduced_g_hist, reduced_g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(reduced_r_hist, reduced_r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

# Création de l'image pour afficher les histogrammes réduits
hist_w = 512
bin_w = int(round(hist_w / M))

histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

for i in range(1, M):
    cv.line(histImage, (bin_w * (i - 1), hist_h - int(reduced_b_hist[i - 1])),
            (bin_w * i, hist_h - int(reduced_b_hist[i])),
            (255, 0, 0), thickness=2)
    cv.line(histImage, (bin_w * (i - 1), hist_h - int(reduced_g_hist[i - 1])),
            (bin_w * i, hist_h - int(reduced_g_hist[i])),
            (0, 255, 0), thickness=2)
    cv.line(histImage, (bin_w * (i - 1), hist_h - int(reduced_r_hist[i - 1])),
            (bin_w * i, hist_h - int(reduced_r_hist[i])),
            (0, 0, 255), thickness=2)

cv.imshow('Source image', src)
cv.imshow('Reduced Histogram Demo', histImage)
cv.waitKey()
