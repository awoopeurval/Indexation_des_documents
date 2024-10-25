from __future__ import print_function, division
import cv2 as cv
import numpy as np
import random as rng

rng.seed(12345)

def calculer_distance_euclidienne(vecteur1, vecteur2):
    """
    Calcule la distance euclidienne entre deux vecteurs de moments de Hu.
    """
    return np.linalg.norm(vecteur1 - vecteur2)


def thresh_callback(val):
    threshold = val

    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 

    # Get the moments
    mu = [cv.moments(contour) for contour in contours]

    # Calculate Hu Moments
    huMoments = [cv.HuMoments(m).flatten() for m in mu]

    # Get the mass centers
    mc = [(m['m10'] / (m['m00'] + 1e-5), m['m01'] / (m['m00'] + 1e-5)) for m in mu]

    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i, contour in enumerate(contours):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, contours, i, color, 2)
        cv.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)


    cv.imshow('Contours', drawing)

    # Calculate the area with the moments 00 and compare with the result of the OpenCV function
    for i, contour in enumerate(contours):
        print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' %
              (i, mu[i]['m00'], cv.contourArea(contour), cv.arcLength(contour, True)))
        
         # Print Hu Moments for each contour
        print(' * Contour[%d] - Hu Moments: %s' % (i, ', '.join(['%.5f' % moment for moment in huMoments[i]])))

    # Exemple de calcul de distance entre deux vecteurs de moments de Hu (pour les deux premiers contours si disponibles)
    if len(huMoments) > 1:
        distance = calculer_distance_euclidienne(huMoments[0], huMoments[1])
        print(f"Distance entre les deux premiers contours: DISTANCE={distance}")

# Spécifiez directement le chemin de l'image ici
image_path = 'F:\github.com_extension\indexation_document\coil-100\obj9__0.png'

src = cv.imread(cv.samples.findFile(image_path))
if src is None:
    print('Could not open or find the image:', image_path)
    exit(0)

# Convertir l'image en niveaux de gris et la flouter
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))

source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)

max_thresh = 255
thresh = 100  # seuil initial
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv.waitKey()
