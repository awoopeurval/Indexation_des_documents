import cv2 as cv
import numpy as np
import os
from scipy.spatial import distance
from tabulate import tabulate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cProfile
from line_profiler import LineProfiler
from memory_profiler import profile
import time


def calculate_histogram(image, M=32):
    # Convertir en espace couleur HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Calculer les histogrammes pour chaque canal
    hist = [cv.calcHist([hsv_image], [i], None, [256], [0, 256]) for i in range(3)]
    
    # Réduire les histogrammes
    reduced_hist = np.zeros((3, M))
    for i in range(3):
        for j in range(M):
            reduced_hist[i, j] = np.sum(hist[i][j * (256 // M):(j + 1) * (256 // M)])
    
    return reduced_hist.flatten()

def calculate_hu_moments(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    moments = cv.moments(gray_image)
    hu_moments = cv.HuMoments(moments).flatten()
    return hu_moments

def calculate_color_distance(hist1, hist2):
    return cv.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv.HISTCMP_CHISQR)

def calculate_shape_distance(hu1, hu2):
    return distance.euclidean(hu1, hu2)

def calculate_global_similarity(hist1, hu1, hist2, hu2, w1=0.5, w2=0.5):
    color_distance = calculate_color_distance(hist1, hist2)
    shape_distance = calculate_shape_distance(hu1, hu2)
    return w1 * color_distance + w2 * shape_distance

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img))
    return images

mem_logs = open('mem_profile_direct.log','a')
@profile(stream=mem_logs)
def find_closest_images(query_image_path, folder, N=5, w1=0.5, w2=0.5):
    query_image = cv.imread(query_image_path)
    if query_image is None:
        print('Could not open or find the query image!')
        return []
    
    query_hist = calculate_histogram(query_image)
    query_hu = calculate_hu_moments(query_image)
    print(query_hu)
    
    images = load_images_from_folder(folder)
    distances = []
    times =[] #
    
    for filename, img in images:
        start_time_req = time.time() #
        img_hist = calculate_histogram(img)
        img_hu = calculate_hu_moments(img)
        distance = calculate_global_similarity(query_hist, query_hu, img_hist, img_hu, w1, w2)
        
        end_time_req = time.time() #
        times.append((filename, end_time_req-start_time_req)) #
        distances.append((filename, distance))
    
    distances.sort(key=lambda x: x[1])

    for filename, time_req in times:
        print(f"Temps d'execution pour {filename} : {time_req} s")
    total_time = sum(time_taken for _, time_taken in times)
    return distances[:N], total_time


# Chemins des images et dossier
query_image_path = 'F:\github.com_extension\indexation_document\coil-100\obj20__250.png'
images_folder = 'F:\github.com_extension\indexation_document\coil-100'

# Trouver les N images les plus proches
N = 5
start = time.time()
closest_images,total_time = find_closest_images(query_image_path, images_folder, N)
end=time.time()

# temps moyen d'execution par image
average_time_per_request = total_time / len(closest_images)
# Afficher les résultats
headers = ["Filename", "Distance"]
print(tabulate(closest_images, headers=headers, tablefmt="grid"))
print(f"Temps d'execution totale: {end-start} s")
print(f"Temps d'execution moyen par image: {(end-start)/N} s")