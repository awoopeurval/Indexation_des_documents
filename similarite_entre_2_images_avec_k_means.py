import cv2 as cv
import numpy as np
import os
from scipy.spatial import distance
from tabulate import tabulate
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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


mem_logs_kmeans = open('mem_profile_kmeans.log','a')
@profile(stream=mem_logs_kmeans)
def find_closest_images_with_K_means(query_image_path, folder, N=5, w1=0.5, w2=0.5, n_clusters=3):
    query_image = cv.imread(query_image_path)
    if query_image is None:
        print('Could not open or find the query image!')
        return []
    
    query_hist = calculate_histogram(query_image)
    query_hu = calculate_hu_moments(query_image)
       
    images = load_images_from_folder(folder)
    features = []
    
    for filename, img in images:
        img_hist = calculate_histogram(img)
        img_hu = calculate_hu_moments(img)
        combined_features = np.concatenate((img_hist, img_hu))
        features.append(combined_features)

    features_array = np.array(features)
    print(f"FEATURES ARRAY : \n {features_array} \n")

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features_array)

    query_features = np.concatenate((query_hist, query_hu)).reshape(1, -1)
    print(f"QUERY FEATURES : \n {query_features} \n")
    query_cluster = kmeans.predict(query_features)
    print(f"QUERY CLUSTER : \n {query_cluster} \n")

    # Cercher les images presentes dans le meme cluster que l'image requete
    same_cluster_indices = np.where(kmeans.labels_ == query_cluster[0])[0]
    distances = []
    times =[] #

    # Calculer les distances entre l'image requete et les images du meme cluster seulement
    for idx in same_cluster_indices:
        start_time_req = time.time() #
        filename, _ = images[idx]
        distance = calculate_global_similarity(query_hist,query_hu, features_array[idx][:len(query_hist)], features_array[idx][len(query_hist):], w1, w2)
        end_time_req = time.time() #
        times.append((filename, end_time_req-start_time_req)) #
        
        distances.append((filename, distance))

    for filename, time_req in times:
        print(f"Temps d'execution pour {filename} : {time_req} s")
    total_time = sum(time_taken for _, time_taken in times)
    # Filtrer en fonction de la distance et selection le top N 
    distances.sort(key=lambda x: x[1])
    return distances[:N], total_time


# Chemins des images et dossier
query_image_path = 'F:\github.com_extension\indexation_document\coil-100\obj20__250.png'
images_folder = 'F:\github.com_extension\indexation_document\coil-100'

# Trouver les N images les plus proches
N = 5

# Nombre de cluster
n_clusters = 20

start = time.time()
closest_images,total_time = find_closest_images_with_K_means(query_image_path, images_folder, N, n_clusters)
end=time.time()

# temps moyen d'execution par image
average_time_per_request = total_time / len(closest_images)
# Afficher les résultats
headers = ["Filename", "Distance"]
print(tabulate(closest_images, headers=headers, tablefmt="grid"))
print(f"Temps d'execution : {end-start} s")
print(f"Temps d'execution moyen par image: {(end-start)/N} s")
