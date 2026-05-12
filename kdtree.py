from sklearn.neighbors import KDTree
import numpy as np
import os
from PIL import Image
import time

dataset = "C:/Users/moreo/OneDrive/Desktop/lfw_funneled"


def loadImages(numOfImages = 1000):
    images = []
    labels = []
    for person in os.listdir(dataset):
        personPath = os.path.join(dataset, person)
        
        if not os.path.isdir(personPath):
            continue
        
        for people in os.listdir(personPath):
            peoplePath = os.path.join(personPath, people)
            
            if people.lower().endswith((".png", ".jpg" ,"jpeg")):
                try:
                    image = Image.open(peoplePath).convert("L").resize((64, 64))
                    vector = np.asarray(image).ravel()
                    images.append(vector)
                    labels.append(person)
                
                    if len(images) >= numOfImages:
                        return np.array(images), labels
                except:
                    continue
    return np.array(images), labels

images, labels = loadImages(4000)
tree = KDTree(images)
start = time.perf_counter()
search = time.perf_counter()

for i in range(len(images)):
    distance, index = tree.query([images[i]], k = 2)
    
end = time.perf_counter()

totalTime = end - start


print(f"Total time to search: {totalTime:.6f} seconds")    
   
   
   
   
