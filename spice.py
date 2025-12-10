import numpy as np
import matplotlib.pyplot as plt
import time

### TODO 1
### Load data from data_path
### Return : np array of size Nx2
def load_data(data_path):
    return np.loadtxt(data_path, delimiter=',')

### TODO 2.1
### If init_centers is None, initialize randomly
def initialise_centers(data, K, init_centers=None):
    if init_centers is not None:
        return init_centers
    idx = np.random.choice(data.shape[0], K, replace=False)
    return data[idx]

### TODO 2.2
### Initialize labels to all ones
def initialise_labels(data):
    return np.ones(data.shape[0], dtype=int)

### TODO 3.1 : E step
### Return distances (NxK)
def calculate_distances(data, centers):
    diff = data[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)

### TODO 3.2 : E step
### Assign nearest center label
def update_labels(distances):
    return np.argmin(distances, axis=1)

### TODO 4 : M step
### Update centers
def update_centers(data, labels, K):
    return np.array([data[labels == k].mean(axis=0) for k in range(K)])

### TODO 6 : Check convergence
def check_termination(labels1, labels2):
    return np.array_equal(labels1, labels2)

### DON'T CHANGE ANYTHING BELOW
def kmeans(data_path:str, K:int, init_centers):
    data = load_data(data_path)
    centers = initialise_centers(data, K, init_centers)
    labels = initialise_labels(data)

    start_time = time.time()

    while True:
        distances = calculate_distances(data, centers)
        labels_new = update_labels(distances)
        centers = update_centers(data, labels_new, K)
        if check_termination(labels, labels_new):
            break
        else:
            labels = labels_new

    end_time = time.time()
    return centers, labels, end_time - start_time

### TODO 7
def visualise(data_path, labels, centers):
    data = load_data(data_path)

    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')

    plt.title('K-means clustering')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.savefig('kmeans.png')
    return plt

# Run the code
if __name__ == "__main__":
    data_path = 'spice_locations.txt'
    K, init_centers = 2, None

    centers, labels, time_taken = kmeans(data_path, K, init_centers)
    print('Time taken for the algorithm to converge:', time_taken)

    visualise(data_path, labels, centers)
