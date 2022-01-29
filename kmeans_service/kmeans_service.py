import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tensorflow.keras.datasets import mnist

NUM_OF_IMAGES_PER_CLUSTER = 5

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data formatting
X_train_formatted = X_train.reshape(len(X_train), -1)
X_train_formatted = X_train_formatted.astype(np.float32) / 255.
print("loaded mnist data")


def calculate_clusters(clusters_amount):
    # Initialize the K-Means model
    kmeans_model = MiniBatchKMeans(n_clusters=clusters_amount)

    # Fitting the model to training set
    kmeans_model.fit(X_train_formatted)

    response = build_response_from_model(kmeans_model)

    print("will return - {}".format(response))
    return response


def build_response_from_model(kmeans_model):
    response = []
    for i in range(len(kmeans_model.cluster_centers_)):
        center = kmeans_model.cluster_centers_[i]
        center = get_image_format(center)
        cluster = KMeansClusterResponse(i, KMeansNumberInfo(center))
        for label_index in range(len(kmeans_model.labels_)):
            label_cluster_index = kmeans_model.labels_[label_index]
            if label_cluster_index == cluster.cluster_index:  # fill only this center's image list
                cluster.add_cluster_number(KMeansNumberInfoExtended(X_train[label_index], label_index, int(y_train[label_index])))
                if len(cluster.cluster_numbers) == NUM_OF_IMAGES_PER_CLUSTER:
                    break
        response.append(cluster)

    return response


def get_image_format(center):
    center = np.reshape(center, (28, 28))
    center = center * 255
    center = center.astype(np.uint8)
    return center


class KMeansClusterResponse:
    def __init__(self, cluster_index, centroid_number):
        self.cluster_index = cluster_index
        self.centroid_number = centroid_number
        self.cluster_numbers = []

    def add_cluster_number(self, number_info_extended):
        self.cluster_numbers.append(number_info_extended)


class KMeansNumberInfo:
    def __init__(self, image_data):
        self.image_data = image_data


class KMeansNumberInfoExtended(KMeansNumberInfo):
    def __init__(self, image_data, number_index, number_value):
        super().__init__(image_data)
        self.number_index = number_index
        self.number_value = number_value
