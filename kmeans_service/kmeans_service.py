import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from tensorflow.keras.datasets import mnist
import json

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

    results = build_response_from_model(kmeans_model)

    print("will return - {}".format(results))
    return results


# def build_response_from_images_data(result_dict):
#     results = []
#     for cluster in result_dict.values():
#         center_index = np.argmax(y_train == cluster["cluster_center_number"])
#         center_image = X_train[center_index]
#         kmeans_cluster = KMeansClusterResponse(KMeansImageInfo(cluster["cluster_center_number"], center_image))
#         for cluster_number in cluster["cluster_numbers"]:
#             kmeans_cluster.add_cluster_number(KMeansImageInfo(cluster_number[1], X_train[cluster_number[0]]))
#         results.append(kmeans_cluster)
#     return results


def build_response_from_model(kmeans_model):
    response = []
    for i in range(len(kmeans_model.cluster_centers_)):
        center = kmeans_model.cluster_centers_[i]
        center = get_image_format(center)
        cluster = KMeansClusterResponse(i, KMeansNumberInfo(center))
        for label_index in range(len(kmeans_model.labels_)):
            label_cluster_index = kmeans_model.labels_[label_index]
            if label_cluster_index == cluster.cluster_index:
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


def find_cluster_center_numbers(kmeans_result_labels, y_train, clusters_amount):
    # fins the clusters center numbers according to the most matching number to each index
    reference_labels = {}

    for i in range(clusters_amount):
        index = np.where(kmeans_result_labels == i, 1, 0)
        num = np.bincount(y_train[index==1]).argmax()
        reference_labels[i] = num

    return reference_labels


def sanity_print(result_dict):
    fig, axs = plt.subplots(1, NUM_OF_IMAGES_PER_CLUSTER, figsize=(12, 12))
    plt.gray()
    # loop through subplots and add mnist images
    for i, ax in enumerate(axs.flat):
        ax.imshow(X_train[result_dict[0]['cluster_numbers'][i][0]])
        ax.axis('off')
        ax.set_title('Number {}'.format(result_dict[0]['cluster_numbers'][i][1]))
    plt.show()


class KMeansClusterResponse():
    def __init__(self, cluster_index, centroid_number):
        self.cluster_index = cluster_index
        self.centroid_number = centroid_number
        self.cluster_numbers = []

    def add_cluster_number(self, number_info_extended):
        self.cluster_numbers.append(number_info_extended)

    def to_json(self):
        return {
            "cluster_index": self.cluster_index,
            "centroid_number": self.centroid_number.to_json(),
            "cluster_numbers": [number.to_json() for number in self.cluster_numbers]
        }

class KMeansNumberInfo:
    def __init__(self, image_data):
        self.image_data = image_data

    def to_json(self):
        return {
            "image_data": self.image_data,
        }


class KMeansNumberInfoExtended(KMeansNumberInfo):
    def __init__(self, image_data, number_index, number_value):
        super().__init__(image_data)
        self.number_index = number_index
        self.number_value = number_value

    def to_json(self):
        return {
            "image_data": self.image_data,
            "number_index": self.number_index,
            "number_value": self.number_value,
        }