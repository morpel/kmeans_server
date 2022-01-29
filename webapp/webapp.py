from flask import Flask, render_template, request, Response, jsonify
from kmeans_service import kmeans_service
from PIL import Image
import io
import base64
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def get_base64_image(image_data):
    im = Image.fromarray(image_data.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)
    return str(base64.b64encode(rawBytes.read()).decode('utf-8'))


@app.route('/calculate_clusters', methods=['POST'])
def calculate_clusters():
    kmeans_clusters = kmeans_service.calculate_clusters(int(request.get_json()[0]['cluster_size']))

    response_data = []
    for cluster in kmeans_clusters:
        cluster_response = {
            'cluster_index': cluster.cluster_index,
            'centroid_number': {
                'image_data_base64': get_base64_image(cluster.centroid_number.image_data)
            },
            'cluster_numbers': []
        }
        for cluster_number in cluster.cluster_numbers:
            cluster_response["cluster_numbers"].append({
                'number_value': cluster_number.number_value,
                'number_index': cluster_number.number_index,
                'image_data_base64': get_base64_image(cluster_number.image_data)
            })
        response_data.append(cluster_response)

    response_json = {'result': response_data}
    return response_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008, debug=True)