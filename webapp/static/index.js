function calculate_clusters() {
    var cluster_size = parseInt(document.getElementById("clusters_amount").value);
    var server_data = [{"cluster_size": cluster_size}];
    $.ajax({
        type: "POST",
        url: "/calculate_clusters",
        data: JSON.stringify(server_data),
        contentType: "application/json",
        dataType: 'json',
        success: function(response) {
            console.log(response)
            $('#result').empty();
            var clusterView = ''
            $.each(response.result, function (index, value){
                clusterView += '<div>'
                clusterView += '<strong> Cluster Index: ' + value.cluster_index + '</strong>'
                clusterView += '<div> Centroid Image Data: <img src="data:image/png;base64, ' + value.centroid_number.image_data_base64 + '"  alt="Red dot"/> </div>'
                clusterView += '</br>'
                clusterView += '<div> Cluster Images </div>'
                $.each(value.cluster_numbers, function (index, value){
                    clusterView += '<div> Image: ' + index + ' - </div>'
                    clusterView += '<div> Number Value: ' + value.number_value + '</div>'
                    clusterView += '<div> Number Index: ' + value.number_index + '</div>'
                    clusterView += '<div> Image Data: <img src="data:image/png;base64, ' + value.image_data_base64 + '"  alt="Red dot"/> </div>'
                    clusterView += '</br>'
                })
                clusterView += '</br></br></div>'
            })
            $('#result').append(clusterView)
        }
    });
}