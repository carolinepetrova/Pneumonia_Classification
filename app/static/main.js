const submitRequest = (url, body) => {
    var response = fetch(url, {
            method: 'post',
            body: JSON.stringify(body),
            headers: new Headers({
                "content-type": "application/json"
            })
        }).then(response => {
            return response.json();
        })
        .then((json) => {
            return json;
        })
        .catch(error => console.error(error));
    return response;
}

const classify = async () => {
    var elem = document.getElementById("image").value;
    var body = {
        image: elem
    }
    console.log(body);
    var response = await submitRequest(`${window.origin}/predict`, body);
    var output = document.getElementById("output");
    var img_orig = document.getElementById("img-original");
    var heatmap = document.getElementById("img-activation");
    output.innerHTML = "<h2>Output: <b>" + response['prediction'] + "</b></h2>";
    img_orig.innerHTML = '<img src="' + elem + '" class="">';
    heatmap.src = "static/output/" + response['output'];
}