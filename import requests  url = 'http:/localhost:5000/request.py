import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'precipitation':2, 'temp_max':9, 'temp_min':6})

print(r.json())
