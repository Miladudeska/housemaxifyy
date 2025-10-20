import json
from urllib import request

url = 'http://127.0.0.1:5000/api/predict'
payload = json.dumps({'sqft_living': 1500, 'bedrooms': 3, 'bathrooms': 2}).encode('utf-8')
req = request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
with request.urlopen(req, timeout=10) as resp:
    body = resp.read().decode('utf-8')
    print('status', resp.status)
    print(body)
