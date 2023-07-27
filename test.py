import requests

# Replace '{YOUR_API_KEY}' with your actual HERE API key
api_key = '_XecI_2z9_7QVDELBzW_dT8VeRjUW4uJtOkxpm4Qvrs'

url = f'https://router.hereapi.com/v8/routes'
params = {
    'origin': '52.66181,13.38251',
    'destination': '51.772324,12.402652',
    'return': 'summary,typicalDuration',
    # 'spans': 'dynamicSpeedInfo,length,consumption,speedLimit,length',
    'transportMode': 'privateBus',
    'vehicle[speedCap]': '27',
    'vehicle[grossWeight]': '135000',
    'departureTime': 'any',
    'ev[freeFlowSpeedTable]': '0,0.239,27,0.239,45,0.259,60,0.196,75,0.207,90,0.238,100,0.26,110,0.296,120,0.337,130,0.351,250,0.351',
    # 'ev[trafficSpeedTable]': '0,0.349,27,0.319,45,0.329,60,0.266,75,0.287,90,0.318,100,0.33,110,0.335,120,0.35,130,0.36,250,0.36',
    'ev[ascent]': '9',
    'ev[descent]': '4.3',
    'apikey': api_key
}

response = requests.get(url, params=params)
data = response.json()
print(data)
# print("Departure time:", data["routes"][0]["sections"][0]["departure"]["time"])
# print("Summary:", data["routes"][0]["sections"][0]["summary"])





