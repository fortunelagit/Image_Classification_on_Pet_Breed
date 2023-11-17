import requests

resp = requests.post("http://127.0.0.1:5000", 
                     files={'file': open('./Abyssinian_24', 'rb')})

print(resp.json())
