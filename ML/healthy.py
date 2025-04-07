import requests
import json

if __name__ == "__main__":
    url = "http://127.0.0.1:5003/invocations"
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"inputs": [[-1.75938045, -1.2340347, -1.41327673, 0.76150439, 2.20097247, -0.10937195, 0.58931542, 0.1135538]]})

    response = requests.post(url, headers=headers, data=data)
    print(response.json())
