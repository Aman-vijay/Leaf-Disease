import base64
import requests
import json
import data
from plant_identification import identify_plant_names

def encode_file(file_name):
    with open(file_name, "rb") as file:
        return base64.b64encode(file.read()).decode("ascii")



def identify_plant_diseses(file_names):
    # More optional parameters: https://github.com/flowerchecker/Plant-id-API/wiki/Plant-Health-Assessment
    params = {
        "images": [encode_file(img) for img in file_names],
        "latitude": 49.1951239,
        "longitude": 16.6077111,
        "datetime": 1582830233,
        # Modifiers docs: https://github.com/flowerchecker/Plant-id-API/wiki/Modifiers
        "modifiers": ["crops_fast", "similar_images"],
        "language": "en",
        # Disease details docs: https://github.com/flowerchecker/Plant-id-API/wiki/Disease-details
        "disease_details": ["cause",
                          "common_names",
                          "classification",
                          "description",
                          "treatment",
                          "url",
                          ],
        }

    headers = {
        "Content-Type": "application/json",
        "Api-Key": "wv4wog1KvtowfDYpNwpkHmYsx3W5pivc8UrVUpS9axhOTEsFzq",
        }

    response = requests.post("https://api.plant.id/v2/health_assessment",
                             json=params,
                             headers=headers)
    


    with open("C:/Users/user/Downloads/file2_lateblight.json", "w+") as f:
        json.dump(response.json(), f)
                         

    
       
    with open("C:/Users/user/Downloads/file2_lateblight.json", "r") as f:
        data = json.load(f)

# NOTE: DONE
    for i in range(len(data)):
        dd = data["health_assessment"]["diseases"][i]["disease_details" ]
        cn = dd["common_names"]
        if cn is not None:
            print(cn[0])
            
    return response.json()        

if __name__ == '__main__':
    identify_plant_names(["C:/Users/user/Downloads/potatoheal.jpg"])
    identify_plant_diseses(["C:/Users/user/Downloads/potatoheal.jpg"])
    