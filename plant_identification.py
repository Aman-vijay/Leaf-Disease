import base64
import requests
import json
import data


def encode_file(file_name):
    with open(file_name, "rb") as file:
        return base64.b64encode(file.read()).decode("ascii")


def identify_plant_names(file_names):
    # More optional parameters: https://github.com/flowerchecker/Plant-id-API/wiki/Plant-identification
    params = {
        "images": [encode_file(img) for img in file_names],
        "latitude": 49.1951239,
        "longitude": 16.6077111,
        "datetime": 1582830233,
        # Modifiers docs: https://github.com/flowerchecker/Plant-id-API/wiki/Modifiers
        "modifiers": ["crops_fast", "similar_images"],
        "plant_language": "en",
        # Plant details docs: https://github.com/flowerchecker/Plant-id-API/wiki/Plant-details
        "plant_details": ["common_names",
                          "edible_parts",
                          "gbif_id",
                          "name_authority",
                          "propagation_methods",
                          "synonyms",
                          "taxonomy",
                          "url",
                          "wiki_description",
                          "wiki_image",
                          ],
        }

    headers = {
        "Content-Type": "application/json",
        "Api-Key": "wv4wog1KvtowfDYpNwpkHmYsx3W5pivc8UrVUpS9axhOTEsFzq",
        }

    response = requests.post("https://api.plant.id/v2/identify",
                             json=params,
                             headers=headers)

    with open("C:/Users/user/Downloads/iden_lateblight.json", "w+") as f:
        json.dump(response.json(), f)
                         

    
       
    with open("C:/Users/user/Downloads/iden_lateblight.json", "r") as f:
        data = json.load(f)

    pd = data["suggestions"][0]["plant_details"]["common_names"]
    print(pd)

    return response.json()


if __name__ == '__main__':
   (identify_plant_names(["C:/Users/user/Downloads/potatoheal.jpg"]))