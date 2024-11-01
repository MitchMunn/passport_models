import requests
from PIL import Image
from io import BytesIO
import datasets

from transformers import ViTFeatureExtractor, ViTForImageClassification


def bytes_to_pil(example_batch):
    example_batch["img"] = [
        Image.open(BytesIO(b)) for b in example_batch.pop("img_bytes")
    ]
    return example_batch


if __name__ == "__main__":
    # r = requests.get('https://github.com/dchen236/FairFace/blob/master/detected_faces/race_Asian_face0.jpg?raw=true')
    # im = Image.open(BytesIO(r.content))

    ds = datasets.load_dataset("nateraw/fairface", trust_remote_code=True)
    ds = ds.with_transform(bytes_to_pil)

    predictions = []
    gt = []
    for i in range(5):
        image = ds["train"][i]["img"]
        gt.append(ds["train"][i]["age"])

        # Init model, transforms
        model = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier")
        transforms = ViTFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")

        # Transform our image and pass it through the model
        inputs = transforms(image, return_tensors="pt")
        output = model(**inputs)

        # Predicted Class probabilities
        proba = output.logits.softmax(1)

        # Predicted Classes
        preds = proba.argmax(1)
        predictions.append(preds.item())

    print(gt)
    print(predictions)
