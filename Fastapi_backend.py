from fastapi import FastAPI , File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = tf.keras.models.load_model("../1")
class_names = ["corn_cercospera_leaf", "corn_common_rust", "corn_leaf_blight","corn_healthy", "potato_early_blight","potato_healthy","potato_late_blight","tomato_bacterial_spot","tomato_healthy","tomato_leaf_mold"]
@app.get("/ping")
async def ping():
    return "hello"

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File()
):
    contents = await file.read()
    image = read_file_as_image(contents)
    image_batch = np.expand_dims(image, axis=0)
    predictions = model.predict(image_batch)
    predictions_array = np.array(predictions[0])
    predicted_class = class_names[np.argmax(predictions_array)]
    return predicted_class


if __name__ == "__main__":
    uvicorn.run(app , host = 'localhost', port = 8000 )