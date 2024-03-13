import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
from openai import OpenAI
import openai


#Tensorflow Model Prediction
def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

def model_prediction(test_image):
    model = load_model('C:/Users/Admin/.streamlit/model.h5',compile=False)
    contents = test_image.read()
    image = read_file_as_image(contents)
    image_batch = np.expand_dims(image, axis=0)
    predictions = model.predict(image_batch)
    predictions_array = np.array(predictions[0])
    return np.argmax(predictions_array) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","Chatbot"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "image.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ["corn_cercospera_leaf", "corn_common_rust", "corn_leaf_blight","corn_healthy", "potato_early_blight","potato_healthy","potato_late_blight","tomato_bacterial_spot","tomato_healthy","tomato_leaf_mold"]
        disease_info = {
            "corn_cercospera_leaf": "Corn Cercospera Leaf Spot is a foliar disease that reduces photosynthetic capacity, stalk strength, and kernel weight. It can be managed by using resistant corn hybrids and applying fungicides, especially if applied early when few pustules have appeared on the leaves.",
            "corn_common_rust": "Corn Common Rust is a foliar disease that reduces photosynthetic capacity, stalk strength, and kernel weight. The best management practice is to use resistant corn hybrids. Fungicides can also be beneficial, especially if applied early when few pustules have appeared on the leaves.",
            "corn_leaf_blight": "Corn Leaf Blight is a foliar disease that reduces photosynthetic capacity, stalk strength, and kernel weight. This disease can be controlled by using resistant corn hybrids and practicing crop rotation.",
            "corn_healthy": "This corn plant appears to be healthy.",
            "potato_early_blight": "Potato Early Blight is a foliar disease that can cause significant yield losses in potato crops. It can be managed by using resistant potato varieties, practicing crop rotation, and applying fungicides.",
            "potato_healthy": "This potato plant appears to be healthy.",
            "potato_late_blight": "Potato Late Blight is a foliar disease that can cause significant yield losses in potato crops. It can be controlled by using resistant potato varieties, practicing crop rotation, and applying fungicides.",
            "tomato_bacterial_spot": "Tomato Bacterial Spot is a bacterial disease that affects the fruit and leaves of tomato plants. It can be managed by using resistant tomato varieties, practicing crop rotation, and applying copper-based bactericides.",
            "tomato_healthy": "This tomato plant appears to be healthy.",
            "tomato_leaf_mold": "Tomato Leaf Mold is a foliar disease that can reduce photosynthetic capacity and yield in tomato crops. It can be controlled by using resistant tomato varieties, practicing crop rotation, and applying fungicides."
        }
        st.write(disease_info[class_name[result_index]])
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

elif (app_mode == "Chatbot"):
    st.title("Plant - Chatbot🌿")
    client = OpenAI(api_key = st.secrets["OPENAI_API_KEY"])
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Can I Help You?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
