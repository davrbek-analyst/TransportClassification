import streamlit as st
from fastai.vision.all import *
import pathlib
import platform
import plotly.express as px

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
    
#Title
st.title('Transport classifying model')

#Uploading
file = st.file_uploader("Uploading image", type=['png','jpeg','gif','svg'])
if file:
    st.image(file)
    #PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner('transport_model.pkl')

    # prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Prediction: {pred}")
    st.info(f'Probability: {probs[pred_id]*100:.1f}%')
    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
