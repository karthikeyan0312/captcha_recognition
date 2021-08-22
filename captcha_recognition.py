import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

st.title("Captcha Recognition")
class FileReference:
    def __init__(self, filename):
        self.filename = filename

def hash_file_reference(file_reference):
    with open(file_reference.filename) as f:
      return f.read()
@st.cache(hash_funcs={FileReference: hash_file_reference})
def load(path=r"/app/captcha_recognition/model.h5"):
    model=load_model(path)  
    return model

model=load()

#Init main values
symbols = "abcdefghijklmnopqrstuvwxyz0123456789" # All symbols captcha can contain
num_symbols = len(symbols)
img_shape = (50, 200, 1)

def predict(upload):
    #img = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    file_bytes = np.asarray(bytearray(upload.read()),dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if img is not None:
        img = img / 255.0
    else:
        st.warning("Error in Uploaded Image")
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (5, 36))
    
    l_ind = []
    probs = []
    for a in ans:
        l_ind.append(np.argmax(a))
    print(l_ind)
    capt = ''
    for l in l_ind:
        capt += symbols[l]
    return capt#, sum(probs) / 5

upload=st.file_uploader('Choose a File ',type=["jpg","jpeg","png"])
if upload  is not None:
    try:
        p=predict(upload)
        st.success("Predicted Captcha : "+p)
    except:
        st.warning("You Have Upladed Wrong Image")
else:
    st.info("Upload image")
