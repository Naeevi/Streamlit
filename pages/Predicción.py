import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pickle

# Configurar la página
st.set_page_config(page_title="Predicción", layout="wide")

st.title("¡Realicemos predicciones! :crystal_ball:")
st.markdown("Puedes dibujar un número o cargar una foto. ¡Elige la opción que quieras!")
st.subheader("Dibuja un número :pencil2:")

# Crear el scaler
scaler = StandardScaler()

# Cargar el modelo desde el archivo
with open("svm_digits_model.pkl", "rb") as f:
    modelo = pickle.load(f)

scaler = modelo["scaler"]
clf = modelo["clf"]

# Crear el canvas para dibujar
canvas = st_canvas(
    stroke_width=10, 
    stroke_color="white", 
    background_color="black", 
    height=150, 
    width=150, 
    drawing_mode="freedraw", 
    key="canvas")

def preprocess_image(image, scaler):
    # Convertir a escala de grises
    image = image.convert('L')

    #Redimensionar, ya que SVM fue entrenado con imagenes 8x8
    image = image.resize((8, 8)) 
    
    #Necesitamos convertirla en una matriz de números, y que 
    # sean hexadecimales, no de 0 a 255, porque digits usa eso, por eso escalamos
    image_array = np.array(image)
    image_array = 16 * (image_array / 255.0)
   
    # Aplanar la imagen para que sea un vector de 64 elementos
    image_array = image_array.flatten().reshape(1, -1)

    # Aplicar el mismo scaler que usaste al entrenar
    image_array = scaler.transform(image_array)
     
    # SVM expera (n_sample, n_features)
    # Aseguramos una fila (1 imagen), con 64 columnas (64 píxeles)
    # image_array = image_array.reshape(1, -1) 

    return image_array

# Preprocesar la imagen del canvas para SVM
def preprocesar_canvas_para_svm(image_data, scaler):
    if image_data is None:
        return None
    
    image_scaled = preprocess_image(Image.fromarray(image_data.astype('uint8')), scaler)
    return image_scaled


#Predice mediante la imagen 
def predict(image, scaler):
    img_array = preprocess_image(image, scaler)
    prediccion = clf.predict(img_array)
    return prediccion[0]

# Predice con la imagen
if st.button("Predict"):
    if canvas.image_data is not None:
        img_array = np.array(canvas.image_data)  # Asegurar que sea NumPy array
        img_processed = preprocesar_canvas_para_svm(img_array.astype('uint8'), scaler)

        prediction = clf.predict(img_processed)
        st.subheader("Predicción")
        st.write(f"El modelo predice que el número es: **{prediction}**")

st.subheader("Carga una imagen :camera:")

archivo_subido = st.file_uploader("Sube una imagen manuscrita (JPG o PNG)", type=["jpg", "png"])

if archivo_subido is not None:
    # Mostrar imagen con PIL
    image = Image.open(archivo_subido)
    st.image(image, caption='Imagen subida', width=150)  
    st.write("")

    # Make a prediction
    prediction = predict(image, scaler)
    st.subheader(f"✅ El modelo predice que el número es: **{prediction}**")

st.write("Esta app usa OpenCV para procesar imágenes y Scikit-learn para predecir dígitos manuscritos.")


