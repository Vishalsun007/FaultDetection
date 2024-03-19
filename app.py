import streamlit as st
import pickle
import numpy as np
import tensorflow as tf


failure_mapper = {
    0: 'No Failure',
    1: 'Power Failure',
    2: 'Tool Wear Failure',
    3: 'Overstrain Failure',
    4: 'Random Failures',
    5: 'Heat Dissipation Failure'
}

def process_input(input_arr):
    with open("random_forest_model.pkl", "rb") as file:
        rf = pickle.load(file)

    model = tf.keras.models.load_model('keras_model.h5')

    ar = np.array(input_arr).reshape(1, -1)

    rf_prediction = rf.predict(ar)

    keras_prediction = model.predict(ar)
    keras_argmax = np.argmax(keras_prediction)

    return rf_prediction[0], keras_argmax

def main():
    st.title("Fault Detection App")

    air_temperature = st.number_input("Enter Air Temperature:", value=0.0, step=0.1)
    process_temperature = st.number_input("Enter Process Temperature:", value=0.0, step=0.1)
    torque = st.number_input("Enter Torque:", value=0.0, step=0.1)

    type_mapper_options = {'M': 0, 'L': 1, 'H': 2}
    type_mapper = st.selectbox("Select Type:", options=list(type_mapper_options.keys()))

    input_arr = [air_temperature, process_temperature, torque, 0, type_mapper_options[type_mapper]]

    if st.button("Submit"):
        rf_prediction, keras_argmax = process_input(input_arr)
        st.write("RandomForest Prediction:", failure_mapper[rf_prediction])
        st.write("Keras Model Prediction (Argmax):", failure_mapper[keras_argmax])

if __name__ == "__main__":
    main()
