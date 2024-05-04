import streamlit as st
import numpy as np

from common import DataStorage as ds


@st.cache_resource
def load_data(name):
    # load/get model from cache
    return ds.load_model(name)

def main(names):
    # title
    st.title('Iris dataset')

    # load model 
    model = load_data('../data/models/model.pkl')
    
    # load scaler
    scaler = load_data('../data/models/scaler.pkl')

    # data entry in interface streamlit
    with st.container():
        sepal_length = st.slider(
            'Sepal length',
            min_value=4.0,
            max_value=8.0,
            value=4.0,
            step=0.1,
            format="%.1f",
        )
        sepal_width = st.slider(
            'Sepal width',
            min_value=2.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
            format="%.1f",
        )
        petal_length = st.slider(
            'Petal length',
            min_value=1.0,
            max_value=7.0,
            value=1.0,
            step=0.1,
            format="%.1f",
        )
        petal_width = st.slider(
            'Petal width',
            min_value=0.0,
            max_value=2.5,
            value=0.0,
            step=0.1,
            format="%.1f",
        )

        if st.button('Predict'):
            input_data = np.array([
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]).reshape(1, -1)

            # preprocess input data
            input_data_scaled = scaler.transform(input_data)
            
            # predict
            prediction = model.predict(input_data_scaled)
            
            st.markdown("### Prediction")
            st.markdown(f"Predicted species is: **{names[prediction[0]]}**")


if __name__ == "__main__":
    NAMES_FOR_PREDICT = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    main(NAMES_FOR_PREDICT)
