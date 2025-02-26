import joblib
import pandas as pd
import streamlit as st


model = joblib.load(r'cancerLogistic.joblib')
scaler = joblib.load(r'scaler.pkl')
columns = pd.read_csv(r'parametros.csv').columns[1:]
numeric_var = {column: 0 for column in columns}

for item in numeric_var:
    
    if 'radius' in item or 'texture' in item:
        valor = st.number_input(f'{item}', step=0.01, value=0.0)
        
    else:
        valor = st.number_input(f'{item}', step=0.0001, value=0.0, format="%.4f")
        
    numeric_var[item] = valor

btn = st.button('Prever o resultado')

if btn:
    user_input = pd.DataFrame(numeric_var, [0])
    user_input_esc = scaler.transform(user_input)
    result = model.predict(user_input_esc)
    
    msg = 'Malign' if result[0] == 1 else 'Benign'
    st.write(msg)

