import streamlit as st
import pandas as pd
import pickle


st.title('Descubra sua Credit Score')
st.header('O que é Credit Score?')
st.subheader('Credit Score é uma pontuação resultante dos hábitos de pagamento e relacionamento do consumidor com o mercado de crédito. Fatores como pagar contas em dia, histórico de dívidas negativadas, saldo devedor etc, representam o risco de inadimplência e saúde financeira geral de uma pessoa. Pode ser Ruim, Regular, Bom.')
st.subheader('\n')
st.subheader('\n')

st.sidebar.subheader('Insira seus dados abaixo e descubra sua pontuação.')
divida = st.sidebar.slider("Qual é a sua dívida?", 0,10000, step = 250)/2
salario_mensal = st.sidebar.slider("Qual o seu salário mensal?", 0,25000, step = 500)/2
investimento = st.sidebar.slider("Quanto você investe mensalmente?", 0,1000, step = 50)/2
tempo_cartao = st.sidebar.slider("Quantos anos você possui o cartão?", 0,25)*12
fatura = st.sidebar.radio('Você paga a fatura do cartão em dia?',('sim','não'))


dados_dict = {'Outstanding_Debt':divida,
              'Monthly_Inhand_Salary':salario_mensal,
              'Credit_History_age':tempo_cartao,
              'Amount_invested_monthly':investimento,
              'Payment_of_Min_Amount':fatura      
             }

col = ['Outstanding_Debt','Monthly_Inhand_Salary','Credit_History_age','Amount_invested_monthly',
       'Payment_of_Min_Amount']

dados_deploy = pd.DataFrame(dados_dict,columns=col,index = [0])

if st.button('Classifique seu Credit Score'):
    pickle_model_xgb = pickle.load(open('modelostreamlit.pkl', 'rb'))
    st.text('Seu Score é: {0}'.format(pickle_model_xgb.predict(dados_deploy)))