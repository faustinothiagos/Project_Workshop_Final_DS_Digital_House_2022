import streamlit as st
import pandas as pd
import pickle


st.title('Descubra sua Credit Score')
st.header('O que é Credit Score?')
st.subheader('Credit Score é uma pontuação resultante dos hábitos de pagamento e relacionamento do consumidor com o mercado de crédito. Fatores como pagar contas em dia, histórico de dívidas negativadas, saldo devedor etc, representam o risco de inadimplência e saúde financeira geral de uma pessoa. Pode ser Ruim, Regular, Bom.')
st.subheader('\n')
st.subheader('ATENÇÃO: Este modelo não é o oficial e foi criado com a finalidade de ter uma versão interativa no StreamLit. Como as variáveis envolvidas não são tão fáceis de se obter/lembrar para inserir, foram escolhidas algumas delas com esse propósito.  A performance deste modelo é muito pior. Não é recomendado para uso comercial.')

st.sidebar.subheader('Insira seus dados abaixo e descubra sua pontuação.')
divida = st.sidebar.slider("Qual é a sua dívida?", 0,10000, step = 250)/2
salario_mensal = st.sidebar.slider("Qual o seu salário mensal?", 0,25000, step = 500)/2
investimento = st.sidebar.slider("Quanto você investe mensalmente?", 0,1000, step = 50)/2
tempo_cartao = st.sidebar.slider("Quantos anos você possui o cartão?", 0,25)*12
fatura = st.sidebar.radio('Você paga a fatura do cartão em dia?',('sim','não'))

if fatura == 'sim':
       fatura = 1
else:
       fatura = 0


dados_dict = {'Monthly_Inhand_Salary'     :salario_mensal,
              'Outstanding_Debt'          :divida,
              'Amount_invested_monthly'   :investimento,
              'Credit_History_age'        :tempo_cartao,
              'Payment_of_Min_Amount'     :fatura
}


col = ['Outstanding_Debt','Monthly_Inhand_Salary','Credit_History_age','Amount_invested_monthly', 'Payment_of_Min_Amount']

dados_deploy = pd.DataFrame(dados_dict,columns=col,index = [0])

if st.button('Classifique seu Credit Score'):
   pickle_model_xgb = pickle.load(open('Code\modelostreamlit.pkl', 'rb'))
   cs = pickle_model_xgb.predict(dados_deploy)
   if cs == [0]:
      cs = 'Bom'
   elif cs == [1]:
      cs = 'Ruim'
   else:
      cs = 'Regular'
   st.text('Seu Score é: {0}'.format(cs))