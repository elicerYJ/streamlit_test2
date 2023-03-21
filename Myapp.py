import streamlit as st

st.title("텍스트 데이터 기반 문서 분류 프로젝트")
st.markdown('---')

st.header("프로젝트 목표")
st.write("한국어 원문 데이터(법원 판결문)의 요약문을 카테고리('일반행정', '세무', '특허', '형사', '민사', '가사')별로 분류하는 프로젝트 수행")

st.header("데이터 출처")
st.write("https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=580")


st.header("프로젝트 개요")
st.write("이번 프로젝트에서는 LSTM 기술을 활용하여 법원 판결문을 분류하는 프로젝트를 수행합니다.") 
st.write("법원 판결문 데이터를 형태소 분석기를 활용하여 한국어 텍스트를 전처리하는 방법과 이를 학습하여 분류 성능을 확인합니다.")

# --- ch1 ---
st.subheader("1. 데이터 읽기")
st.write("pandas를 사용하여 `project_data_all3.json` 데이터를 읽고 dataframe 형태로 저장하겠습니다.")

code = '''
import pandas as pd
import streamlit as st

fp = './project_data_all3.json'
df = pd.read_json(fp)

# streamlit을 이용한 데이터프레임 출력
st.dataframe(df)
'''
st.code(code, language='python')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

fp = './project_data_all3.json'
df = pd.read_json(fp)
st.dataframe(df)

st.write("`category` 종류 확인 결과")
st.bar_chart(df['category'].value_counts())

df['category'] = df['category'].replace({'가사': 0, '형사': 1, '특허': 2, '민사': 3, '일반행정': 4, '세무': 5})

st.write('''
정답 레이블이 되는 `category`데이터를 `target`변수에 저장합니다.
학습할 요약문 데이터 `abstractive`를 `data`변수에 저장합니다.
''')

code = '''
target = df['category'].values
data = df['abstractive'].values
'''
st.code(code, language='python')

target = df['category'].values
data = df['abstractive'].values

st.write("target 데이터의 개수를 확인 : " + str(len(target)))
st.write("abstractive 데이터의 개수를 확인 : " + str(len(data)))

# ---ch2---
st.subheader("2. 형태소 분석하기")
st.write("KoNLPy(\'코엔엘파이\'라고 읽습니다)는 한국어 정보처리를 위한 파이썬 패키지입니다.")
st.write("이번 프로젝트에서는 Okt(Twitter) 클래스를 활용하겠습니다.")

code = '''
# 판결요약문을 KoNLPy 의 Okt 클래스로 형태소 분석
from konlpy.tag import Okt

# Okt 객체 선언
okt = Okt()

# stemming기반 형태소 분석
# 먼저 요약문 1개만 품사 태깅을 해보겠습니다.
pos_results = okt.pos(data[0][0], norm=True, stem=True)

# 품사를 태깅한다는 것은 주어진 텍스트를 형태소 단위로 나누고 명사, 조사, 동사 등의 형태소를 배열 형태로 만다는 과정입니다.
st.write(pos_results)
'''
st.code(code, language='python')

from konlpy.tag import Okt

okt = Okt()
pos_results = okt.pos(data[0][0], norm=True, stem=True)

st.write("(▾를 누르면 결과를 축소할 수 있습니다)")
st.write(pos_results)

# 판결요약문 데이터를 형태소 분석 결과로 저장 
data_tokenized = []

# 학습데이터로 명사만 사용
for text in data:
    data_tokenized.append(okt.nouns(text[0]))

# 행태소 분석된 결과를 확인
st.write("판결요약문 데이터를 형태소 분석 결과로 저장하여 형태소 분석을 통해 명사만 추출 (개수 : "+str(len(data_tokenized))+")")
st.write("원본 문장 : " + str(data[0]))
st.write("(▾를 누르면 결과를 축소할 수 있습니다)")
st.write(data_tokenized[0])

from bokeh.plotting import figure

p = figure(
    x_axis_label = "length of samples",
    y_axis_label = "number of samples"
)
a = [len(s) for s in data_tokenized]

np_test = np.array(a)
unique, count = np.unique(np_test, return_counts=True)

p.quad(
    top=count[:-1], 
    bottom=0, 
    left=unique[:-1],
    right=unique[1:],
    fill_color="navy", 
    line_color="white", 
    alpha=0.5
)

st.write("`data_tokenized` 변수의 각 배열마다 몇개의 명사가 들어있는지 히스토그램으로 확인하면 대부분의 요약문이 20~60개의 명사를 가지고 있다는 것을 확인할 수 있습니다.")
st.bokeh_chart(p, use_container_width=True)

# ---ch3---
st.subheader("3. Keras 텍스트 전처리")
st.write("형태소 분석된 결과를 학습하기 위해서 Keras를 활용하겠습니다.")

code = '''
# Keras의 텍스트 전처리기를 이용하여 정수 인코딩
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

# Local에서 학습한 tokenizer 객체 호출하기
import pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
'''

st.code(code, language='python')

from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

st.write('''
streamlit이 있는 페이지에 학습 코드(`.fit_on_texts()`)을 작성하면 페이지가 호출될 때마다 학습을 진행하게 됩니다.
그럴경우 가상환경의 리소스 부족으로 인해 페이지가 정상적으로 출력되지 않는 문제가 있습니다.

학습은 여러분의 Local 환경에서 진행해주시고 잘 학습된 모델을 저장해주세요.
Streamlit 페이지 내의 코드에서 저장된 모델을 불러와서 진행해주시면 리소스를 줄일 수 있습니다
''')

tokenizer = Tokenizer()

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

data_index = tokenizer.texts_to_sequences(data_tokenized)


# ---ch3---
st.subheader("4. LSTM으로 판결 요약문 분류하기")

st.write('''
텍스트 분류를 LSTM을 통해서 수행하겠습니다.
먼저 `data_index`의 학습할 데이터를 학습데이터 80%, 테스트데이터 20% 비율로 나눠주겠습니다. 그리고 앞서 설명한바와 같이 각 카테고리의 비율을 유지하기 위하여 `stratify` 에 파라미터에 정답 레이블 데이터를 설정해줍니다.
''')

code = '''
# LSTM으로 판결요약문 분류하기
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# class 비율(train:validation)에 유지하기 위해 stratify 옵션을 target으로 지정
X_train, X_test, y_train, y_test = train_test_split(data_index, target, test_size=0.2, stratify=target, random_state=100)
'''
st.code(code, language='python')

from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

X_train, X_test, y_train, y_test = train_test_split(data_index, target, test_size=0.2, stratify=target, random_state=100)

st.write('''
훈련용 데이터와 테스트용 데이터를 `원-핫 인코딩` 하겠습니다.
`원-핫 인코딩`은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식입니다.
이번 실습에서는 카테고리('일반행정', '세무', '특허', '형사', '민사', '가사')의 개수가 6개이므로 벡터의 크기는 6이 됩니다.
''')
         
max_len = 40

X_train = pad_sequences(X_train, maxlen=max_len) # 훈련용 판결요약문 패딩
X_test = pad_sequences(X_test, maxlen=max_len) # 테스트용 판결요약문 패딩


y_train = to_categorical(y_train) # 훈련용 판결요약문 레이블의 원-핫 인코딩
y_test = to_categorical(y_test) # 테스트용 판결요약문 레이블의 원-핫 인코딩


st.write('''
`Embedding()`은 최소 두 개의 인자를 받습니다. 
첫번째 인자는 단어 집합의 크기, 즉 총 단어의 개수입니다.
두번째 인자는 임베딩 벡터의 출력 차원, 즉 결과로서 나오는 임베딩 벡터의 크기입니다.
결과적으로 아래의 코드는 120차원을 가지는 임베딩 벡터 1,000개를 생성합니다. 
마지막으로 6개의 카테고리를 분류해야하므로, 출력층에서는 6개의 뉴런을 사용합니다. 활성화 함수로는 소프트맥스를 사용하여 6개의 확률분포를 만듭니다. 
''')

code = '''
# 사용할 모델 호출
model = Sequential()
model.add(Embedding(1000, 120))
model.add(LSTM(120))
model.add(Dense(6, activation='softmax'))
'''
st.code(code, language='python')


model = Sequential()
model.add(Embedding(1000, 120))
model.add(LSTM(120))
model.add(Dense(6, activation='softmax'))

st.write('''
이제 학습을 진행합니다. 
하지만 streamlit이 있는 페이지에 학습 코드(`.fit()`)을 작성하면 페이지가 호출될 때마다 학습을 진행하게 됩니다.
그럴경우 가상환경의 리소스 부족으로 인해 페이지가 정상적으로 출력되지 않는 문제가 있습니다.

학습은 여러분의 Local 환경에서 진행해주시고 잘 학습된 모델을 저장해주세요.
Streamlit 페이지 내의 코드에서 저장된 모델을 불러와서 진행해주시면 리소스를 줄일 수 있습니다
''')
         
code = '''
history = model.fit(
    X_train, 
    y_train, 
    batch_size=128, 
    epochs=30, 
    callbacks=[es, mc], 
    validation_data=(X_test, y_test)
)
'''
st.code(code, language='python')

code = '''
# 모델 호출 코드
loaded_model = load_model('best_model.h5')
'''
st.code(code, language='python')

loaded_model = load_model('best_model.h5')

st.write('''**테스트 정확도** ''')
st.write(loaded_model.evaluate(X_test, y_test)[1])

loss_df = pd. DataFrame(
    {
        "loss" : [1.671, 1.222, 0.879, 0.6623, 0.5258, 0.406, 0.326, 0.271, 0.218, 0.165, 0.146, 0.132],
        "val_loss" : [1.402, 1.043, 0.875, 0.780, 0.722, 0.760, 0.690, 0.719, 0.720, 0.746, 0.764, 0.847]
    }
)

st.subheader("Model Loss")

col1, col2 = st.columns([1, 3])

with col1 :
    st.dataframe(loss_df)
with col2 :
    st.line_chart(loss_df)


st.subheader("요약문에 대한 예측값 확인")
st.write('''
학습한 모델을 바탕으로 모든 요약문에 대한 예측값을 출력해보겠습니다.
''')

df['category'] = df['category'].replace({0:'가사', 1:'형사', 2:'특허', 3:'민사', 4:'일반행정', 5:'세무'})

X_all = pad_sequences(data_index, maxlen=max_len)
y_all_pred = np.argmax(loaded_model.predict(X_all),axis=1)

df['pred'] =  y_all_pred
df['pred'] = df['pred'].replace({0:'가사', 1:'형사', 2:'특허', 3:'민사', 4:'일반행정', 5:'세무'})

predict_df = pd.DataFrame(
    {
        "contents" : [],
        "real_category" : [],
        "redict_category" : []
    }
)

for i in range(len(df)):
    row = [df['abstractive'][i][0], df['category'][i], df['pred'][i]]
    predict_df.loc[i] = row

option = st.selectbox(
    "몇 개의 예측 결과를 출력할까요? (단위 : 개)",
    (10, 20, 30, 40, 50)
)

st.dataframe(predict_df.head(option))
