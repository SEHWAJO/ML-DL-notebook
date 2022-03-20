#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# # 데이터셋 컬럼들의 의미 잘 이해하기

# ▶ feature Description

# datetime : 시간 (YYYY-MM-DD 00:00:00)
# 
# season : 봄(1) 여름(2) 가을(3) 겨울(4)
# 
# holiday : 휴일(1) 그외(0)
# 
# workingday : 근무일(1) 그외(0)
# 
# weather : 날씨
# 
# temp : 온도
# 
# atemp : 
# 
# humidity : 습도
# 
# windspeed : 바람속도
# 
# casual :
# 
# registered :
# 
# count :
# 
# 카테코리 type : season, holidy, workingday, weather
# 
# 수치형 type : temp, atemp, humidity, windspeed, casual, registered, count

# ▶ 사용할 Library

# In[2]:


# 필요한 라이브러리를 import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import datetime as dt
import scipy

# 프로파일링
from pandas.plotting import scatter_matrix


# ▶ 전체 데이터셋 불러오기

# In[3]:


# data 불러오기
train = pd.read_csv('/Users/sehwajo/Downloads/bike-sharing-demand/train.csv')
test = pd.read_csv('/Users/sehwajo/Downloads/bike-sharing-demand/test.csv')
submission = pd.read_csv('/Users/sehwajo/Downloads/bike-sharing-demand/sampleSubmission.csv')


# # 컬럼명 , 데이터타입, 사이즈, 중복값, 결측치 확인하기

# ▶ 컬럼명 확인

# In[4]:


# 데이터 확인
train.columns


# In[5]:


test.columns


# In[6]:


submission.columns


# In[7]:


train.head(10)


# In[8]:


test.head(10)


# ▶ 데이터타입 확인

# In[9]:


train.info()


# In[10]:


test.info()


# ▶ 데이터 사이즈 확인

# In[11]:


print(train.shape)
print(test.shape)


# ▶ 중복값 확인

# In[12]:


# 중복값 확인
train.T.duplicated()


# In[13]:


# 중복값 확인
test.T.duplicated()


# In[175]:


# 결측치 확인
train.isnull().sum()


# # datetime 컬럼은 날짜로 잘 인식하도록 해주기

# In[14]:


# 데이터의 datetime을 날짜로 인식해주기 위해서
# pandas의 to_datetime 을 활용하여 datetime 컬럼을 날짜로 인식하게끔 형태를 바꾸어 주었다.



train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])


# In[19]:


#현재 데이터 날짜는 yyyy-mm-dd 00:00:00 형태이다. 
#이를 년,월,일,시,분,초로 나누어 연도별, 월별.... 의 자전거 수요량이 어떻게 변화하는지 우선적으로 확인해보겠다.


train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second


test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test['minute'] = test['datetime'].dt.minute
test['second'] = test['datetime'].dt.second


#주 단위로도 확인해보기위해 datetime 패키지에서 지원하는 dayofweek도 출력해보았다.
# dayofweek 는 요일을 가져오는 말
#월(0) 화(1) 수(2) 목(3) 금(4) 토(5) 일(6)
train['dayofweek'] = train['datetime'].dt.dayofweek


train.shape # 컬럼 수가 늘어난 것 확인


# In[20]:


train


# In[22]:


# dayofweek 살펴보기
#월(0) 화(1) 수(2) 목(3) 금(4) 토(5) 일(6)

train['dayofweek'].value_counts()


# # 시각화 해보기 (우선 카테고리 타입 컬럼들만)

# # barplot

# In[76]:


sns.barplot(data = train, x = 'year', y = 'count')


# In[77]:


sns.barplot(data = train, x = 'month', y='count')


# In[78]:


sns.barplot(data = train, x = 'day', y= 'count')


# In[79]:


sns.barplot(data = train, x = 'hour', y= 'count')


# In[80]:


sns.barplot(data = train, x = 'minute', y= 'count')


# # Boxplot

# In[21]:


# 카테고리 변수들을 Boxplot으로 시각화하기

fig, axes = plt.subplots(nrows = 5, ncols = 2, figsize=(16, 18)) 
sns.boxplot(data = train, y="count", x = "season", orient = "v", ax = axes[0][0]) 
sns.boxplot(data = train, y="count", x = "holiday", orient = "v", ax = axes[0][1]) 
sns.boxplot(data = train, y="count", x = "workingday", orient = "v", ax = axes[1][0]) 
sns.boxplot(data = train, y="count", x = "weather", orient = "v", ax = axes[1][1]) 
sns.boxplot(data = train, y="count", x = "dayofweek", orient = "v", ax = axes[2][0]) 
sns.boxplot(data = train, y="count", x = "month", orient = "v", ax = axes[2][1]) 
sns.boxplot(data = train, y="count", x = "year", orient = "v", ax = axes[3][0]) 
sns.boxplot(data = train, y="count", x = "hour", orient = "v", ax = axes[3][1]) 
sns.boxplot(data = train, y="count", x = "minute", orient = "v", ax = axes[4][0])


# # pointplot

# In[83]:


plt.figure(figsize=(10, 5))
sns.pointplot(data = train, x = 'hour', y = 'count')


# In[84]:


# workingday (카테고리형) , 시간대별로 point plot 확인
#seaborn에서 카테고리형 데이터를 고려한 그래프를 출력하고 싶으면, hue변수를 추가해준다.
#따라서 시간대별로 holiday == 1일때, holiday =- 0 일때로 출력된다.
# 근무일 : 1, 근무일이 아닌 경우 : 0

plt.figure(figsize=(10, 5))
sns.pointplot(data = train, x = 'hour', y = 'count', hue = 'workingday')


# In[85]:


plt.figure(figsize=(10, 5))
sns.pointplot(data = train, x = 'hour', y = 'count', hue = 'holiday')

# 휴일이 아닌 날 : 0, 휴일인 날 : 1


# In[86]:


plt.figure(figsize=(10, 5))
sns.pointplot(data = train, x = 'hour', y = 'count', hue = 'weather')
# 아주깨끗한날씨(1) 약간의 안개와 구름(2) 약간의 눈,비(3) 아주많은비와 우박(4)
# 4번 날씨 데이터는 거의 없음


# # 수치형 컬럼들은 산점도와 상관계수를 확인해보자

# In[18]:


# 수치형 컬럼들의 상관관계를 알아보자
# 산점도와 상관계수를 통해 수치형 features가 target과 어떤 관계를 가지고 있는지 파악하자

# 수치형 컬럼만 corr_data라는 변수에 넣어주기 
num_data = train[["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"]] 


# 수치형 컬럼들을 산점도로 통해 시각화하여 살펴보기
scatter_matrix(num_data, alpha=0.5, figsize=(30, 30), diagonal='kde')
plt.show()


# In[87]:


# 위에서 만들어 놓은 수치형 데이터 (corr_data) 로 상관계수 맵 그리기


# 상관계수맵 그리기

colormap = plt.cm.PuBu 

f , ax = plt.subplots(figsize = (12,10)) 
plt.title('Correlation of Numeric Features with Rental Count',y=1,size=18) 
sns.heatmap(num_data.corr()
            , linewidths=0.1
            , square=True
            , annot=True
            ,cmap=colormap)


# In[20]:


# count열을 보면 가장 상관관계가 높은 변수는 registered이다. 이 변수는 test 데이터에는 없다.
# casual + registered = count 니까 상관관계를 보는 것이 의미가 없음
# 그 다음으로 상관관계가 높은 변수는 casual 이다.
# 그 외 온도, 습도, 풍속은 거의 관계가 없어보인다.
# temp(온도)와 atemp(체감온도는) 둘이 상관관계가 매우 높은 것을 보아 카디널리티(다중공선성)가 의심됨.


# In[88]:


# 상관계수 구하기
num_data_corr = num_data.corr()
num_data_corr


# In[89]:


# 2) 상관계수 : 순서대로 나열 후 상관계수 수치를 확인하여 feature 선택 여부 결정
num_data_target_corr = num_data_corr[['count']].sort_values('count', ascending=False)
num_data_target_corr[1:6]


# # 수치형(연속형) 변수 스캐터플랏으로 살펴보기

# In[18]:


# 수치형(연속형) 변수 살펴보기 - registered 랑 casual은 의미가 없기 때문에 제외하고 살펴볼 것

fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (12,5))

sns.scatterplot(data = train, x = 'windspeed', y = 'count', ax = ax1 )
sns.scatterplot(data = train, x = 'temp', y = 'count', ax = ax2)
sns.scatterplot(data = train, x = 'humidity', y = 'count', ax = ax3)


# In[24]:


#windspeed의 경우 0에 많은 데이터가 몰려있습니다.  
#일반적으로 풍속이 0인 경우가 흔치 않으므로 
#Null데이터를 0으로 대체한게 아닌가 생각해볼 수 있습니다.


# # 의심이 가는 피쳐는 시각화로 자세히 보기

# In[91]:


# windspeed 에 대한 시각화를 해보니 0값이 많은 걸 더 잘 알 수 있다.
plt.figure(figsize=(20, 15))
sns.countplot(data = train, x = "windspeed")


# In[92]:


# windspeed 데이터중 값이 0 인 개수
len(train[train['windspeed']==0])


# ### 위에서 발견한 문제 windspeed 컬럼에서 결측치를 0으로 넣어놓은 것으로 예상되는 것은 뒤에서 다시 다룸

# # 박스플롯 그려서 수치형?연속형? 변수에 대한 이상치 제거하기

# In[187]:


# 연속형 변수에 대한 이상치 제거

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows = 6, figsize = (12,10))

sns.boxplot(data = train, x = 'windspeed', ax = ax1)
sns.boxplot(data = train, x = 'humidity', ax = ax2)
sns.boxplot(data = train, x = 'temp', ax = ax3)
sns.boxplot(data = train, x = 'casual', ax = ax4)
sns.boxplot(data = train, x = 'registered', ax = ax5)
sns.boxplot(data = train, x = 'count', ax = ax6)


# In[188]:


from collections import Counter 

def detect_outliers(df, n, features): 
    outlier_indices = [] 
    for col in features: 
        Q1 = np.percentile(df[col], 25) 
        Q3 = np.percentile(df[col], 75) 
        IQR = Q3 - Q1 
        
        outlier_step = 1.5 * IQR 
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col) 
    outlier_indices = Counter(outlier_indices) 
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n) 
    
    return multiple_outliers 

Outliers_to_drop = detect_outliers(train, 2, ["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"])


# In[193]:


# 이상치 drop 하기전 데이터셋 크기
train.shape


# In[194]:


# 이상치 drop 후 데이터셋 크기 확인
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop = True)
train.shape


# # 왜도와 첨도 확인해서 이상치 제거하기

# 5.2 왜도(skewness)와 첨도(kurtosis) 확인
# 데이터 분석에서의 왜도와 첨도는 중요하다. 간단하게 정리하자면, 아래와 같고, 왜도와 첨도의 수치를 보고 데이터의 치우침 정도를 알 수 있기 때문이다.
# 
# 왜도
# 
# 데이터의 분포가 한쪽으로 쏠린것을 의미
# skew의 수치가 -2~+2 정도가 되어야 치우침이 없는 데이터
# -2이하는 왼쪽으로 쏠린 데이터 (negative) +2 이상은 오른쪽으로 쏠린 데이터(positive)
# positive일경우 변환방법 : square root, cube, log(밑10)
# negative일경우 변환방법 : square, cube root, logarithmic(밑2인log)
# 
# 첨도
# 
# 
# 분포의 뾰족함이나 평평함에 관련된 것이 아니라 분포의 tail에 대한 모든 것
# 한쪽 꼬리부분의 극값과 다른쪽 꼬리의 극값과의 차이를 보여줌
# 아웃라이어를 찾을 때 주로 사용
# 첨도가 높다 -> 아웃라이어가 많이 있다

# In[97]:


# 피쳐 엔지니어링 하기

# 왜도(Skewness)와 첨도(Kurtosis) 확인하기

df_train_num = train[["count", "temp", "atemp", "casual", "registered", "humidity", "windspeed"]] 

for col in df_train_num: 
    print('{:15}'.format(col), 
          'Skewness: {:05.2f}'.format(train[col].skew()) ,
          ' ' ,
          'Kurtosis: {:06.2f}'.format(train[col].kurt()) 
         )


# In[98]:


f, ax = plt.subplots(1, 1, figsize = (10,6)) 
g = sns.distplot(train["count"], color = "b", label="Skewness: {:2f}".format(train["count"].skew()), ax=ax) 
g = g.legend(loc = "best") 

print("Skewness: %f" % train["count"].skew()) 
print("Kurtosis: %f" % train["count"].kurt())


# 결론적으로 수치상으로는 왜도와 첨도에 문제가 없게 출력된다. 하지만 데이터의 히스토그램을 보아하니, count 가 0에 굉장히 많이 치우쳐저 있는 것을 확인 할 수 있다. 이때 Log scaling을 통해 정규화 시켜주도록 하자.
# 
# (여기서 주의할 점은, y값인 count 값에 log를 취해주었으니, 마지막에 나온 예측결과값에는 다시 log를 취해주어야 원래 원하던 값이 나온다!!!!

# In[99]:


#일반적으로 왜도는 0을 기준으로 판단하고, 첨도는 3을 기준으로 판단한다고 합니다. 

#수치적으로는 그렇게 큰 차이는 없지만 distplot으로 확인해보니 0에 많이 치우친 것을 확인할 수 있습니다. 

#Log Scaling을 이용하여 정규화 시켜주도록 합니다

train["count_Log"] = train["count"].map(lambda i:np.log(i) if i>0 else 0) 
f, ax = plt.subplots(1, 1, figsize = (10,6)) 
g = sns.distplot(train["count_Log"], color = "b", label="Skewness: {:2f}".format(train["count_Log"].skew()), ax=ax)
g = g.legend(loc = "best") 


print("Skewness: %f" % train['count_Log'].skew()) 
print("Kurtosis: %f" % train['count_Log'].kurt()) 

train.drop('count', axis= 1, inplace=True)


# 타겟변수인 count에 Log를 취해준 모습입니다.
# 
#  
# 
# 모델에 사용되는 모든 변수들은 첨도와 왜도를 확인해주는 것이 좋습니다. 
# 
# 또한 변수들의 크기 차이에 민감한 모델을 사용한다면 더욱 Scaling은 필수입니다. 
# 
#  
# 
# 이번 경우는 사실 애매하긴 하지만 저는 개인적으로 이럴 때 모델 성능 기준으로 테스트해보고 결정합니다. 
# 
# 일단 이번 커널에서는 타겟변수에만 Log를 취해주도록 하겠습니다! 
# 
# 
# 
# 출처: https://hong-yp-ml-records.tistory.com/76?category=823206 [HONG YP's Data Science BLOG]

# # 풍속 0값 처리하기 - 해결 못함
# 풍속이 0일때가 거의 없는 것을 고려하여 windspeed 값을 대체해주기로 한다. 
# 
# << 결측값 처리 방법>>
# 
# - 결측값을 앞 방향 혹은 뒷 방향으로 채우기
# - 결측값을 변수별 평균으로 채우기
# - 결측값을 가진 데이터를 삭제하기
# - 결측값을 0 이나, 아예 다른 값으로 (-999) 대체하기
# - 결측값을 예측된 값으로 대체하기 (머신러닝을 돌려서)
# 
# windspeed가 null값인 경우를 0으로 처리한 것이라고 생각했기에, 결측치를 채우는 방법중에 하나의 방법인 예측된 값으로 대체하는 방법으로 0을 바꾸어주려고 한다.
# 

# In[195]:


# windspeed 0인 값 존재여부 확인
len(train[train['windspeed'] == 0.0])


# In[101]:


# train과 test의 windspeed 값 시각화

fig, (ax1, ax2) = plt.subplots(2,1)
fig.set_size_inches(20,15)

#갯수를 세야하니 countplot
sns.countplot(data = train, x = 'windspeed', ax = ax1)
sns.countplot(data = test, x = 'windspeed', ax = ax2)

# windspeed의 값에서 0.0 이 사라진 것을 확인할 수 있다.


# 

# In[196]:





# # 필요없는 컬럼 삭제
# 일단 workingday는 holiday 와 너무 비슷한 양상을 띄고 있어서 workiingday를 삭제해주기로 했다. 
# 
# 또 temp와 atemp의 상관관계가 매우 높아 다중공선성이 의심됐기 때문에, atemp 변수를 삭제해 주었다. 
# 
# year, month, day 등 시간에 대한 변수가 따로 존재하기때문에 datetime 도 삭제해주고, 
# 
# 초단위, 분단위에 따른 자전거 수요량의 변화는 알기 쉽지 않다고 판단하여 없애 주었다. 
# 
# 
# (train 데이터세트와 test 데이터세트에서 모두 지워줄 것)
# 
# 그리고 registered랑 casual을 합친 값이 count 이기 때문에 registered와 casual 도 지우자

# In[102]:


# submission의 형태를 살펴보았을 때, datetime을 기준으로 예측값을 적었다.
# 따라서 test의 datetime은 미래의 submission을 위해서 따로 저장해두기로 한다.

test_datetime = test['datetime']


# In[103]:


train.columns


# In[104]:


train.drop(['datetime', 'workingday', 'atemp', 'registered', 'casual', 'minute', 'second'], axis = 1, inplace = True)
test.drop(['datetime', 'workingday', 'atemp', 'minute', 'second'], axis = 1, inplace = True) 


# In[105]:


# 결과적으로 남은 train 데이터셋


# In[107]:


# 결과적으로 남은 test 데이터셋
test


# # 원핫인코딩 - 필요없는 컬럼 Drop 하고 나서 해주는게 편함

# In[ ]:


# 범주형 변수들인 season, holiday, workingday, weather 에서 값들이 수치를 의미하지 않기 때문에 원핫인코딩으로 변수를 처리해주었다.

#prefix 란, 변수 생성명 앞에 weather_1 이런식으로 생성되게 하는 것

train = pd.get_dummies(train, columns = ['season'], prefix = 'season')
test = pd.get_dummies(test, columns = ['season'], prefix = 'season')

train = pd.get_dummies(train, columns = ['weather'], prefix = 'weather')
test = pd.get_dummies(test, columns = ['weather'], prefix = 'weather')


train = pd.get_dummies(train, columns = ['holiday'], prefix = 'holiday')
test = pd.get_dummies(test, columns = ['holiday'], prefix = 'holiday')


# In[207]:


train


# In[ ]:




