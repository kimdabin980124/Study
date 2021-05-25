import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#피어슨 상관계수 분석, #p값 분석
df = pd.read_csv('gong.csv')
df=df.dropna(axis=0)
coef,p_value=stats.pearsonr(df.iloc[0][1:].astype('float'),df.iloc[1][1:].astype('float'))
print('피어슨 상관계수') # 피어슨 상관계수 출력
print(coef)
print('p값') #p값 출력
print(p_value)


lm=LinearRegression()
shuttle = df.iloc[0][1:].astype('int')
subway = df.iloc[1][1:].astype('int')

sub=np.array(subway).reshape(-1,1)
shu=np.array(shuttle).reshape(-1,1)
lm.fit(sub,shu)


plotsub = []
plotshu = []
for i in shuttle :
    plotshu.append(i)

for i in subway :
    plotsub.append(i)

plotd= {'shuttle' : plotshu, 'subway' : plotsub }
plotdata=pd.DataFrame(plotd)
sns.regplot(x='subway',y='shuttle' ,data=plotdata)
plt.ylim(0,)
plt.title("Regression Plot : Subway & Shuttle")
plt.show()

Yhat=lm.predict(sub)

print('MSE') #MSE
print(mean_squared_error(shu,Yhat))
print('R^2 상관계수') #R^2
print(lm.score(sub,shu))

print('셔틀버스 탑승자 예측') # 셔틀버스 탑승자 예측
print(lm.predict(sub))
expect=lm.predict(sub)
hourexpect=[sum(expect[0:5]),sum(expect[5:11]),sum(expect[11:17]),sum(expect[17:23]),sum(expect[23:29]),sum(expect[29:35]),sum(expect[35:41]),sum(expect[41:47]),sum(expect[47:53]),sum(expect[53:59]),sum(expect[59:65])]
#시간대별 탑승자 수 예측

final=[]
for i in hourexpect:
    final.append(int(i))


print('최종 시간대별 총 탑승자 ')
print(final)

print('set bus timetable')

for i in final:
    print(60/(i/45))