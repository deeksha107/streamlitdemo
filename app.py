import streamlit as stm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import MonthEnd
from sklearn import linear_model, metrics, model_selection
import seaborn as sns
from pandas.tseries.offsets import MonthEnd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
sns.set_style('whitegrid')
import scipy.stats as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import statistics as s
from scipy import stats
from sklearn.metrics import mean_squared_error as mse
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model, metrics, model_selection
import io
import warnings
warnings.filterwarnings("ignore")


def main():
    stm.title("Chemical Price Forecast Project ")
    file1 = stm.sidebar.file_uploader( "Upload Chemical Price file",type=("xlsx", "csv")) 
    file2 = stm.sidebar.file_uploader( "Upload Affecting factors file",type=("xlsx", "csv"))
    if file1 and file2:
        data_rw,eco=load_data(file1,file2)
        Zones = list(set(data_rw['Zone']))
        Chemicals = list(set(data_rw['Lib Produit']))
        chemical=stm.sidebar.selectbox('Choose the Chemicals',Chemicals)
        zone = stm.sidebar.selectbox('Choose the Zones',Zones)
        month=stm.sidebar.slider('Select the Month',1, 12)
        CI=stm.sidebar.number_input("Confidence Interval")
        CI_SI=stm.sidebar.checkbox("CI_SI")
        
        CI_ML=stm.sidebar.checkbox("CI_ML")
        if stm.sidebar.button("Run"):
            forecasting_overall_1(data_rw,eco,data_on_chem_1(data_rw,eco,chemical),month,zone,chemical,CI,CI_SI,CI_ML)
            #pred_forecast_CI_2(data_rw,eco,data_on_chem_1(data_rw,eco,chemical),month,zone,chemical,CI,CI_SI,CI_ML)
 
def hash_io(input_io):
    return (input_io.getvalue(), input_io.tell())


@stm.cache(hash_funcs={io.BytesIO: hash_io, io.StringIO: hash_io})
def load_data(file1,file2):
    data_rw =pd.read_excel(file1)
    eco = (pd.read_excel(file2, 'daily data',skiprows=6, na_values='#N/A N/A').iloc[:, 1:].rename(columns={'Unnamed: 1': 'time'}))
    return data_rw,eco
    
def data_on_chem_1(data_rw,eco,chemical):
    data_Zno = data_rw[data_rw['Lib Produit'] == chemical].reset_index(drop=True)
    data_Zno['Date'] = pd.to_datetime(data_Zno[['YEAR', 'MONTH']].assign(Day=1))+MonthEnd(1)
    data_Zno['MONTH']=data_Zno['Date'].apply(lambda x: x.strftime('%b'))
    #data_Zno['Zone']
    data_Zno_f=data_Zno[['Date','Zone','TOTAL AGREE PRICE']]
    Data_avg=data_Zno_f.groupby(['Date','Zone']).mean().reset_index()
    g_2=eco.copy()
    g_2['month'] = pd.DatetimeIndex(g_2['time']).month
    g_2['year'] = pd.DatetimeIndex(g_2['time']).year
    g_2['Date'] = pd.to_datetime(g_2[['year', 'month']].assign(Day=1))+MonthEnd(1)
    Chemical_avg_2=g_2.groupby(['Date']).mean().reset_index()
    mod_111 = pd.merge(Data_avg,Chemical_avg_2,left_on='Date', right_on='Date')
    return mod_111
    


    
def data_prepare(data_rw,eco,data,lag,Zone,chemical):
    data=data_on_chem_1(data_rw,eco,chemical)
    np_9=data[(data['Zone']==Zone)]
    np_99=np_9.reset_index().copy().iloc[0:np_9.shape[0]-lag]
    np_99['TAP4']=0*np_99.shape[0]
    for i in range(np_99.shape[0]-lag):
        np_99['TAP4'][i]=np_99['TOTAL AGREE PRICE'][i+lag]
    np_109=np_9.reset_index().copy()
    np_109['TAP4']=0*np_9.shape[0]
    for i in range(np_109.shape[0]-lag):
        np_109['TAP4'][i]=np_109['TOTAL AGREE PRICE'][i+lag]
    np_19=np_99.drop(['Date'],axis=1)
    np_20=np_19.drop(['Zone'],axis=1)
    np_21=np_20.copy()
    for i in range(lag):
        np_21['TAP4'][np_21.shape[0]-lag+i]=np.nan
    np_22=np_21.iloc[0:np_21.shape[0]-lag]
    temp= np_22[np_22.corr()['TAP4'][(np_22.corr()['TAP4']>0.3)|(np_22.corr()['TAP4']<-0.3)].keys()].dropna(axis=1)
    temp1=temp.dropna(axis=1)
    if 'index' in temp1.columns:
        temp2=temp1.drop(['index'],axis=1)
        temp3=temp2.drop(['TAP4'],axis=1)
    else:
        temp3=temp1.drop(['TAP4'],axis=1)
    x_columns =temp3.columns
    if len(x_columns)>1:
        stm.write(x_columns)
    return np_22,temp3,np_21,np_9,np_109,x_columns,lag,chemical
    
def Model_preparation(np_9,np_22,x_columns,lag,CI,CI_SI,chemical,Zone):
    np_27=np_9.tail(lag)[x_columns]
    x = np_22[x_columns].values
    y = np_22['TAP4'].values
    ##Data Preparation ends

    ##Train test split start and model finalisation
    min_max_scaler = preprocessing.StandardScaler()
    x_train, x_test, y_train, y_test = train_test_split(    
        x, y, test_size=0.25, random_state=42)

    x_train_normed = x_train
    x_test_normed = x_test
    
    #importances = [ np.exp(i/len(x_train_normed))-1 for i in np.arange(0,len(x_train_normed)) ]
    
    ##Normal RF
    name = 'Random Forest Regression'
    ##Iter1--OK for ZNO and insol
    if ((chemical =='ZNO' or 'INSOL SULFUR 20H' ) and (Zone== 'EUR'or 'ASI'or 'ADN'or 'ADS')) or ((chemical =='TMQ') and (Zone=='EUR'or 'ADN'or 'ADS')):
        regressor_11 = RandomForestRegressor(n_estimators =13 , n_jobs = -2, oob_score = True, criterion='mse',
                                      max_features = x_train.shape[1],  random_state = 0,max_leaf_nodes = 13,max_samples = 29
                                         )
    ##Iter2--for 'TMQ' and 'ASI'
    if (chemical =='TMQ') and (Zone=='ASI'):
        regressor_11 =RandomForestRegressor(n_estimators =13 , n_jobs = -2, oob_score = True, criterion='mse',
                                      max_features=x_train.shape[1],  random_state = 0,max_leaf_nodes = 20,max_samples = 3
          )
        #regressor_11=get_full_rbf_svm_clf_1(x_train_normed, y_train, n_estimators_range =[13,50,100], max_features_range = [10,12,14],max_leaf_nodes_range = [10,20,30],max_samples_range = [3,10,15])
    #max_features=8

    
    ##Normal RF
    ##Gridsearch CV
    
    
    
    ##Grid searchCV
    print(regressor_11)
    regressor_11.fit(x_train_normed, y_train)
    y_pred = regressor_11.predict(x_test_normed)
    print('{} Train Error: {}, Rsq: {}'.format(name, np.sqrt(mean_squared_error(y_train, np.array([regressor_11.predict(x_train_normed)]).T)),
                                              r2_score(y_train, np.array([regressor_11.predict(x_train_normed)]).T)))
    print('{} Test Error: {}, Rsq: {}'.format(name, np.sqrt(mean_squared_error(y_test, np.array([y_pred]).T)),#
                                     r2_score(y_test, np.array([y_pred]).T)))
    if CI_SI==True:
        CI_SI_1=CI_SI_2(x_train,y_train,np_27.values,2,regressor_11,lag,CI)
    else:
        CI_SI_1=0
    print('CI_SI_1',CI_SI_1)
    return regressor_11,CI_SI_1
    
def prediction(np_22,np_21,np_9,regressor_11,x_columns,lag):
##Prediction starts
    
    x = np_22[x_columns].values
    y = np_22['TAP4'].values
    
    regressor_11.fit(x, y)
    np_26=np_21.iloc[np_21.shape[0]-lag:np_21.shape[0]][x_columns]
    y_pred1 = regressor_11.predict(np_26.values)
    B=np_9['TOTAL AGREE PRICE'][np_9.shape[0]-lag:np_9.shape[0]].values
    A=y_pred1
    X=[]
    #upper_limit,lower_limit = A+CI_SI(x,y,x_test,2,regressor_11,lag,95),A-CI_SI(x_train,y_train,x_test,2,regressor_11,lag,95)
    for i in range(0,lag):
        X.append(np_9.reset_index()['Date'][np_9.shape[0]-1]-pd.DateOffset(months=i))
    result_2=pd.DataFrame()
    result_2['Date']=X
    result_2['TOTAL AGREE PRICE']=A
    #result_2['upper_limit']=upper_limit
    #result_2['lower_limit']=lower_limit
    result_2['prediction']=y_pred1
    result_3=result_2.sort_values(['Date'])
    ##Prediction ends
    return result_3
    
def forecasting(np_109,regressor_11,np_9,lag,x_columns,CI_SI_1,CI,CI_SI,CI_ML):
    ##Forecast begin
    x_for=np_109.iloc[0:np_109.shape[0]-lag].drop(['TAP4'],axis=1)[x_columns].values
    y_for=np_109.iloc[0:np_109.shape[0]-lag]['TAP4'].values
    regressor_11.fit(x_for, y_for)
    M=[]
    np_27=np_9.tail(lag)[x_columns]
    y_pred2=regressor_11.predict(np_27.values)
    for i in range(0,lag):
        #M.append(data_on_chem_1(data_rw,eco,'ZNO')['Date'][data_on_chem_1(data_rw,eco,'ZNO').shape[0]-1]+pd.DateOffset(months=i))
        M.append(np_9.reset_index()['Date'][np_9.shape[0]-1]+pd.DateOffset(months=i))
    np_28=pd.DataFrame()
    np_28['forecast']=y_pred2
    np_28['date']=M
    print('i am in forecasting',CI_SI_1)
    if CI_SI == True:
        upper_limit_1,lower_limit_1=np_28['forecast']+CI_SI_1,np_28['forecast']-CI_SI_1
        
        np_28['upper_limit_1']=upper_limit_1
        np_28['lower_limit_1']=lower_limit_1
        
    if CI_ML == True:
        yhat_upper,yhat_lower=CI_ML_1(regressor_11,np_109,np_9,lag,x_columns,CI)
        np_28['yhat_upper']=yhat_upper
        np_28['yhat_lower']=yhat_lower
    return np_28

def CI_ML_1(regressor_11,np_109,np_9,lag,x_columns,CI):
##Confidence Interval Calculation starts
    model = regressor_11
    np_110=np_109.iloc[0:np_109.shape[0]-lag]
    train = np_110
    np_27=np_9.tail(lag)[x_columns]
    new_input  = np_27##x variable of forecast

    X=np_109.iloc[0:np_109.shape[0]-lag].drop(['TAP4'],axis=1)[x_columns].values
    y=np_109.iloc[0:np_109.shape[0]-lag]['TAP4'].values

    importances = [ np.exp(i/len(train))-1 for i in np.arange(0,len(train)) ]

    model = model.fit(X,y, sample_weight=importances)
    scale=st.norm.ppf(CI/100)
    preds = []
    for estimator in model.estimators_:
        preds.append(estimator.predict(new_input))

    yhat = np.mean(preds, axis=0)
    yhat_upper = np.mean(preds, axis=0) + scale*np.std(preds, axis=0)
    yhat_lower = np.mean(preds, axis=0) - scale*np.std(preds, axis=0)
    return yhat_upper,yhat_lower
    
def MAPE(result_3,np_9,lag):
##MAPE Calculation Block
    y_pred=result_3['prediction']
    y_actual=np_9['TOTAL AGREE PRICE'][np_9.shape[0]-lag:np_9.shape[0]].values
    MAPE=(np.sum(np.abs(y_pred -y_actual))/np.sum(y_actual))*100
    ## MAPE calculation ends
    return MAPE
def plot_forecast(np_9,np_28,result_3,lag,Zone,chemical,CI_SI,CI_ML,CI_SI_1,x_columns):
    ##Plotting and visualisation block begins
 
    fig,ax1=plt.subplots(1,1)
    fig.set_figheight(12)
    fig.set_figwidth(18)
    ax1.plot(np_9['Date'], np_9['TOTAL AGREE PRICE'],label='Whole data', linestyle='--', marker='o', color='b')
    ax1.plot(np_9.iloc[np_9.shape[0]-lag:np_9.shape[0]]['Date'], np_9.iloc[np_9.shape[0]-lag:np_9.shape[0]]['TOTAL AGREE PRICE'],label='Actual data',linestyle='-', marker='o', color='g')
    ax1.plot(result_3['Date'], result_3['TOTAL AGREE PRICE'],label='prediction',marker='o', color='black')
    ax1.plot(np_28['date'],np_28['forecast'],label='forecast',marker='o', color='grey')
    if CI_ML==True:
        ax1.plot(np_28['date'],np_28['yhat_upper'], color='y')
        ax1.plot(np_28['date'],np_28['yhat_lower'], color='y')
        ax1.fill_between(np_28['date'], np_28['yhat_upper'], np_28['yhat_lower'],label='ML Confidence Interval')
    if CI_SI==True:
        ax1.plot(np_28['date'],np_28['upper_limit_1'], color='orange')
        ax1.plot(np_28['date'],np_28['lower_limit_1'], color='orange')
        ax1.fill_between(np_28['date'], np_28['upper_limit_1'], np_28['lower_limit_1'],label='Simple Confidence Interval')
    ax1.legend()
    ax1.set_title('For Zone  '+ Zone[0:3]+' '+' Price Of '+ chemical +'  vs time'+' with  MAPE'+'='+str(MAPE(result_3,np_9,lag)),fontsize=18)
    ax1.set_xlabel('Month')
    ax1.set_ylabel(chemical+'  Price')
    stm.pyplot(fig)
    A=MAPE(result_3,np_9,lag)
    ##Plotting and visualisation block ends
    return A
    
def CI_SI_2(x_train,y_train,x_test,cv,model,lag,CI):
    prediction = model.predict(x_test)
    plot_intervals=True 
    scale=st.norm.ppf(CI/100)
    if plot_intervals:
        cv = cross_val_score(model, x_train, y_train, 
                                    cv=cv, 
                                    scoring="neg_mean_squared_error")
        #mae = cv.mean() * (-1)
        deviation = np.sqrt(cv.std())
    return scale*deviation
    
def pred_forecast_CI_2(data_rw,eco,data,lag,Zone,chemical,CI,CI_SI=False,CI_ML=False):
    np_22,temp3,np_21,np_9,np_109,x_columns,lag,chemical=data_prepare(data_rw,eco,data_on_chem_1(data_rw,eco,chemical),lag,Zone,chemical)
    regressor_11,CI_SI_1=Model_preparation(np_9,np_22,x_columns,lag,CI,CI_SI,chemical,Zone)
    result_3=prediction(np_22,np_21,np_9,regressor_11,x_columns,lag)
    np_28=forecasting(np_109,regressor_11,np_9,lag,x_columns,CI_SI_1,CI,CI_SI,CI_ML)
    A=plot_forecast(np_9,np_28,result_3,lag,Zone,chemical,CI_SI,CI_ML,CI_SI_1,x_columns)
    return A

	
###NEW CODE ADDTION




def Data_prep_aggregrate(data_rw,eco,data,chemical,lag,correlation_values=0):
    #lag=np.arange(13)
    MAPE_df=pd.DataFrame()
    D=data_on_chem_1(data_rw,eco,'ZNO')['Zone'].unique()
    for i in range(len(D)):
        MAPE_df[D[i]]=0*lag
    lag1=lag
    D=data_on_chem_1(data_rw,eco,chemical)['Zone'].unique()
    data=data_on_chem_1(data_rw,eco,chemical)
    N=[]
    for i in range(0,lag):
        N.append(data_on_chem_1(data_rw,eco,chemical)['Date'][data_on_chem_1(data_rw,eco,chemical).shape[0]-1]+pd.DateOffset(months=i))
    A=data['Zone'].unique()
    frames=[]
    for i in range(len(A)):
        frames.append(data[(data['Zone']==A[i])])
    result = pd.concat(frames)
    L=[]
    M=[]
    #D=result['Zone'].unique()
    for i in range(len(A)):
        np_9=result[(result['Zone']==A[i])]
        np_99=np_9.reset_index().copy().iloc[0:np_9.shape[0]-lag]
        np_99['TAP4']=0*np_99.shape[0]
        for i in range(np_99.shape[0]-lag):
            np_99['TAP4'][i]=np_99['TOTAL AGREE PRICE'][i+lag]
        np_19=np_99.drop(['Date'],axis=1)
        #np_20=np_19.drop(['Zone'],axis=1)
        np_21=np_19.copy()
        for i in range(lag):
            np_21['TAP4'][np_21.shape[0]-lag+i]=np.nan




        np_22=np_21.iloc[0:np_21.shape[0]-lag]
        #temp= np_22[np_22.corr()['TAP4'][(np_22.corr()['TAP4']>0.5)|(np_22.corr()['TAP4']<-0.5)].keys()].dropna(axis=1)
        temp1=np_22.dropna(axis=1)
        if 'index' in temp1.columns:
            temp2=temp1.drop(['index'],axis=1)
            temp3=temp2.drop(['TAP4'],axis=1)
        else:
            temp3=temp1.drop(['TAP4'],axis=1)
        L.append(np_22)
        M.append(np_21)
    E=[]
    for i in range(len(D)):
        E.append(L[i])
    result1 = pd.concat(E)

    J=[]
    for i in range(len(D)):
        J.append(M[i])
    result_pred = pd.concat(J)
    S=[]
    G=[]
    for i in range(len(result['Zone'].unique())):  
        np_99=result[result['Zone']==result['Zone'].unique()[i]]
        np_109=np_99.reset_index().copy()
        np_109['TAP4']=0*np_99.shape[0]
        for i in range(np_109.shape[0]-lag):
            np_109['TAP4'][i]=np_109['TOTAL AGREE PRICE'][i+lag]
        S.append(np_109)

    for i in range(len(A)):
        G.append(S[i])
    result_for = pd.concat(G)

    #temp= result1[result1.corr()['TAP4'][(result1.corr()['TAP4']>correlation_values)|(result1.corr()['TAP4']<-correlation_values)].keys()].dropna(axis=1)
    temp3= result1[result1.corr()['TAP4'][(result1.corr()['TAP4']>0)|(result1.corr()['TAP4']<0)].keys()].dropna(axis=1)
    
    #if 'index' in temp3.columns:
    if 'index' in temp3.columns:
        x = temp3.drop(['TAP4','index'],axis=1).values
    else:
        x = temp3.drop(['TAP4'],axis=1).values
    y = temp3['TAP4'].values
    #print('x',x)
    #print('y',y)
    ##Train test split start and model finalisation
    min_max_scaler = preprocessing.StandardScaler()
    x_train, x_test, y_train, y_test = train_test_split(    
        x, y, test_size=0.25, random_state=42)



    #x_train_normed = min_max_scaler.fit_transform(x_train)
    #x_test_normed = min_max_scaler.fit_transform(x_test)
    
    x_train_normed = x_train
    x_test_normed = x_test


    name = 'Random Forest Regression'
    #if (chemical=='TBBS'):
    regressor_11 = RandomForestRegressor(n_estimators =13 , n_jobs = -2, oob_score = True, criterion='mse',
                                      max_features = x_train.shape[1],  random_state = 0,max_leaf_nodes = 13,max_samples = 23)
    regressor_11.fit(x_train_normed, y_train)
    #print('n_features',regressor_11.n_features)
    y_pred = regressor_11.predict(x_test_normed)
    print('{} Train Error: {}, Rsq: {}'.format(name, np.sqrt(mean_squared_error(y_train, np.array([regressor_11.predict(x_train_normed)]).T)),
                                              r2_score(y_train, np.array([regressor_11.predict(x_train_normed)]).T)))
    print('{} Test Error: {}, Rsq: {}'.format(name, np.sqrt(mean_squared_error(y_test, np.array([y_pred]).T)),
                                     r2_score(y_test, np.array([y_pred]).T)))
    CI_SI_1=CI_SI(x_train,y_train,x_test,2,regressor_11,lag,85)
    if 'index' in temp3.columns:
        x_columns=temp3.drop(['TAP4','index'],axis=1).columns
    else:
        x_columns=temp3.drop(['TAP4'],axis=1).columns
    
    
    
    
    print('x_columns in model',x_columns)
    
    
    return temp3,x_columns,regressor_11,CI_SI_1


# In[22]:


def data_prepare_1(data,lag,Zone,chemical):
    data=data_on_chem_1(data_rw,eco,chemical)
    np_9=data[(data['Zone']==Zone)]
    np_99=np_9.reset_index().copy().iloc[0:np_9.shape[0]-lag]
    np_99['TAP4']=0*np_99.shape[0]
    for i in range(np_99.shape[0]-lag):
        np_99['TAP4'][i]=np_99['TOTAL AGREE PRICE'][i+lag]
    np_109=np_9.reset_index().copy()
    np_109['TAP4']=0*np_9.shape[0]
    for i in range(np_109.shape[0]-lag):
        np_109['TAP4'][i]=np_109['TOTAL AGREE PRICE'][i+lag]
    np_19=np_99.drop(['Date'],axis=1)
    np_20=np_19.drop(['Zone'],axis=1)
    np_21=np_20.copy()
    for i in range(lag):
        np_21['TAP4'][np_21.shape[0]-lag+i]=np.nan
    np_22=np_21.iloc[0:np_21.shape[0]-lag]
    #temp= np_22[np_22.corr()['TAP4'][(np_22.corr()['TAP4']>0)|(np_22.corr()['TAP4']<-0)].keys()].dropna(axis=1)
    temp= np_22.dropna(axis=1)
    #temp=outlier_det(temp)##to be removed after iteration
    temp1=temp.dropna(axis=1)
    if 'index' in temp1.columns:
        temp2=temp1.drop(['index'],axis=1)
        temp3=temp2.drop(['TAP4'],axis=1)
    else:
        temp3=temp1.drop(['TAP4'],axis=1)
    x_columns =temp3.columns
    if len(x_columns)>1:
        stm.write(x_columns)
    return np_22,np_21,np_9,np_109,lag,chemical


# In[23]:


def prediction_1(np_22,np_21,np_9,regressor_11,x_columns,lag):
##Prediction starts
    
    #x = np_22[x_columns].values
    #y = np_22['TAP4'].values
    
    #regressor_11.fit(x, y)
    np_26=np_21.iloc[np_21.shape[0]-lag:np_21.shape[0]][x_columns]
    print('x_columns in prediction',x_columns)
    print('np_26.columns in prediction',np_26.columns)
    y_pred1 = regressor_11.predict(np_26)
    B=np_9['TOTAL AGREE PRICE'][np_9.shape[0]-lag:np_9.shape[0]].values
    A=y_pred1
    X=[]
    #upper_limit,lower_limit = A+CI_SI(x,y,x_test,2,regressor_11,lag,95),A-CI_SI(x_train,y_train,x_test,2,regressor_11,lag,95)
    for i in range(0,lag):
        X.append(np_9.reset_index()['Date'][np_9.shape[0]-1]-pd.DateOffset(months=i))
    result_2=pd.DataFrame()
    result_2['Date']=X
    result_2['TOTAL AGREE PRICE']=A
    #result_2['upper_limit']=upper_limit
    #result_2['lower_limit']=lower_limit
    result_2['prediction']=y_pred1
    result_3=result_2.sort_values(['Date'])
    ##Prediction ends
    return result_3
    


# In[24]:


def forecasting_1(np_109,regressor_11,np_9,lag,x_columns,CI_SI_1,CI,CI_SI =True,CI_ML=True):
    ##Forecast begin
    #x_for=np_109.iloc[0:np_109.shape[0]-lag].drop(['TAP4'],axis=1)[x_columns].values
    #y_for=np_109.iloc[0:np_109.shape[0]-lag]['TAP4'].values
    #regressor_11.fit(x_for, y_for)
    M=[]
    print('x_columns in forcasting',x_columns)
    #print('np_27.columns',np_27.columns)
    np_27=np_9.tail(lag)[x_columns]
    print('x_columns in forcasting',x_columns)
    y_pred2=regressor_11.predict(np_27.values)
    for i in range(0,lag):
        #M.append(data_on_chem_1(data_rw,eco,'ZNO')['Date'][data_on_chem_1(data_rw,eco,'ZNO').shape[0]-1]+pd.DateOffset(months=i))
        M.append(np_9.reset_index()['Date'][np_9.shape[0]-1]+pd.DateOffset(months=i))
    np_28=pd.DataFrame()
    np_28['forecast']=y_pred2
    np_28['date']=M
    print('i am in forecasting',CI_SI_1)
    if CI_SI == True:
        upper_limit_1,lower_limit_1=np_28['forecast']+CI_SI_1,np_28['forecast']-CI_SI_1
        
        np_28['upper_limit_1']=upper_limit_1
        np_28['lower_limit_1']=lower_limit_1
        
    if CI_ML == True:
        yhat_upper,yhat_lower=CI_ML_1(regressor_11,np_109,np_9,lag,x_columns,CI)
        np_28['yhat_upper']=yhat_upper
        np_28['yhat_lower']=yhat_lower
    return np_28


# In[25]:


def CI_SI(x_train,y_train,x_test,cv,model,lag,CI):
    prediction = model.predict(x_test)
    plot_intervals=True 
    scale=st.norm.ppf(CI/100)
    if plot_intervals:
        cv = cross_val_score(model, x_train, y_train, 
                                    cv=cv, 
                                    scoring="neg_mean_squared_error")
        #mae = cv.mean() * (-1)
        deviation = np.sqrt(cv.std())
    return scale*deviation


# In[26]:


def CI_ML_1(regressor_11,np_109,np_9,lag,x_columns,CI):
##Confidence Interval Calculation starts
    model = regressor_11
    np_110=np_109.iloc[0:np_109.shape[0]-lag]
    train = np_110
    np_27=np_9.tail(lag)[x_columns]
    new_input  = np_27##x variable of forecast

    X=np_109.iloc[0:np_109.shape[0]-lag].drop(['TAP4'],axis=1)[x_columns].values
    y=np_109.iloc[0:np_109.shape[0]-lag]['TAP4'].values

    importances = [ np.exp(i/len(train))-1 for i in np.arange(0,len(train)) ]

    model = model.fit(X,y, sample_weight=importances)
    scale=st.norm.ppf(CI/100)
    preds = []
    for estimator in model.estimators_:
        preds.append(estimator.predict(new_input))

    yhat = np.mean(preds, axis=0)
    yhat_upper = np.mean(preds, axis=0) + scale*np.std(preds, axis=0)
    yhat_lower = np.mean(preds, axis=0) - scale*np.std(preds, axis=0)
    return yhat_upper,yhat_lower
    


# In[27]:


def plot_forecast_1(np_9,np_28,result_3,lag,Zone,chemical,CI_SI,CI_ML,CI_SI_1,x_columns):
    ##Plotting and visualisation block begins
 
    fig,ax1=plt.subplots(1,1)
    fig.set_figheight(12)
    fig.set_figwidth(18)
    ax1.plot(np_9['Date'], np_9['TOTAL AGREE PRICE'],label='Whole data', linestyle='--', marker='o', color='b')
    ax1.plot(np_9.iloc[np_9.shape[0]-lag:np_9.shape[0]]['Date'], np_9.iloc[np_9.shape[0]-lag:np_9.shape[0]]['TOTAL AGREE PRICE'],label='Actual data',linestyle='-', marker='o', color='g')
    ax1.plot(result_3['Date'], result_3['TOTAL AGREE PRICE'],label='prediction',marker='o', color='black')
    ax1.plot(np_28['date'],np_28['forecast'],label='forecast',marker='o', color='grey')
    if CI_ML==True:
        ax1.plot(np_28['date'],np_28['yhat_upper'], color='y')
        ax1.plot(np_28['date'],np_28['yhat_lower'], color='y')
        ax1.fill_between(np_28['date'], np_28['yhat_upper'], np_28['yhat_lower'],label='ML Confidence Interval')
    if CI_SI==True:
        ax1.plot(np_28['date'],np_28['upper_limit_1'], color='orange')
        ax1.plot(np_28['date'],np_28['lower_limit_1'], color='orange')
        ax1.fill_between(np_28['date'], np_28['upper_limit_1'], np_28['lower_limit_1'],label='Simple Confidence Interval')
    
    
    y_pred=result_3['TOTAL AGREE PRICE'].values
    y_actual=(np_9.iloc[np_9.shape[0]-lag:np_9.shape[0]]['TOTAL AGREE PRICE']).values
    print('y_pred',y_pred)
    print('y_actual',y_actual)
    MAPE=(np.sum(np.abs(y_pred -y_actual))/np.sum(y_actual))*100
    ax1.legend()
    ax1.set_title('For Zone  '+ Zone[0:3]+' '+' Price Of '+ chemical +'  vs time'+' with  MAPE'+'='+str(MAPE),fontsize=18)
    ax1.set_xlabel('Month')
    ax1.set_ylabel(chemical+'  Price')
    stm.pyplot(fig)
    #A=MAPE(result_3,np_9,lag)
    A=MAPE
    ##Plotting and visualisation block ends
    return A


# In[28]:


def MAPE(result_3,np_9,lag):
##MAPE Calculation Block
    y_pred=result_3['prediction']
    y_actual=np_9['TOTAL AGREE PRICE'][np_9.shape[0]-lag:np_9.shape[0]].values
    MAPE=(np.sum(np.abs(y_pred -y_actual))/np.sum(y_actual))*100
    ## MAPE calculation ends
    return MAPE


# In[29]:


def pred_forecast_supplentey(data_rw,eco,data,lag,Zone,chemical,CI,CI_SI=False,CI_ML=False):
    np_22,np_21,np_9,np_109,lag,chemical=data_prepare_1(data_on_chem_1(data_rw,eco,chemical),lag,Zone,chemical)
    temp3,x_columns,regressor_11,CI_SI_1=Data_prep_aggregrate(chemical,lag,correlation_values=0)
    print(x_columns)
    result_3=prediction_1(np_22,np_21,np_9,regressor_11,x_columns,lag)
    #x_columns.delete('index')
    np_28=forecasting_1(np_109,regressor_11,np_9,lag,x_columns,CI_SI_1,CI,CI_SI,CI_ML)
    A=plot_forecast_1(np_9,np_28,result_3,lag,Zone,chemical,CI_SI,CI_ML,CI_SI_1,x_columns)
    return A	
	
	
	
###	NEW CODE ADDTION ends here


###Execution COde begins


def forecasting_overall_1(data_rw,eco,data,lag,Zone,chemical,CI,CI_SI,CI_ML):
    print(chemical,Zone=='ADS')
    print(Zone)
    #print(data)
    print(CI,CI_SI,CI_ML)
    print('iam in pred_forecast_CI_2 started ')
    if ((chemical=='TBBS') and (Zone=='EUR' or 'ASI')) or ((chemical=='6PPD') and (Zone=='ADS' or 'ASI')) or ((chemical=='CBS') and (Zone=='ADN' or 'EUR' or 'ASI')) or ((chemical=='COCL2 SOLUTION') and (Zone==('EUR' or 'ADS' or 'ADN'))):
        print('iam in pred_forecast_CI_2 ')
        pred_forecast_CI_2(data,lag,Zone,chemical,CI,CI_SI,CI_ML)
        print('iam in pred_forecast_CI_2 did that ')
    elif ((chemical=='TBBS') and (Zone=='ADN')) or ((chemical=='6PPD') and (Zone=='ADN' or 'EUR')):
        pred_forecast_supplentey(data,lag,Zone,chemical,CI,CI_SI,CI_ML)
    else:
        print('nothing has happened')

###Execution COde ends
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
if __name__ == '__main__':
    main()
