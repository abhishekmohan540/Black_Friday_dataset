#!/usr/bin/env python
# coding: utf-8

# # Black friday dataset EDA and feature engineering

#     Problem Statement:-
#     A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
# 
#     Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df_train = pd.read_csv(r"E:\Black_Friday\train.csv")


# In[4]:


df_train.head()


# In[5]:


df_train.shape


# In[6]:


df_test = pd.read_csv(r"E:\Black_Friday\test.csv")


# In[7]:


df_test.head()


# In[8]:


df_test.shape


# In[9]:


# Merge both the data sets
df = pd.concat([df_train, df_test], axis = 0)
df


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


# Missing values
df.isnull().sum()


# In[13]:


# Duplicate value
df[df.duplicated()==True].count()


# In[14]:


df[df.duplicated()]


# In[15]:


# There are no duplicated rows present


# In[16]:


df.head()


# In[17]:


# USer id is of no use for predicting purchase, do drop it


# In[18]:


df.drop("User_ID", axis=1, inplace=True)


# In[19]:


df.head()


# In[20]:


# Handle categoriocal feature 


# In[21]:


## Gender
df['Gender'] = df['Gender'].map({'F':0,'M':1})


# In[22]:


df.head()


# In[23]:


df["Age"].unique()


# In[24]:


df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})


# In[25]:


df.head()


# In[26]:


df['Occupation'].unique()


# In[27]:


df['Occupation'].isnull().sum()


# In[28]:


# City_Category
df["City_Category"].unique()


# In[29]:


from sklearn.preprocessing import OneHotEncoder


# In[30]:


encoder = OneHotEncoder(sparse=False)


# In[31]:


encoder_new= encoder.fit_transform(df[['City_Category']])


# In[32]:


city_category  = pd.DataFrame(data=encoder_new, columns=['A','B','C'])


# In[33]:


city_category


# In[34]:


df.shape


# In[35]:


pd.concat([df,city_category], axis=1)


# In[36]:


df = df.reset_index(drop=True)
df


# In[37]:


df=pd.concat([df,city_category], axis=1)
df


# In[38]:


df.drop('City_Category', axis=1, inplace=True)


# In[39]:


df


# In[40]:


# Stay_In_Current_city
df['Stay_In_Current_City_Years'].unique()


# In[41]:


df.dtypes


# In[42]:


df['Stay_In_Current_City_Years']= df['Stay_In_Current_City_Years'].str.replace("+","")


# In[43]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)


# In[44]:


df['Stay_In_Current_City_Years'].unique()


# In[45]:


df.info()


# In[46]:


df.head()


# In[47]:


df['Marital_Status'].unique()


# In[48]:


#handling Missing Values


# In[49]:


df.isnull().sum()


# In[50]:


# Missing values are present in product category 2 and 3. In Purchase it is of test data


# In[51]:


df['Product_Category_2'].unique()


# In[52]:


df['Product_Category_2'].value_counts()


# In[53]:


# Since product category is discrete value, it is not continuous value, hence mean and median not to be used. we will use 
# mode to fill nan values.


# In[59]:


df['Product_Category_2'].mode()[0]


# In[55]:


df['Product_Category_2']= df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[56]:


df['Product_Category_2'].isnull().sum()


# In[57]:


df['Product_Category_3'].isnull().sum()


# In[58]:


df['Product_Category_3'].mode()[0]


# In[60]:


df['Product_Category_3']= df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[61]:


df['Product_Category_3'].isnull().sum()


# In[62]:


df.head()


# In[63]:


df.isnull().sum()


# In[64]:


df['Product_ID'].count()


# In[65]:


df['Product_ID'].value_counts()


# In[66]:


df.drop('Product_ID', axis=1, inplace=True)


# In[67]:


df.head()


# In[69]:


df.dtypes


# ## Visualization

# In[72]:


sns.barplot(x='Age',y='Purchase', data=df)


# In[76]:


# People from age 51-55 made highest purchases


# In[73]:


sns.barplot(x='Age',y='Purchase', data=df, hue='Gender')


# In[74]:


sns.barplot(x='Gender',y='Purchase', data=df)


# In[75]:


# Men purchase more than women


# In[77]:


sns.barplot(x='Occupation',y='Purchase', data=df)


# In[80]:


# Product category-1 vs Purchase
sns.barplot(x='Product_Category_1',y='Purchase', data=df)


# In[81]:


# Product category-2 vs Purchase
sns.barplot(x='Product_Category_2',y='Purchase', data=df)


# In[82]:


# Product category-3 vs Purchase
sns.barplot(x='Product_Category_3',y='Purchase', data=df)


# ## Feature Scaling

# In[83]:


df.head()


# In[98]:


df.shape


# In[99]:


df_test_data = df[df['Purchase'].isnull()]
df_test_data


# In[100]:


df_train_data = df[~df['Purchase'].isnull()]
df_train_data


# In[102]:


X = df_train_data.drop('Purchase', axis=1)
X.head()


# In[103]:


X.shape


# In[104]:


y= df_train_data[['Purchase']]
y.head()


# In[105]:


X.shape


# In[106]:


y.shape


# In[107]:


# Train test split
from sklearn.model_selection import train_test_split


# In[108]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[109]:


X_train.shape


# In[110]:


X_test.shape


# In[112]:


y_train.shape


# In[115]:


X_train


# In[116]:


X_test


# In[133]:


## feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[134]:


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[135]:


X_train


# In[136]:


X_test


# ## Model Training

# In[143]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# In[144]:


import numpy as np
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, mse, rmse, r2_square


# In[145]:


models = [LinearRegression(),Lasso(),Ridge(),ElasticNet(), RandomForestRegressor()]

for model in models:
    our_model = model
    our_model.fit(X_train,y_train)
    y_pred = our_model.predict(X_test)
    mae, mse, rmse, r2_square = evaluate_model(y_test, y_pred)
    print(our_model)
    print("Model Performance")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("R2 Score: ", r2_square*100)
    print("=============================================")
    print("\n")
    


# In[ ]:




