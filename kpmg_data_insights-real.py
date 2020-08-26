#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import math
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# ## OLD SPROCKET LTD PTY CUSTOMERS

# In[2]:


kpmg=pd.read_csv('kpmg_data.csv')
kpmg.head(8)


# In[3]:


#1 Checking the dimension of the kpmg dataframe
kpmg.shape


# In[4]:


#2 Checking the data structure and also if there is missing value in any row in the data frame
kpmg.info()


# In[5]:


#missing data
total = kpmg.isnull().sum().sort_values(ascending=False)
percent = (kpmg.isnull().sum()/kpmg.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


# In[6]:


kpmg['transaction_date']=pd.to_datetime(kpmg['transaction_date'])


# In[7]:


kpmg.order_status=kpmg.order_status.astype('category')


# In[8]:


kpmg.brand=kpmg.brand.astype('category')


# In[9]:


kpmg.product_line=kpmg.product_line.astype('category')


# In[10]:


kpmg.product_class=kpmg.product_class.astype('category')


# In[11]:


kpmg.product_size=kpmg.product_size.astype('category')


# In[12]:


kpmg.gender=kpmg.gender.astype('category')


# In[13]:


kpmg['product_first_sold_date']=pd.to_datetime(kpmg['product_first_sold_date'])


# In[14]:


kpmg.job_title=kpmg.job_title.astype('category')


# In[15]:


kpmg.job_industry_category=kpmg.job_industry_category.astype('category')


# In[16]:


kpmg.wealth_segment=kpmg.wealth_segment.astype('category')


# In[17]:


kpmg.owns_car=kpmg.owns_car.astype('category')


# In[18]:


kpmg.dtypes


# In[19]:


kpmg.describe()


# In[20]:


#I am converting numerical variable to categorical variable using pandas' cut function
kpmg['ageGroup']=pd.cut(kpmg.age,[18,25,35,60,125],labels=['young adults', 'adults', 'middle aged', 'old adults'])
kpmg['ageGroup'].head()


# In[21]:


#Checking the overall counts of customers age categorising them into age groups in the dataset
pd.DataFrame(kpmg.ageGroup.value_counts())


# In[22]:


#checking histogram of entite kpmg dataset
kpmg.hist(figsize=(10,8));


# **BAR CHART OF QUALITATIVE VARIABLE IN THE DATASET**

# In[23]:


#Checking the overall counts of customers age categorising them into age groups in the dataset
pd.DataFrame(kpmg.ageGroup.value_counts())


# In[24]:


kpmg.ageGroup=kpmg.ageGroup.astype('category')


# In[25]:


#Plotting the age Group on a bar chart
plt.figure(figsize=(10,4))
plt.subplot(1,2,1);
kpmg.ageGroup.value_counts().plot(kind='bar',color=['C4','C5','C6','C7']);
plt.title('Age group of customers',fontsize=18)
plt.xlabel('ageGoup',fontsize=18)
plt.ylabel('customers counts',fontsize=18);

plt.subplot(1,2,2);
ageGroup_values=[5848,5039,3201,2738]
ageGroup_labels=["middle aged","young adults","adults","old adults"]
plt.axis("equal")
plt.title("Pie chart showing the customers ageGroup")
plt.pie(ageGroup_values,labels=ageGroup_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0.1,0.1,0],wedgeprops={'edgecolor':'black'});


# In[26]:


#Checking number of customers wealth in the kpmg dataset
pd.DataFrame(kpmg.owns_car.value_counts())


# In[27]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg.owns_car.value_counts().plot(kind='bar',title='Customers online order',color=['C0','C1']);
plt.xlabel('owns car')
plt.ylabel('Counts');

plt.subplot(1,2,2)
order_values=[10969,5857]
order_labels=["Yes","No"]
plt.axis("equal")
plt.title("Pie chart showing the customer's who have a car or not")
plt.pie(order_values,labels=order_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0.1],wedgeprops={'edgecolor':'black'});


# In[28]:


#Checking number of customers that made an online order in the kpmg dataset
pd.DataFrame(kpmg.online_order.value_counts())


# In[29]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg.online_order.value_counts().plot(kind='bar',title='Customers online order',color=['C0','C1']);
plt.xlabel('online order')
plt.ylabel('Counts');

plt.subplot(1,2,2)
order_values=[8422,8404]
order_labels=["False","True"]
plt.axis("equal")
plt.title("Pie chart showing the customer's online order counts")
plt.pie(order_values,labels=order_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0.1],wedgeprops={'edgecolor':'black'});


# In[30]:


#Checking number of customers that made have their order approved or not in the kpmg dataset
pd.DataFrame(kpmg.order_status.value_counts())


# In[31]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg.order_status.value_counts().plot(kind='bar',title='Customers order status',color=['C0','C1']);
plt.xlabel('order status')
plt.ylabel('Counts');

plt.subplot(1,2,2)
order_values=[16673,153]
order_labels=["Approved","Cancelled"]
plt.axis("equal")
plt.title("Pie chart showing the customer's order status")
plt.pie(order_values,labels=order_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0.1],wedgeprops={'edgecolor':'black'});


# In[32]:


#Checking number of customers that made an online order in the kpmg dataset
pd.DataFrame(kpmg.brand.value_counts())


# In[33]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg.brand.value_counts().plot(kind='bar',title='Customers brand',color=['green']);
plt.xlabel('brand')
plt.ylabel('Counts');

plt.subplot(1,2,2)
brand_values=[3607,2834,2814,2586,2533,2452]
brand_labels=["Solex","WeareA2B","Giant Bicycles","OHM Cycles","Trek Bicycles","Norco Bicycles"]
plt.axis("equal")
plt.title("Pie chart showing the customer's brand")
plt.pie(brand_values,labels=brand_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0.1,0,0,0,0.1],wedgeprops={'edgecolor':'black'});


# In[34]:


#Checking number of customers product line in the kpmg dataset
pd.DataFrame(kpmg.product_line.value_counts())


# In[35]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg.product_line.value_counts().plot(kind='bar',title='Customers product line',color=['blue']);
plt.xlabel('product line')
plt.ylabel('Counts');

plt.subplot(1,2,2)
pline_values=[12045,3350,1062,369]
pline_labels=["Standard","Road","Touring","Mountain"]
plt.axis("equal")
plt.title("Pie chart showing the customer's product line")
plt.pie(pline_values,labels=pline_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0,0.1],wedgeprops={'edgecolor':'black'});


# In[36]:


#Checking number of customers product class in the kpmg dataset
pd.DataFrame(kpmg.product_class.value_counts())


# In[37]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg.product_class.value_counts().plot(kind='bar',title='Customers product class',color=['C0','C1','C2']);
plt.xlabel('product class')
plt.ylabel('Counts');

plt.subplot(1,2,2)
pclass_values=[11747,2554,2525]
pclass_labels=["medium","high","low"]
plt.axis("equal")
plt.title("Pie chart showing the customer's product class")
plt.pie(pclass_values,labels=pclass_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0.1],wedgeprops={'edgecolor':'black'});


# In[38]:


#Checking number of customers product size in the kpmg dataset
pd.DataFrame(kpmg.product_size.value_counts())


# In[39]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg.product_size.value_counts().plot(kind='bar',title='Customers product size',color=['C0','C1','C2']);
plt.xlabel('product size')
plt.ylabel('Counts');

plt.subplot(1,2,2)
psize_values=[11057,3378,2391]
psize_labels=["medium","high","low"]
plt.axis("equal")
plt.title("Pie chart showing the customer's product size")
plt.pie(psize_values,labels=psize_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0.1],wedgeprops={'edgecolor':'black'});


# In[40]:


#Checking number of customers gender in the kpmg dataset
pd.DataFrame(kpmg.gender.value_counts())


# In[41]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg.gender.value_counts().plot(kind='bar',title='Customers product size',color=['C0','C1','C2']);
plt.xlabel('customer gender')
plt.ylabel('Counts');

plt.subplot(1,2,2)
psize_values=[16022,801,3]
psize_labels=["Female","Male","Uncategorized Gender"]
plt.axis("equal")
plt.title("Pie chart showing the customer's gender")
plt.pie(psize_values,labels=psize_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0.1],wedgeprops={'edgecolor':'black'});


# In[42]:


#Checking number of customers jobs in the kpmg dataset
pd.DataFrame(kpmg.job_title.value_counts())


# In[43]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg.job_title.value_counts().plot(kind='bar',title='Customers Job',color=['red']);
plt.xlabel('job title')
plt.ylabel('Counts');

plt.subplot(1,2,2)
jt_values=[5840,5039,1933,1694,1501,793,9,8,6,3]
jt_labels=["Research associate","food chemist","executive secretary","PA cord","IA","operator","recruiter","SP","HR Ass","AP"]
plt.axis("equal")
plt.title("Pie chart showing the customer's job title")
plt.pie(jt_values,labels=jt_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0.1,0,0,0.1,0,0,0,0.1],wedgeprops={'edgecolor':'black'});


# In[44]:


#Checking number of customers industry in the kpmg dataset
pd.DataFrame(kpmg.job_industry_category.value_counts())


# In[45]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg.job_industry_category.value_counts().plot(kind='bar',title='Customers Job Category',color=['red']);
plt.xlabel('job industry category')
plt.ylabel('Counts');

plt.subplot(1,2,2)
ji_values=[7534,6972,1515,793,9,3]
ji_labels=["Manufacturing","Health","Financial services","Agriculture","Property","IT"]
plt.axis("equal")
plt.title("Pie chart showing the customer's Job category")
plt.pie(ji_values,labels=ji_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0.1,0,0,0.1],wedgeprops={'edgecolor':'black'});


# In[46]:


#Checking number of customers wealth in the kpmg dataset
pd.DataFrame(kpmg.wealth_segment.value_counts())


# In[47]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg.wealth_segment.value_counts().plot(kind='bar',title='Customers wealth segment',color=['C0','C1','C2']);
plt.xlabel('customer wealth segment')
plt.ylabel('Counts');

plt.subplot(1,2,2)
wealth_values=[9266,7551,9]
wealth_labels=["Mass Customer","High Net Worth","Affluent Customer"]
plt.axis("equal")
plt.title("Pie chart showing the customer's wealth segment")
plt.pie(wealth_values,labels=wealth_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0.1],wedgeprops={'edgecolor':'black'});


# ## NEW SPROCKET LTD PTY CUSTOMERS

# In[48]:


kpmg_n=pd.read_csv('new_cust.csv')
kpmg_n.head(8)


# In[49]:


kpmg_n = kpmg_n.loc[:, ~kpmg_n.columns.str.contains('^Unnamed')]


# In[50]:


#1 Checking the dimension of the kpmg new customer dataframe
kpmg_n.shape


# In[51]:


#2 Checking the data structure and also if there is missing value in any row in the data frame
kpmg_n.info()


# In[52]:


kpmg_n.dtypes


# In[53]:


#missing data
total2 = kpmg_n.isnull().sum().sort_values(ascending=False)
percent2 = (kpmg_n.isnull().sum()/kpmg_n.isnull().count()).sort_values(ascending=False)
missing_data2 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])
missing_data2


# In[54]:


#Fillin in the missing value and rechecking the dataset info
kpmg_n.fillna(kpmg_n.mean(),inplace=True)
kpmg_n.info()


# In[55]:


kpmg_n.gender=kpmg_n.gender.astype('category')


# In[56]:


kpmg_n.job_title=kpmg_n.job_title.astype('category')


# In[57]:


kpmg_n.job_industry_category=kpmg_n.job_industry_category.astype('category')


# In[58]:


kpmg_n.wealth_segment=kpmg_n.wealth_segment.astype('category')


# In[59]:


kpmg_n.owns_car=kpmg_n.owns_car.astype('bool')


# In[60]:


kpmg_n.dtypes


# In[61]:


kpmg_n.describe()


# In[62]:


#checking histogram of entite kpmg dataset
kpmg_n.hist(figsize=(10,8));


# In[63]:


#I am converting numerical variable to categorical variable using pandas' cut function
kpmg_n['ageGroup']=pd.cut(kpmg_n.age,[18,25,35,60,125],labels=['young adults', 'adults', 'middle aged', 'old adults'])
kpmg_n['ageGroup'].head()


# **BAR CHART OF QUALITATIVE VARIABLE IN THE KPMG NEW CUSTOMER DATASET**

# In[64]:


#Checking the overall counts of customers age categorising them into age groups in the dataset
pd.DataFrame(kpmg_n.ageGroup.value_counts())


# In[65]:


#Plotting the age Group on a bar chart
plt.figure(figsize=(10,4))
plt.subplot(1,2,1);
kpmg_n.ageGroup.value_counts().plot(kind='bar',color=['C4','C5','C6','C7']);
plt.title('Age group of new customer',fontsize=18)
plt.xlabel('ageGoup',fontsize=18)
plt.ylabel('customers counts',fontsize=18);

plt.subplot(1,2,2);
ageGroup_values=[493,286,124,97]
ageGroup_labels=["middle aged","young adults","adults","old adults"]
plt.axis("equal")
plt.title("Pie chart showing the new customers ageGroup")
plt.pie(ageGroup_values,labels=ageGroup_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0.1,0.1,0],wedgeprops={'edgecolor':'black'});


# In[66]:


#Checking number of customers that made an online order in the kpmg dataset
pd.DataFrame(kpmg_n.gender.value_counts())


# In[67]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg_n.gender.value_counts().plot(kind='bar',title='New Customers product size',color=['C0','C1','C2']);
plt.xlabel('new customer gender')
plt.ylabel('Counts');

plt.subplot(1,2,2)
gen_values=[513,470,17]
gen_labels=["Female","Male","Uncategorized Gender"]
plt.axis("equal")
plt.title("Pie chart showing the new customer's gender")
plt.pie(gen_values,labels=gen_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0.1],wedgeprops={'edgecolor':'black'});


# In[68]:


#Checking number of customers industry in the kpmg dataset
pd.DataFrame(kpmg_n.job_industry_category.value_counts())


# In[69]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg_n.job_industry_category.value_counts().plot(kind='bar',title='New Customers Job Category',color=['red']);
plt.xlabel('new job industry category')
plt.ylabel('Counts');

plt.subplot(1,2,2)
ji_values=[203,199,152,78,64,51,37,26,25]
ji_labels=["Financial services","Manufacturing","Health","Retail","Property","IT","Entertainment","Agriculture","Telecom"]
plt.axis("equal")
plt.title("Pie chart showing the new customer's Job category")
plt.pie(ji_values,labels=ji_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0.1,0,0,0.1,0,0,0.1],wedgeprops={'edgecolor':'black'});


# In[70]:


#Checking number of customers wealth in the kpmg dataset
pd.DataFrame(kpmg_n.wealth_segment.value_counts())


# In[71]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg_n.wealth_segment.value_counts().plot(kind='bar',title='New Customers wealth segment',color=['C0','C1','C2']);
plt.xlabel('new customer wealth segment')
plt.ylabel('Counts');

plt.subplot(1,2,2)
wealth_values=[508,251,241]
wealth_labels=["Mass Customer","High Net Worth","Affluent Customer"]
plt.axis("equal")
plt.title("Pie chart showing the new customer's wealth segment")
plt.pie(wealth_values,labels=wealth_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0.1],wedgeprops={'edgecolor':'black'});


# In[72]:


#Checking number of customers wealth in the kpmg dataset
pd.DataFrame(kpmg_n.owns_car.value_counts())


# In[73]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg_n.owns_car.value_counts().plot(kind='bar',title='New Customers online order',color=['C0','C1']);
plt.xlabel('owns car')
plt.ylabel('Counts');


# ## OLD CUSTOMERS VS NEW CUSTOMERS
# 
# 
# **Showing relationships between varibales in the KPMG new and old customers dataset**

# **(1) Gender vs Bike related purchase for the past 3 years**

# In[74]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1);
ax = sb.barplot(x="gender", y="past_3_years_bike_related_purchases", data=kpmg, estimator=lambda x: len(x) / len(kpmg) * 100)
ax.set(ylabel="Percent");

plt.subplot(1,2,2);
ax = sb.barplot(x="gender", y="past_3_years_bike_related_purchases", data=kpmg_n, estimator=lambda x: len(x) / len(kpmg_n) * 100)
ax.set(ylabel="Percent");


# **(2) Age Group Distribution**

# In[75]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1);
ageGroup_values=[493,286,124,97]
ageGroup_labels=["middle aged","young adults","adults","old adults"]
plt.axis("equal")
plt.title("Pie chart showing the new customers ageGroup")
plt.pie(ageGroup_values,labels=ageGroup_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0.1,0.1,0],wedgeprops={'edgecolor':'black'});

plt.subplot(1,2,2);
ageGroup_values=[5848,5039,3201,2738]
ageGroup_labels=["middle aged","young adults","adults","old adults"]
plt.axis("equal")
plt.title("Pie chart showing the customers ageGroup")
plt.pie(ageGroup_values,labels=ageGroup_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0.1,0.1,0],wedgeprops={'edgecolor':'black'});


# **(3) JOB INDUSTRY DISTRIBUTION**

# In[76]:


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
kpmg_n.job_industry_category.value_counts().plot(kind='bar',title='New Customers Job Category',color=['red']);
plt.xlabel('new job industry category')
plt.ylabel('Counts');


plt.subplot(1,2,2)
kpmg.job_industry_category.value_counts().plot(kind='bar',title='Customers Job Category',color=['red']);
plt.xlabel('job industry category')
plt.ylabel('Counts');


# In[77]:


#New customers job category
val=[203,199,152,78,64,51,37,26,25]
col=(0.2,0.3,0.4,0.5)
x=np.arange(9)
fig,ax=plt.subplots()
ax.set_ylabel('Number of people')
ax.set_title('NEW CUSTOMERS JOB CATEGORY')
plt.bar(x,val,color=col,width=0.5)

for i in range(len(val)):
    plt.text(x = i-0.25, y=val[i]+0.1, s=val[i], size = 10)
    
    plt.xticks(x,("Finance","Man","Health","Retail","Pro","IT","Enter","Agric","Telecom"))
    
    


# In[78]:


val=[7534,6972,1515,793,9,3]
col=(0.2,0.3,0.4,0.5)
x2=np.arange(6)
fig2,ax2=plt.subplots()
ax2.set_ylabel('Number of people')
ax2.set_title('OLD CUSTOMERS JOB CATEGORY')
plt.bar(x2,val,color=col,width=0.5)

for i in range(len(val)):
    plt.text(x = i-0.25, y=val[i]+0.1, s=val[i], size = 10)
    
    plt.xticks(x2,("Manu","Health","Finance","Agric","Property","IT"))
    


# In[79]:


plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
jin_values=[203,199,152,78,64,51,37,26,25]
jin_labels=["Financial services","Manufacturing","Health","Retail","Property","IT","Entertainment","Agriculture","Telecom"]

plt.axis("equal")
plt.title("Pie chart showing the new customer's Job category")
plt.pie(jin_values,labels=jin_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0.1,0,0,0.1,0,0,0.1],wedgeprops={'edgecolor':'black'});


plt.subplot(1,2,2)
jio_values=[7534,6972,1515,793,9,3]
jio_labels=["Manufacturing","Health","Financial services","Agriculture","Property","IT"]
plt.axis("equal")
plt.title("Pie chart showing the old customer's Job category")
plt.pie(jio_values,labels=jio_labels,radius=1.0,autopct='%0.1f%%',shadow=True,explode=[0,0,0.1,0,0,0.1],wedgeprops={'edgecolor':'black'});


# **(4) Numbers of cars owned and not owned by state**

# In[80]:


byc=kpmg.groupby("state").owns_car.value_counts()
byc


# In[81]:


#plotting stacked bar chart of state vs own cars by old customers
byc.unstack().plot(kind='bar',stacked=True);
plt.title('Number of cars owned and not owned in each state',fontsize=18)
plt.xlabel('state',fontsize=18)
plt.ylabel('customers',fontsize=18);



# In[82]:


byd=kpmg_n.groupby("state").owns_car.value_counts()
byd


# In[83]:


byd.unstack().plot(kind='bar',stacked=True);
plt.title('Number of cars owned and not owned in each state',fontsize=18)
plt.xlabel('state',fontsize=18)
plt.ylabel('customers',fontsize=18);


# **(5) Wealth segment by Age category**

# In[84]:


byy=kpmg.groupby("wealth_segment").ageGroup.value_counts()
byy


# In[85]:


#plotting stacked bar chart of state vs own cars by old customers
byy.unstack().plot(kind='bar',stacked=True);
plt.title('Wealth Segmentation By Age category',fontsize=18)
plt.xlabel('wealth segmentation',fontsize=18)
plt.ylabel('customers',fontsize=18);



# In[86]:


byz=kpmg_n.groupby("wealth_segment").ageGroup.value_counts()
byz


# In[87]:


#plotting stacked bar chart of state vs own cars by old customers
byz.unstack().plot(kind='bar',stacked=True);
plt.title('Wealth Segmentation By Age category',fontsize=18)
plt.xlabel('wealth segmentation',fontsize=18)
plt.ylabel('customers',fontsize=18)


# ## MODEL DEVELOPMENT

# **(1) Using the full old customer for the model**

# In[88]:


kpmg.drop(['transaction_id'],axis=1,inplace=True)
kpmg.dtypes


# In[89]:


kpmg.head(5)


# In[90]:


gen=pd.get_dummies (kpmg["gender"],drop_first=True)
gen.head(2)


# In[91]:


order=pd.get_dummies (kpmg["order_status"],drop_first=True)
order.head(2)


# In[92]:


brand=pd.get_dummies (kpmg["brand"],drop_first=True)
brand.head(2)


# In[93]:


pl=pd.get_dummies (kpmg["product_line"],drop_first=True)
pl.head(2)


# In[94]:


pc=pd.get_dummies (kpmg["product_class"],drop_first=True)
pc.head(2)


# In[95]:


ps=pd.get_dummies (kpmg["product_size"],drop_first=True)
ps.head(2)


# In[96]:


age=pd.get_dummies (kpmg["ageGroup"],drop_first=True)
age.head(2)


# In[97]:


jobt=pd.get_dummies (kpmg["job_title"],drop_first=True)
jobt.head(2)


# In[98]:


jic=pd.get_dummies (kpmg["job_industry_category"],drop_first=True)
jic.head(2)


# In[99]:


wealth=pd.get_dummies (kpmg["wealth_segment"],drop_first=True)
wealth.head(2)


# In[100]:


kpmg=pd.concat([kpmg,gen,order,brand,pl,pc,ps,age,jobt,jic,wealth],axis=1)
kpmg.head(4)


# In[101]:


kpmg.drop(['product_id','customer_id','transaction_date','transaction_month','product_first_sold_date','dob','age','deceased_indicator','owns_car','address','postcode','state','country'],axis=1,inplace=True)


# In[102]:


kpmg.head(3)


# In[103]:


kpmg.dtypes


# In[104]:


##Drop the initial columns of the dummy variable
kpmg.drop(['gender','order_status','brand','product_line','product_class','product_size','ageGroup','job_title','job_industry_category','wealth_segment'],axis=1,inplace=True)


# In[105]:


X=kpmg.drop("online_order",axis=1)

y=kpmg['online_order']


# In[106]:


from sklearn.cross_validation import train_test_split


# In[107]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[108]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# In[109]:


#checking the logistic model coefficient
result.params


# In[110]:


#Exponentiate the coefficients to get the log odds
np.exp(result.params)


# In[111]:


#Checking if variables are significant with their pvalues
result.pvalues     


# **The p-values for most of the variables are smaller than 0.05, except four variables, therefore, we will remove them.**

# In[112]:


# odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
np.exp(conf)


# In[113]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:





# **Predicting the test set results and calculating the accuracy**

# In[114]:


y_pred = logmodel.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logmodel.score(X_test, y_test)))


# **Confusion Matrix**

# In[115]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[116]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sb.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# **The result is telling us that we have 1368+1144 correct predictions and 1161+1375 incorrect predictions.**

# **Compute precision, recall, F-measure and support**

# In[117]:


# import the metrics class
from sklearn import metrics


# In[118]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[119]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# **ROC Curve**

# In[120]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logmodel.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logmodel.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).

# In[ ]:




