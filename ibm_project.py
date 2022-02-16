import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from functions import train_model

st.set_page_config(
    page_title="IBM Attrition analysis",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header('IBM Attrition Analysis')

csv = r'C:\Users\asus\Desktop\IBM.csv'
ibm_df = pd.read_csv(csv, sep=',')
st.download_button(
     label="Download data as CSV",
     data=csv,
     file_name='IBM.csv',
     mime='text/csv',
 )

st.write("")
st.write('''**IBM** is an American multinational technology corporation, with operations in over 171 countries. 
IBM produces and sells computer hardware, middleware and software, and provides hosting and consulting services in areas ranging from mainframe computers to nanotechnology''')
st.write('''What will be done here is to build **classification model(s) to predict the attrition of employees** in a service-providing organization where trained and experienced people are important assets.''')

st.sidebar.subheader('Controls')
st.sidebar.download_button('Download data as CSV', csv, file_name='ibm_df.csv')
data_dictionary = st.sidebar.checkbox('Data dictionary')
show_raw_data = st.sidebar.checkbox('Show raw data')
st.text_area('Attrition',
value='''The term attrition refers to a gradual but deliberate reduction in staff numbers that occurs as employees retire or resign and are not replaced. It is commonly used to describe downsizing in a firm's employee pool by human resources (HR) professionals.''')

if data_dictionary:
    st.subheader('Data dictionary')
    st.markdown(
        """
        * **Age**: Age of employee
        * **Attrition**: Employee attrition status
        * **Department**: Department of work
        * **DistanceFromHome**
        * **Education**: 1-Below College; 2- College; 3-Bachelor; 4-Master; 5-Doctor;
        * **EducationField**
        * **EnvironmentSatisfaction**: 1-Low; 2-Medium; 3-High; 4-Very High;
        * **JobSatisfaction**: 1-Low; 2-Medium; 3-High; 4-Very High;
        * **MaritalStatus**
        * **MonthlyIncome**
        * **NumCompaniesWorked:** Number of companies worked prior to IBM
        * **WorkLifeBalance**: 1-Bad; 2-Good; 3-Better; 4-Best;
        * **YearsAtCompany**: Current years of service in IBM'
        """)

if show_raw_data:
    st.subheader('Raw data')
    st.write(ibm_df)

st.write(ibm_df.info())

null_value = ibm_df.isnull().values.any()
st.write('Is there any null value? : ', str(null_value))

st.subheader('What is the Attrition and Age distribution?')
col_1, col_2 = st.columns(2)
with col_1:
    fig, ax = plt.subplots(figsize=(10, 6))
    my_explode = (0, 0.1)
    plt.pie(ibm_df['Attrition'].value_counts(), labels=ibm_df['Attrition'].value_counts().index, shadow=True,
            autopct='%1.2f%%', explode=my_explode)
    st.pyplot(fig)
    st.caption('Attrition')

with col_2:
    fig, ax = plt.subplots(figsize=(10, 6))
    ibm_df.Age.hist(ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')

    st.pyplot(fig)
    st.caption('Ages distribution')

#Discretizing Age in 4 classes and trasforming Attrition and MaritalStatus values
my_labels = ['18_30', '31_36', '37_43', '44_60']
ibm_df['Age'] = pd.cut(ibm_df['Age'], bins=[17, 30, 36,	43, 60], labels=my_labels)

ageclass = ibm_df.Age.replace({'18_30': 1, '31_36': 2, '37_43': 3, '44_60': 4})
ibm_df['AgeClass'] = ageclass
ibm_df = ibm_df[['Age', 'AgeClass', 'Attrition',	'Department',	'DistanceFromHome',	'Education',	'EducationField',	'EnvironmentSatisfaction',	'JobSatisfaction',	'MaritalStatus',	'MonthlyIncome',	'NumCompaniesWorked',	'WorkLifeBalance',	'YearsAtCompany']]

ibm_df['Attrition'].unique()
ibm_df.Attrition.replace({'Yes': 1, 'No': 0}, inplace=True)
ibm_df['MaritalStatus'].unique()
ibm_df.MaritalStatus.replace({'Single': 1, 'Married': 2, 'Divorced': 0}, inplace=True)

ibm_df.drop(columns=['WorkLifeBalance'], inplace=True)
ibm_df.head()
print(ibm_df.head())

#Discretizing DistanceFromHome and MonthlyIncome + adding DistanceClass and IncomeClass
Distance_labels = ['0_2', '3_6', '7_13', '14_28']
ibm_df['DistanceFromHome'] = pd.cut(ibm_df['DistanceFromHome'], bins=[0,	2,	6,	13,	29], labels=Distance_labels)
distanceclass = ibm_df.DistanceFromHome.replace({'0_2': 1, '3_6': 2, '7_13': 3, '14_28': 4})
ibm_df['DistanceClass'] = distanceclass
ibm_df = ibm_df[['Age', 'AgeClass', 'Attrition',	'Department',	'DistanceFromHome', 'DistanceClass',	'Education',	'EducationField',	'EnvironmentSatisfaction',	'JobSatisfaction',	'MaritalStatus',	'MonthlyIncome',	'NumCompaniesWorked',	'YearsAtCompany']]

Income_labels = ['(990.01, 5756.5]', '(5756.5, 10504.0]', '(10504.0, 15251.5]', '(15251.5, 19999.0]']
ibm_df['MonthlyIncome'] = pd.cut(ibm_df['MonthlyIncome'], bins=4, labels=Income_labels)
incomeclass = ibm_df.MonthlyIncome.replace({'(990.01, 5756.5]': 1, '(5756.5, 10504.0]': 2, '(10504.0, 15251.5]': 3, '(15251.5, 19999.0]': 4})
ibm_df['IncomeClass'] = incomeclass
ibm_df = ibm_df[['Age', 'AgeClass', 'Attrition',	'Department',	'DistanceFromHome', 'DistanceClass',	'Education',	'EducationField',	'EnvironmentSatisfaction',	'JobSatisfaction',	'MaritalStatus',	'MonthlyIncome', 'IncomeClass',	'NumCompaniesWorked',	'YearsAtCompany']]

#Data after data wrangling
data_dict_after_manipulation = st.sidebar.checkbox('Data dict after manipulation')
show_manipulated_data = st.sidebar.checkbox('Show manipulated data')

if data_dict_after_manipulation:
    st.subheader('Data dictionary after manipulation')
    st.markdown(
        """
        * **Age**: Age of employee
        * **AgeClass**: 1-'18_30'; 2-'31_36'; 3-'37_43'; 4-'44_60'
        * **Attrition**: Employee attrition status
        * **Department**: Department of work
        * **DistanceFromHome**
        * **DistanceClass**: 1-'0_2'; 2-'3_6'; 3-'7_13'; 4-'14_28'
        * **Education**: 1-Below College; 2- College; 3-Bachelor; 4-Master; 5-Doctor;
        * **EducationField**
        * **EnvironmentSatisfaction**: 1-Low; 2-Medium; 3-High; 4-Very High;
        * **JobSatisfaction**: 1-Low; 2-Medium; 3-High; 4-Very High;
        * **MaritalStatus**: 1-Single; 2-Married; 0-Divorced
        * **MonthlyIncome**
        * **IncomeClass**: 1-(990.01, 5756.5]; 2-(5756.5, 10504.0]; 3-(10504.0, 15251.5]; 4-(15251.5, 19999.0]
        * **NumCompaniesWorked:** Number of companies worked prior to IBM
        * **YearsAtCompany**: Current years of service in IBM'
        """)

if show_manipulated_data:
    st.subheader('Data after manipulation')
    st.write(ibm_df)

col_1, col_2 = st.columns(2)
with col_1:
    st.subheader('What is Age distribution now?')
    fig, ax = plt.subplots(figsize=(10, 6))
    ibm_df.Age.hist(ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.caption('Ages distribution in classes')

with col_2:
    st.subheader('Is Age correlated to Attrition?')
    st.write(ibm_df[['Age', 'Attrition']].groupby('Age', as_index=False).mean())

col_1, col_2 = st.columns(2)
with col_1:
    st.subheader('What is the MaritalStatus distribution?')
    fig, ax = plt.subplots(figsize=(10, 6))
    ibm_df.MaritalStatus.hist(ax=ax)
    ax.set_xlabel('MaritalStatus')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.caption('MaritalStatus distribution')

with col_2:
    st.subheader('Is MaritalStatus correlated to Attrition?')
    st.write(ibm_df[['MaritalStatus', 'Attrition']].groupby('MaritalStatus', as_index=False).mean())

plt.figure(figsize=(10, 6))
ibm_df.groupby('MaritalStatus')['Attrition'].count().plot.bar()
plt.show()

col_1, col_2 = st.columns(2)
with col_1:
    st.subheader('What is the MonthlyIncome distribution?')
    fig, ax = plt.subplots(figsize=(10, 6))
    ibm_df.MonthlyIncome.hist(ax=ax)
    ax.set_xlabel('MonthlyIncome')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.caption('MonthlyIncome distribution')

with col_2:
    st.subheader('Is MonthlyIncome correlated to Attrition?')
    st.write(ibm_df[['MonthlyIncome', 'Attrition']].groupby('MonthlyIncome', as_index=False).mean())

#Correlation Matrix
st.subheader('Correlation Matrix')
fig, ax = plt.subplots(figsize=(10, 6))
ibm_corr = ibm_df.corr()
sns.heatmap(ibm_corr, annot=True)
plt.show()
st.write(fig)

plt.figure(figsize=(10, 6))
plt.bar(ibm_df['Age'], ibm_df['YearsAtCompany'])
plt.title('Years at IBM per Age')
plt.xlabel('Age')
plt.ylabel('Years at the company')
plt.show()

ibm_df.info()

st.subheader('Distribution of the different attributes')
n = 3
fig = plt.figure(figsize=(15, 15))
for i in range(len(ibm_df.columns)):
    if len(ibm_df[ibm_df.columns[i]].unique()) <= 6:
        plt.subplot(int(len(ibm_df.columns)/n), n, i+1)
        sns.countplot(ibm_df[ibm_df.columns[i]])
st.write(fig)
plt.show()


#undersample data
yes_count = ibm_df[ibm_df['Attrition'] == 1].count()
class_1 = int(yes_count[1])
c1 = ibm_df[ibm_df['Attrition'] == 1]
c0 = ibm_df[ibm_df['Attrition'] == 0]
ibm_df_0 = c0.sample(class_1)
undersample_ibm_df = pd.concat([ibm_df_0, c1], axis=0)

#Classification models
with st.expander('Show model'):

    st.subheader('A model to predict the Attrition of employees')
    select_model = st.selectbox('Select model:', ['RandomForest', 'GaussianNB'])

    model = RandomForestClassifier()
    if select_model == 'GaussianNB':
        model = GaussianNB()

    choices = st.multiselect('Select features', ['AgeClass', 'IncomeClass', 'YearsAtCompany', 'MaritalStatus'])

    test_size = st.slider('Test size: ', min_value=0.1, max_value=0.9, step=0.1)

    data_modelled = st.selectbox('Select data to analyse', ['unbalanced_data', 'undersample_data'])
    y = ibm_df['Attrition']
    if data_modelled == 'undersample_data':
        y = undersample_ibm_df['Attrition']

    if len(choices) > 0 and st.button('RUN MODEL'):
        with st.spinner('Training...'):
            x = ibm_df[choices]
            if data_modelled == 'undersample_data':
                x = undersample_ibm_df[choices]
            train_model(x, y, model, random_state=42, test_size=test_size)

with st.expander('What happenes when undersampling'):
    col_1, col_2 = st.columns(2)
    with col_1:
        st.subheader('How Attrition was distributed:')
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.pie(ibm_df['Attrition'].value_counts(), labels=ibm_df['Attrition'].value_counts().index,
                autopct='%1.2f%%')
        st.pyplot(fig)
        st.caption('Attrition')

    with col_2:
        st.subheader('How Attrition is now distributed:')
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.pie(undersample_ibm_df['Attrition'].value_counts(),
                labels=undersample_ibm_df['Attrition'].value_counts().index, autopct='%1.2f%%')
        st.pyplot(fig)
        st.caption('Attrition')
