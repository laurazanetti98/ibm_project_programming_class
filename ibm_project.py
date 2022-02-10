import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv = r'C:\Users\asus\Desktop\IBM.csv'
ibm_df = pd.read_csv(csv, sep=',')

ibm_df.info()

null_value = ibm_df.isnull().values.any()
print('Is there any null value? : ', null_value)

plt.figure(figsize=(10, 6))
my_explode = (0, 0.1)
plt.pie(ibm_df['Attrition'].value_counts(), labels=ibm_df['Attrition'].value_counts().index, shadow=True, autopct='%1.2f%%', explode=my_explode)
plt.title('Attrition')
plt.show()

plt.figure(figsize=(10, 6))
ibm_df.Age.hist()
plt.show()

my_labels = ['18_30', '31_36', '37_43', '44_60']
ibm_df['Age'] = pd.cut(ibm_df['Age'], bins=[17, 30, 36,	43, 60], labels=my_labels)

ageclass = ibm_df.Age.replace({'18_30': 1, '31_36': 2, '37_43': 3, '44_60': 4})
ibm_df['AgeClass'] = ageclass
ibm_df = ibm_df[['Age', 'AgeClass', 'Attrition',	'Department',	'DistanceFromHome',	'Education',	'EducationField',	'EnvironmentSatisfaction',	'JobSatisfaction',	'MaritalStatus',	'MonthlyIncome',	'NumCompaniesWorked',	'WorkLifeBalance',	'YearsAtCompany']]

ibm_df['Attrition'].unique()
ibm_df.Attrition.replace({'Yes': 1, 'No': 0}, inplace=True)

ibm_df.drop(columns=['WorkLifeBalance'], inplace=True)
ibm_df.head()
print(ibm_df.head())

plt.figure(figsize=(10, 6))
ibm_df.Age.hist()
plt.show()

print(ibm_df[['Age', 'Attrition']].groupby('Age').mean())
print('-' * 20)
print(ibm_df[['MaritalStatus', 'Attrition']].groupby('MaritalStatus').mean())

plt.figure(figsize=(10, 6))
ibm_df.groupby('MaritalStatus')['Attrition'].count().plot.bar()
plt.show()

ibm_corr = ibm_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(ibm_corr, annot=True)
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(ibm_df['Age'], ibm_df['YearsAtCompany'])
plt.title('Years at IBM per Age')
plt.xlabel('Age')
plt.ylabel('Years at the company')
plt.show()

ibm_df.info()

n = 3
fig_1 = plt.figure(figsize=(15, 15))
for i in range(len(ibm_df.columns)):
    if len(ibm_df[ibm_df.columns[i]].unique()) <= 6:
        plt.subplot(int(len(ibm_df.columns)/n), n, i+1)
        sns.countplot(ibm_df[ibm_df.columns[i]])
plt.show()

#Classification model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
y = ibm_df['Attrition']
x = ibm_df['AgeClass']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=2)
model = RandomForestClassifier()
x_train = x_train.to_numpy().reshape(-1, 1)
model.fit(x_train, y_train)
x_test = x_test.to_numpy().reshape(-1, 1)
y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)
print('The accuracy score is: ', accuracy_score(y_test, y_pred))

