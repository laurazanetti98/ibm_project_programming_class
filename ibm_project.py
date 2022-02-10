import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv = r'C:\Users\asus\Desktop\IBM.csv'
ibm_df = pd.read_csv(csv, sep=',')

ibm_df.info()

plt.figure(figsize=(10, 6))
my_explode = (0, 0.1)
plt.pie(ibm_df['Attrition'].value_counts(), labels=ibm_df['Attrition'].value_counts().index, autopct='%1.2f%%')
plt.title('Attrition')
plt.show()

plt.figure(figsize=(10, 6))
ibm_df.Age.hist()
plt.show()

ibm_copy = ibm_df.copy()
ibm_copy

my_labels = ['18_30', '31_36', '37_43', '44_60']
ibm_df['Age'] = pd.cut(ibm_df['Age'], bins=[17, 30, 36,	43, 60], labels=my_labels)

ibm_df['Attrition'].unique()
ibm_df.Attrition.replace( { 'Yes':1, 'No':0}, inplace=True )
ibm_df

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
