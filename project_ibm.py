import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
csv = r'C:\Users\asus\Desktop\IBM.csv'
ibm_df = pd.read_csv(csv, sep=',')
ibm_df.info()

ibm_df[ibm_df['Attrition'] == 'Yes'].count()

plt.figure(figsize=(10, 6))
my_explode = (0, 0.1)
plt.pie(ibm_df['Attrition'].value_counts(), labels=ibm_df['Attrition'].value_counts().index, autopct='%1.2f%%')
plt.title('Attrition')
plt.show()

plt.figure(figsize=(10, 6))
ibm_df.Age.hist()
plt.show()
