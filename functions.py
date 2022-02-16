import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
def train_model(x0, y0, model, random_state = 42, test_size = 0.4):
  x_train, x_test, y_train, y_test = train_test_split(x0, y0, test_size=test_size, stratify=y0, random_state=random_state)
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  st.write('Accuracy score: ', str(accuracy_score(y_test, y_pred)))