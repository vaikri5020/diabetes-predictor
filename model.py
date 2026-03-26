import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,f1_score,classification_report
import pickle as pk

df=pd.read_csv("diabetes.csv")
X=df[['Glucose','BloodPressure']]
y=df['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)
model=LogisticRegression()
model.fit(X_train,y_train)
with open("model.pkl","wb") as f:
    pk.dump(model,f)


print("model trained")    
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred) ) 