import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

url = 'https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'

df = pd.read_csv(url)

df = df.drop(['PassengerId','Cabin', 'Ticket', 'Name'], axis = 1)

df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

embarked_dict = {'S': 0, 'C': 1, 'Q':2}
df['Embarked'] = df['Embarked'].map(embarked_dict)

df['Survived']=pd.Categorical(df['Survived'])
df['Sex']=pd.Categorical(df['Sex'])
df['Embarked']=pd.Categorical(df['Embarked'])

df_processed = df.copy()
df_processed['Age'].fillna(df_processed['Age'].mean(), inplace = True)
df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace = True)

X = df_processed.drop(['Survived'], axis=1)
y = df_processed['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

RFC = RandomForestClassifier(max_depth=8, n_estimators=150, random_state=4)

RFC.fit(X_train,y_train)

# Guardar modelo
filename = '../models/modelo_random_forest.sav'
pickle.dump(RFC, open(filename, 'wb'))

print('Se guardó el modelo')