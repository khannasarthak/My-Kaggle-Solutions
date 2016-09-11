import pandas as pd
import numpy as np
from sklearn import tree
train_csv = 'train.csv'
test_csv = 'test.csv'
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)

train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Sex'][train['Sex'] == 'male'] = 1
train['Sex'][train['Sex'] == 'female'] = 0

features = train[['Pclass','Sex','Age','Fare']]
target = train['Survived'].values
myTree = tree.DecisionTreeClassifier()
myTree.fit(features,target)
# print(myTree.score(features,target))

# Test data
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Sex'][test['Sex'] == 'male'] = 1
test['Sex'][test['Sex'] == 'female'] = 0

test_features = test[['Pclass','Sex','Age','Fare']]
my_prediction = myTree.predict(test_features)

PassengerId = np.array(test['PassengerId']).astype(int)
my_solu = pd.DataFrame(my_prediction,PassengerId, columns=['Survived'])

print (my_solu)

my_solu.to_csv('Solu1.csv', index_label= ['PassengerId'])



