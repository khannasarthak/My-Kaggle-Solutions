import pandas as pd
import numpy as np
from sklearn import tree
train_csv = 'train.csv'
test_csv = 'test.csv'
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)

# print (train['Survived'][train['Sex'] == 'male'].value_counts())
# print (train['Survived'][train['Sex'] == 'female'].value_counts())

test_one = test
test_one['Survived'] = 0

test_one['Survived'][test_one['Sex'] == 'female'] = 1
# print (test_one['Survived'].value_counts())

train['Age'] = train['Age'].fillna(train['Age'].median())

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
# Impute the Embarked variable
train["Embarked"] = train['Embarked'].fillna(train['Embarked']=='S')

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
# Impute the Embarked variable
test["Embarked"] = test['Embarked'].fillna(test['Embarked']=='S')

# Convert the Embarked classes to integer form
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
#Print the Sex and Embarked columns

target = train['Survived'].values
features = train[['Pclass','Sex','Age','Fare']].values

my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(features,target)

test['Fare'] = test['Fare'].fillna(test['Fare'].median())


test_features = test[['Pclass','Sex','Age','Fare']].values
my_prediction = my_tree.predict(test_features)
print (my_prediction)
PassengerId = np.array(test['PassengerId']).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns= ['Survived'])
print (my_solution.shape)

my_solution.to_csv('my_solution_1.csv', index_label = ['PassengerId'])