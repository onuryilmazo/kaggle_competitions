import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

# Train verilerini yükleyin
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Gereksiz sütunları düşür
train_data = train_data.drop(columns=["Name", "Ticket", "Cabin"])
test_data = test_data.drop(columns=["Name", "Ticket", "Cabin"])

test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})
test_data["Embarked"] = test_data["Embarked"].map({"C": 0, "Q": 1, "S": 2})

# Kategorik sütunları işleyin
train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})
train_data["Embarked"] = train_data["Embarked"].map({"C": 0, "Q": 1, "S": 2})

# Bağımsız değişkenleri (X_train) ve hedef değişkeni (y_train) ayırın
X_train = train_data[["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y_train = train_data["Survived"]

# Bağımsız değişkenleri (X_test) alın (Test verilerinde "Survived" sütunu yok)
X_test = test_data[["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

# Eksik değerleri ortalama ile doldur
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

model = RandomForestClassifier(n_estimators=69, max_depth=8, random_state=1)
model.fit(X_train_imputed, y_train)

# Test verileri üzerinde tahmin yap
y_pred = model.predict(X_test_imputed)

# Tahminleri bir DataFrame'e ekleyin
output_df = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})

# Çıktı dosyasını CSV formatında kaydedin
output_df.to_csv('submission.csv', index=False)

#En iyi n_estimators: 69
#En iyi max_depth: 8

'''
n_estimators_values = list(range(60, 120, 2))
max_depth_values = list(range(5, 10))

best_accuracy = 0
best_n_estimators = None
best_max_depth = None

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1)
        scores = cross_val_score(model, X_train_imputed, y_train, cv=5, scoring='accuracy')
        avg_accuracy = np.mean(scores)
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_n_estimators = n_estimators
            best_max_depth = max_depth

print("En iyi Accuracy:", best_accuracy)
print("En iyi n_estimators:", best_n_estimators)
print("En iyi max_depth:", best_max_depth)

'''


