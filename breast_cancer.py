from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from neural_network_2_layers import MLP_Two_Layers
breast_cancer_data = load_breast_cancer()
X = breast_cancer_data['data']
y = breast_cancer_data['target']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42
                                                   )

model = MLP_Two_Layers(
    input_size=30, hidden_size=15, output_size=1, random_state=42
)

model.fit(X_train,
          y_train,
          batch_size=128,
          epochs=900,
          learning_rate=0.001,
          beta=0.9)


y_predict = model.predict(X_test)

print(classification_report(y_test,
                            y_predict,
                            target_names=['positive','negative'],
                            digits=3
                           )
     )