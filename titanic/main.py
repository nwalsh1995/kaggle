import csv
import sys

import torch

test_file = sys.argv[-3]
training_file = sys.argv[-2]
epochs = int(sys.argv[-1])

# Load data from 'train.csv'
def get_dataset(file_name):
    data = []
    labels = []
    passenger_ids = []
    with open(file_name, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                age = int(row["Age"])
            except ValueError:
                # age empty, set default
                age = 30
            try:
                embarked = ord(row["Embarked"])
            except TypeError:
                # embarked empty, set default
                embarked = ord("C")
            try:
                fare = float(row["Fare"])
            except ValueError:
                fare = 7.50

            data.append([
                int(row["Pclass"]),
                0 if row["Sex"] == "male" else 1,
                age,
                int(row["SibSp"]),
                int(row["Parch"]),
                #row["Ticket"],  # String value
                fare,
                #row["Cabin"],  # Some have cabins and some dont, string value
                embarked,
            ])
            passenger_ids.append(row["PassengerId"])
            try:
                labels.append(int(row["Survived"]))
            except Exception:
                continue
    return passenger_ids, data, labels
    
# Pclass, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
input_dim = 7
output_dim = 1


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegression(input_dim, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
_, data, labels = get_dataset(training_file)
labels = torch.tensor(labels).to(torch.float)
data = torch.tensor(data)
criterion = torch.nn.BCELoss()
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(data)
    y_pred = y_pred.squeeze(-1).to(torch.float)
    # Calculate accuracy
    correct = 0
    for pred, label in zip(y_pred, labels):
        if pred >= .5 and label == 1.:
            correct += 1
        elif pred < .5 and label == 0.:
            correct += 1
    print(f"Accuracy of epoch {epoch+1}: {(correct/len(data) * 100)}")
    loss = criterion(y_pred, labels)
    loss.backward()
    optimizer.step()
    

passenger_ids, test_data, _ = get_dataset(test_file)
test_data = torch.tensor(test_data)
with open("model_predictions.csv", "w") as out_file:
    out_file.write("PassengerId,Survived\n")
    preds = model(test_data)
    for passenger_id, pred in zip(passenger_ids, preds):
        decision = 1
        if pred < .5:
            decision = 0
        out_file.write(f"{passenger_id},{decision}\n")
print("Written prediction!")
