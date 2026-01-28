from sklearn.linear_model import LogisticRegression
import joblib

# sample data
X = [[10],[20],[30],[40]]
y = [0,0,1,1]

model = LogisticRegression()
model.fit(X, y)

# save model
joblib.dump(model, "model.pkl")
print("Model saved!")
