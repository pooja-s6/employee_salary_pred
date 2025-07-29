# Step 1: Install Required Libraries
pip install pandas scikit-learn joblib

# Step 2: Import Necessary Modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 3: Create Dummy Income Data (or use your CSV if available)
# You can replace this with pd.read_csv('your_file.csv')
data = {
    'age': [25, 32, 47, 51, 62],
    'income': [50000, 64000, 120000, 110000, 150000],
    'loan_amount': [10000, 15000, 20000, 25000, 30000],
    'approved': [0, 0, 1, 1, 1]  # Target column
}
df = pd.DataFrame(data)

# Step 4: Prepare Features and Target
X = df[['age', 'income', 'loan_amount']]
y = df['approved']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 7: Save Trained Model
joblib.dump(model, 'trained_model.joblib')

# Step 8: Download the Model File to Your System
from google.colab import files
files.download('trained_model.joblib')
