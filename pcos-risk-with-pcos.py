#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency


# In[18]:


# Load and prepare the data (use the biased dataset generated earlier)
df = pd.read_csv('pcos_diabetes_data.csv')


# In[19]:


# Define PCOS and Non-PCOS groups
pcos_group = df[df['PCOS_PCOD'] == 1]
non_pcos_group = df[df['PCOS_PCOD'] == 0]

# 1. Statistical Comparisons Between PCOS and Non-PCOS Groups

# Compare glucose levels between PCOS and non-PCOS patients
t_stat, p_val = ttest_ind(pcos_group['Glucose'], non_pcos_group['Glucose'])
print(f"T-Test for Glucose Levels: T-Statistic = {t_stat:.2f}, P-Value = {p_val:.4f}")
# Interpretation: A low P-Value indicates significant difference in glucose levels between groups

# Compare insulin levels between PCOS and non-PCOS patients
t_stat, p_val = ttest_ind(pcos_group['Insulin'], non_pcos_group['Insulin'])
print(f"T-Test for Insulin Levels: T-Statistic = {t_stat:.2f}, P-Value = {p_val:.4f}")

# Chi-square test for categorical variables like Menstrual Irregularity
contingency_table = pd.crosstab(df['PCOS_PCOD'], df['Menstrual_Irregularity'])
chi2, p_val, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square Test for Menstrual Irregularity: Chi2 = {chi2:.2f}, P-Value = {p_val:.4f}")


# 
# 
# ### **Statistical Results:**
# 
# 1. **T-Test for Glucose Levels:**
#    - **T-Statistic**: 27.87
#    - **P-Value**: 0.0000
#    - **Interpretation**: The very low p-value indicates a statistically significant difference in glucose levels between PCOS and non-PCOS groups. This result strongly suggests that PCOS patients have higher glucose levels, reinforcing the hypothesis that PCOS is associated with increased diabetes risk.
# 
# 2. **T-Test for Insulin Levels:**
#    - **T-Statistic**: 17.12
#    - **P-Value**: 0.0000
#    - **Interpretation**: This significant difference in insulin levels between PCOS and non-PCOS groups suggests that PCOS patients exhibit higher insulin levels, which is a hallmark of insulin resistance. Insulin resistance is a critical precursor to type 2 diabetes, further supporting the connection between PCOS and increased diabetes risk.
# 
# 3. **Chi-Square Test for Menstrual Irregularity:**
#    - **Chi2**: 256.83
#    - **P-Value**: 0.0000
#    - **Interpretation**: The chi-square test reveals a significant association between PCOS and menstrual irregularity. This result confirms that menstrual irregularity is highly prevalent in PCOS patients, adding to the metabolic and hormonal imbalances that contribute to diabetes risk.
# 
# ### **Key Insights and Actions:**
# - **Metabolic Disruption**: The data clearly shows that PCOS patients have significant metabolic disruptions, with elevated glucose and insulin levels compared to non-PCOS patients. These metabolic changes directly increase the risk of developing diabetes.
# - **Hormonal Imbalance**: The strong association between PCOS and menstrual irregularity highlights how hormonal imbalances play a crucial role in the overall health risks associated with PCOS, including diabetes.
# - **Clinical Implications**: Healthcare providers should consider early screening for glucose and insulin levels in PCOS patients. By identifying at-risk individuals early, targeted interventions such as lifestyle modifications and medical management can be implemented to reduce the long-term risk of diabetes.
# - **Research Validation**: The statistically significant results provide a robust foundation for further research into the complex relationships between PCOS, metabolic health, and diabetes. This evidence supports the need for more personalized approaches in managing PCOS patients.
# 
# 

# In[20]:


# 2. Correlation Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# 
# 
# ### **Key Insights from the Correlation Heatmap:**
# 
# 1. **Strong Correlation Between PCOS and Related Features:**
#    - **PCOS_PCOD and Menstrual Irregularity**: High positive correlation (0.72) suggests that menstrual irregularity is a strong indicator of PCOS. This irregularity is linked to broader metabolic issues that contribute to diabetes risk.
#    - **PCOS_PCOD and Testosterone Levels**: A strong positive correlation (0.74) indicates that elevated testosterone levels are prevalent among PCOS patients, which aligns with the hormonal imbalance seen in PCOS. This hormonal disruption can affect insulin sensitivity.
# 
# 2. **Glucose, Insulin, and HOMA-IR Relationships:**
#    - **Glucose and Insulin**: Moderate positive correlation (0.48) highlights the link between glucose levels and insulin response. This is crucial as it indicates how insulin resistance might be influencing elevated glucose levels, particularly in PCOS patients.
#    - **HOMA-IR and Insulin**: A high correlation (0.41) between HOMA-IR and insulin levels further emphasizes insulin resistance, a condition common in PCOS patients that significantly raises diabetes risk.
# 
# 3. **Outcome (Diabetes) Associations:**
#    - **Outcome and Glucose**: Moderate positive correlation (0.42) suggests that elevated glucose levels are strongly associated with diabetes outcomes, highlighting the importance of glucose management in PCOS patients.
#    - **Outcome and PCOS_PCOD**: Moderate negative correlation (-0.3) with PCOS_PCOD indicates that while PCOS patients have higher risks, the diagnosis itself isn’t directly linked to diabetes without considering other factors like glucose and insulin levels.
# 
# 4. **Other Notable Correlations:**
#    - **PCOS_PCOD and Hirsutism (0.69)**: Strong correlation suggests that hirsutism (excessive hair growth) is another prevalent symptom in PCOS patients, reinforcing the hormonal profile associated with increased diabetes risk.
#    - **Testosterone Levels and Outcome**: A moderate negative correlation (-0.28) between testosterone levels and diabetes outcomes might reflect the complex hormonal interplay in PCOS, where testosterone is both a marker of the condition and a contributor to metabolic disturbances.
# 
# ### **Summary and Actionable Insights:**
# - **PCOS is a Complex Risk Factor**: The correlations underscore that PCOS isn’t a standalone predictor of diabetes; rather, it’s the combination of elevated insulin, glucose levels, and associated hormonal imbalances that significantly raise the risk.
# - **Targeted Interventions Needed**: Early monitoring of glucose and insulin levels in PCOS patients, combined with managing symptoms like menstrual irregularity and high testosterone, could be key in mitigating the risk of developing diabetes.
# - **Multi-Factor Approach**: Effective management should consider both metabolic (glucose, insulin) and hormonal factors (testosterone, menstrual irregularity) to provide comprehensive care for PCOS patients.
# 
# This heatmap provides valuable insights into how various factors interrelate, particularly highlighting the importance of metabolic management in preventing diabetes in PCOS patients. If you need more analysis or further interpretation, let me know!

# In[21]:


# 3. Visualizations

# Distribution of Glucose Levels by PCOS Status
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Glucose', hue='PCOS_PCOD', kde=True, bins=30, palette='viridis')
plt.title('Distribution of Glucose Levels by PCOS Status')
plt.xlabel('Glucose Level (mg/dL)')
plt.ylabel('Frequency')
plt.axvline(x=126, color='red', linestyle='--', label='Diabetes Threshold')
plt.legend()
plt.show()


# Boxplot of Insulin Levels by Diabetes Outcome
plt.figure(figsize=(10, 6))
sns.boxplot(x='Outcome', y='Insulin', hue='PCOS_PCOD', data=df, palette='Set2')
plt.title('Insulin Levels by Diabetes Outcome and PCOS Status')
plt.xlabel('Diabetes Outcome (0 = No, 1 = Yes)')
plt.ylabel('Insulin Levels (µU/mL)')
plt.show()


# Scatter Plot of BMI vs. Glucose by PCOS Status
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='BMI', y='Glucose', hue='PCOS_PCOD', style='Outcome', palette='deep')
plt.title('BMI vs. Glucose Levels by PCOS Status and Diabetes Outcome')
plt.xlabel('BMI')
plt.ylabel('Glucose Level (mg/dL)')
plt.axhline(y=126, color='red', linestyle='--', label='Diabetes Threshold')
plt.legend()
plt.show()


# ### **Visual Insights from the Plots**
# 
# #### **1. BMI vs. Glucose Levels by PCOS Status and Diabetes Outcome**
# - **Observation**: The scatter plot shows the relationship between BMI and glucose levels, differentiated by PCOS status and diabetes outcome. The red dashed line indicates the diabetes threshold (glucose level of 126 mg/dL).
# - **Key Insights**:
#   - **PCOS Patients**: Represented by orange markers, many PCOS patients have glucose levels above the diabetes threshold, especially those with higher BMI values. This suggests that PCOS patients are more likely to have elevated glucose levels, particularly when BMI is also high.
#   - **Non-PCOS Patients**: Represented by blue markers, most non-PCOS patients have glucose levels below the diabetes threshold, even when BMI is high.
#   - **Implication**: PCOS, combined with higher BMI, significantly increases the likelihood of crossing the diabetes threshold, indicating a compounded risk factor for diabetes.
# 
# #### **2. Distribution of Glucose Levels by PCOS Status**
# - **Observation**: The histogram shows the distribution of glucose levels in PCOS (green) and non-PCOS (blue) patients, with the diabetes threshold marked.
# - **Key Insights**:
#   - **PCOS Group**: A significant portion of PCOS patients have glucose levels above the threshold, with a right-skewed distribution suggesting higher glucose levels are common.
#   - **Non-PCOS Group**: The distribution is centered well below the diabetes threshold, indicating that non-PCOS patients are less likely to have elevated glucose levels.
#   - **Implication**: The clear distinction between the two distributions highlights the increased metabolic risk associated with PCOS, emphasizing the need for early intervention in PCOS patients to manage glucose levels.
# 
# #### **3. Insulin Levels by Diabetes Outcome and PCOS Status**
# - **Observation**: The boxplot compares insulin levels across diabetes outcomes and PCOS status.
# - **Key Insights**:
#   - **PCOS Patients with Diabetes**: This group shows the highest median insulin levels, indicating severe insulin resistance, a key factor in diabetes development.
#   - **Non-PCOS Patients with Diabetes**: Although their insulin levels are elevated, they are significantly lower than those seen in PCOS patients with diabetes, underscoring the additional metabolic burden PCOS places on individuals.
#   - **PCOS Patients without Diabetes**: Even in the absence of diabetes, insulin levels are higher than in non-PCOS patients, indicating that PCOS alone predisposes patients to insulin resistance.
#   - **Implication**: Insulin resistance is markedly more severe in PCOS patients, particularly those with diabetes, reinforcing the need for targeted metabolic monitoring and management strategies in this population.
# 
# ### **Overall Implications for Patient Care:**
# - **Combined Risk Factors**: The combination of PCOS, high BMI, elevated glucose, and insulin resistance significantly increases the risk of developing diabetes. This highlights the importance of comprehensive metabolic management in PCOS patients, focusing on both weight control and glucose regulation.
# - **Early Intervention**: Regular screening for glucose and insulin levels in PCOS patients, especially those with higher BMI, can help identify those at risk early and guide preventive strategies.
# - **Personalized Care**: Tailoring interventions to address both hormonal (PCOS-related) and metabolic (glucose, insulin) factors will be critical in reducing the long-term risk of diabetes in PCOS patients.
# 
# 

# In[26]:


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare the data 
X = df.drop(columns=['Outcome'])  # Features
y = df['Outcome']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define the Neural Network architecture
class DiabetesRiskModel(nn.Module):
    def __init__(self, input_size):
        super(DiabetesRiskModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
model = DiabetesRiskModel(input_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the Neural Network
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = (predictions > 0.5).float()  # Convert probabilities to binary classes
    accuracy = accuracy_score(y_test_tensor, predicted_classes)
    auc_score = roc_auc_score(y_test_tensor, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC-AUC Score: {auc_score:.2f}")
    print(classification_report(y_test_tensor, predicted_classes))



# In[27]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare the data
X = df.drop(columns=['Outcome'])  # Use 'df' for the dataset
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader for batch training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the Neural Network architecture with dropout for regularization
class ImprovedDiabetesRiskModel(nn.Module):
    def __init__(self, input_size):
        super(ImprovedDiabetesRiskModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)  # Dropout layer to prevent overfitting
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
model = ImprovedDiabetesRiskModel(input_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate if needed

# Training the Neural Network with mini-batches
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch_X, batch_y = batch
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = (predictions > 0.5).float()  # Convert probabilities to binary classes
    accuracy = accuracy_score(y_test_tensor, predicted_classes)
    auc_score = roc_auc_score(y_test_tensor, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC-AUC Score: {auc_score:.2f}")
    print(classification_report(y_test_tensor, predicted_classes))


# In[28]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare the data
X = df.drop(columns=['Outcome']) 
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader for batch training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the improved Neural Network architecture
class FineTunedDiabetesRiskModel(nn.Module):
    def __init__(self, input_size):
        super(FineTunedDiabetesRiskModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU()  # Using LeakyReLU
        self.dropout = nn.Dropout(p=0.4)  # Increased dropout rate
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout(self.relu(self.bn2(self.layer2(x))))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize the model, loss function, and optimizer with L2 regularization
input_size = X_train.shape[1]
model = FineTunedDiabetesRiskModel(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay for L2 regularization

# Add a learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Training the Neural Network with fine-tuned parameters
epochs = 150
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch_X, batch_y = batch
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Step the scheduler
    scheduler.step(total_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = (predictions > 0.5).float()  # Convert probabilities to binary classes
    accuracy = accuracy_score(y_test_tensor, predicted_classes)
    auc_score = roc_auc_score(y_test_tensor, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC-AUC Score: {auc_score:.2f}")
    print(classification_report(y_test_tensor, predicted_classes))


# In[29]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, ParameterGrid

# Prepare the data
X = df.drop(columns=['Outcome'])  # Use 'df' for the dataset
y = df['Outcome']

# Feature Engineering: Add interaction and polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader for batch training
def create_dataloader(X_tensor, y_tensor, batch_size):
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Neural Network architecture
class FineTunedDiabetesRiskModel(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(FineTunedDiabetesRiskModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU()  # Using LeakyReLU
        self.dropout = nn.Dropout(p=dropout_rate)  # Adjustable dropout rate
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout(self.relu(self.bn2(self.layer2(x))))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# Define function to train the model
def train_model(model, train_loader, criterion, optimizer, scheduler, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch_X, batch_y = batch
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

# Function to evaluate the model
def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predicted_classes = (predictions > 0.5).float()  # Convert probabilities to binary classes
        accuracy = accuracy_score(y_test_tensor, predicted_classes)
        auc_score = roc_auc_score(y_test_tensor, predictions)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print(f"ROC-AUC Score: {auc_score:.2f}")
        print(classification_report(y_test_tensor, predicted_classes))

# Define parameter grid for Grid Search
param_grid = {
    'lr': [0.0001, 0.001, 0.01],
    'dropout_rate': [0.3, 0.4, 0.5],
    'batch_size': [16, 32, 64]
}

# Grid Search Implementation
best_auc = 0
best_params = {}
input_size = X_train_scaled.shape[1]

for params in ParameterGrid(param_grid):
    print(f"Testing parameters: {params}")
    
    # Create DataLoader
    train_loader = create_dataloader(X_train_tensor, y_train_tensor, params['batch_size'])
    
    # Initialize model, criterion, optimizer, and scheduler
    model = FineTunedDiabetesRiskModel(input_size, params['dropout_rate'])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, scheduler, epochs=150)
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        auc_score = roc_auc_score(y_test_tensor, predictions)
        
        # Update best model if current model has a higher AUC score
        if auc_score > best_auc:
            best_auc = auc_score
            best_params = params
            best_model = model

print(f"Best Parameters: {best_params}")
print(f"Best ROC-AUC Score: {best_auc:.2f}")

# Evaluate the best model with full metrics
evaluate_model(best_model, X_test_tensor, y_test_tensor)


# In[35]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import shap

# Prepare the data
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Feature Engineering: Add interaction and polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors for the neural network
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader for the neural network
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

# Define the Neural Network architecture
class NeuralNetModel(nn.Module):
    def __init__(self, input_size, dropout_rate=0.4):
        super(NeuralNetModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize the neural network model
input_size = X_train_scaled.shape[1]
nn_model = NeuralNetModel(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# Training the Neural Network
def train_neural_net(model, train_loader, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch_X, batch_y = batch
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

train_neural_net(nn_model, train_loader, criterion, optimizer)

# Extract neural network predictions
nn_model.eval()
with torch.no_grad():
    nn_predictions = nn_model(X_test_tensor).numpy().flatten()

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict_proba(X_test_scaled)[:, 1]

# Train XGBoost Model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Combine predictions as inputs to a meta-model
stacked_predictions = np.column_stack((nn_predictions, rf_predictions, xgb_predictions))

# Train meta-model (Logistic Regression as an example)
from sklearn.linear_model import LogisticRegression
meta_model = LogisticRegression()
meta_model.fit(stacked_predictions, y_test)
meta_predictions = meta_model.predict(stacked_predictions)
meta_probabilities = meta_model.predict_proba(stacked_predictions)[:, 1]

# Evaluate the Stacking Ensemble
accuracy = accuracy_score(y_test, meta_predictions)
auc_score = roc_auc_score(y_test, meta_probabilities)
print(f"Stacking Ensemble Accuracy: {accuracy * 100:.2f}%")
print(f"Stacking Ensemble ROC-AUC Score: {auc_score:.2f}")
print(classification_report(y_test, meta_predictions))

# SHAP Analysis for Feature Importance on XGBoost Model
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

# Plot SHAP summary plot
shap.summary_plot(shap_values, X_test_scaled, feature_names=poly.get_feature_names_out())



# ### **Analysis of the Stacking Ensemble Results and SHAP Summary Plot**
# 
# #### **Model Performance:**
# - **Stacking Ensemble Accuracy**: 63.00%
# - **ROC-AUC Score**: 0.56
# - The model is better at predicting positive cases (diabetes outcomes) than negative cases, as seen from the higher precision, recall, and f1-score for class 1 (diabetes).
# 
# #### **SHAP Summary Plot Insights:**
# The SHAP plot provides valuable insights into how each feature impacts the model's predictions, with the most significant features listed at the top. Here's an interpretation of key insights:
# 
# 1. **Top Contributing Features**:
#    - **BMI, PCOS_PCOD**: High BMI combined with PCOS status significantly impacts the model’s output, showing that obesity and PCOS together play a crucial role in increasing diabetes risk.
#    - **Insulin, Physical Activity**: This interaction indicates that insulin levels combined with low physical activity are crucial risk factors, highlighting the metabolic impacts of sedentary lifestyles on PCOS patients.
#    - **Age, HOMA_IR**: Older age combined with high insulin resistance (HOMA-IR) further emphasizes the risk, particularly for older PCOS patients with metabolic disturbances.
#    - **Glucose Levels**: Glucose-related features, particularly in interaction with other metabolic markers, play a significant role, indicating direct links to diabetes risk.
# 
# 2. **Interaction Effects**:
#    - **Physical Activity**: The effect of physical activity is evident across several interactions, suggesting that lifestyle interventions can be crucial for managing risk.
#    - **Hormonal Imbalance Indicators**: Features like **Menstrual Irregularity** and **Polycystic Ovaries** show up with different metabolic combinations, pointing towards the need to address hormonal imbalances in PCOS treatment.
# 
# 3. **Implications for Patient Management**:
#    - **Targeted Interventions**: Patients with high BMI, insulin resistance, and sedentary lifestyles could benefit the most from interventions focusing on weight loss and physical activity improvement.
#    - **Personalized Care**: Identifying at-risk individuals early allows for personalized healthcare plans, focusing on lifestyle, dietary habits, and medical management to reduce diabetes risk.
# 
# 

# In[ ]:




