# 💻 Laptop Price Predictor

An end-to-end Machine Learning project that predicts the price of a laptop based on its specifications — built from raw, messy real-world data all the way to a deployed, interactive web app.

🔗 **Live App:** (https://laptoppricepredictor-ml.streamlit.app/)
📂 **Repository:** [github.com/divay0407/Laptop_Price_Predictor](https://github.com/divay0407/Laptop_Price_Predictor)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Data Cleaning](#-data-cleaning)
- [Feature Engineering](#-feature-engineering)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Model Building](#-model-building)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [How the App Works](#-how-the-app-works)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🧠 Overview

This project predicts laptop prices using various hardware and configuration specifications like RAM, CPU, GPU, storage type, screen resolution, and more. It covers the **complete ML lifecycle**:

1. Working with raw, uncleaned Excel/CSV data
2. Cleaning and preprocessing
3. Feature extraction and engineering
4. In-depth Exploratory Data Analysis
5. Training and comparing 10+ regression models
6. Hyperparameter tuning
7. Building a preprocessing + model pipeline
8. Deploying a live, interactive web app on Streamlit Community Cloud

---

## ❓ Problem Statement

Laptop prices depend on a complex mix of specifications — brand, processor, RAM, storage type and size, GPU, display quality, and more. Manually estimating a "fair price" for a laptop configuration is difficult for a regular buyer.

**Goal:** Build a regression model that takes in laptop specifications and predicts its expected market price with high accuracy, then make it accessible through a simple web interface.

---

## 📊 Dataset

The raw dataset was obtained as an **unprocessed Excel/CSV file** scraped from laptop listings, containing inconsistent formatting, mixed data types, and missing values.

**Raw columns included (before cleaning):**
- `Company`, `Product`, `TypeName`, `Inches`, `ScreenResolution`
- `Cpu`, `Ram`, `Memory`, `Gpu`, `OpSys`, `Weight`, `Price`

**After cleaning and feature engineering, final columns used for modeling:**

| Column | Description |
|---|---|
| `Company` | Laptop brand (Dell, HP, Apple, etc.) |
| `TypeName` | Category (Ultrabook, Notebook, Gaming, etc.) |
| `Ram` | RAM size in GB |
| `Weight` | Weight in kg |
| `Touchscreen` | 1 if touchscreen, else 0 |
| `Ips` | 1 if IPS panel, else 0 |
| `ppi` | Pixels per inch (derived from resolution + screen size) |
| `Cpu brand` | Simplified processor brand/tier |
| `HDD` | HDD capacity in GB |
| `SSD` | SSD capacity in GB |
| `Gpu brand` | GPU manufacturer |
| `os` | Simplified operating system category |
| `Price` | Target variable (log-transformed for training) |

---

## 🔄 Project Workflow

```
Raw Excel Data
      │
      ▼
Data Cleaning (nulls, duplicates, inconsistent formatting)
      │
      ▼
Feature Extraction (Memory, Resolution, Cpu, Gpu columns)
      │
      ▼
Feature Engineering (ppi, HDD/SSD split, binary flags)
      │
      ▼
Exploratory Data Analysis (EDA)
      │
      ▼
Encoding + Train-Test Split + Scaling
      │
      ▼
Model Training (10+ algorithms compared)
      │
      ▼
Hyperparameter Tuning (GridSearchCV)
      │
      ▼
Pipeline Creation (preprocessing + best model)
      │
      ▼
Model Serialization (df.pkl, pipe.pkl)
      │
      ▼
Streamlit Web App Deployment
```

---

## 🧹 Data Cleaning

The raw data required extensive cleaning before it could be used:

- **Removed null values** across all critical columns
- **Removed duplicate rows**
- **Fixed inconsistent units** — e.g., `Weight` column had `"1.5kg"` as text instead of numeric; stripped units and converted to `float`
- **Standardized categorical text** — inconsistent casing/spacing in brand and OS names
- **Handled placeholder/invalid entries** that didn't belong in numeric columns

---

## 🛠️ Feature Engineering

This was one of the most involved parts of the project — several raw columns were **decomposed into multiple meaningful features**:

### 1. `ScreenResolution` → `Touchscreen`, `Ips`, `ppi`
- Extracted whether a laptop is touchscreen or has an IPS panel using string matching
- Parsed the resolution string (e.g., `"1920x1080"`) to extract width and height
- Computed **ppi (pixels per inch)** using the formula:

```
ppi = sqrt(X_resolution² + Y_resolution²) / Screen_Size_in_inches
```

### 2. `Cpu` → `Cpu brand`
- Extracted processor name and simplified into brand-level categories (e.g., `Intel Core i5`, `Intel Core i7`, `AMD Processor`, `Other Intel Processor`)

### 3. `Memory` → `HDD`, `SSD`
- Original column had mixed formats like `"256GB SSD"`, `"1TB HDD"`, `"128GB SSD + 1TB HDD"`
- Converted `TB` to `GB` (×1000) using string replacement
- Split dual-storage entries on `"+"` into separate components
- Used regex/string parsing to extract numeric SSD and HDD capacities into two clean numeric columns

### 4. `Gpu` → `Gpu brand`
- Extracted GPU manufacturer (Intel, Nvidia, AMD) from the full GPU string

### 5. `OpSys` → `os`
- Grouped multiple OS variants into simplified categories: `Windows`, `Mac`, `Others/No OS/Linux`

### 6. Target Transformation
- `Price` was **right-skewed**, so applied a **log transformation** (`np.log(Price)`) to normalize the distribution and improve linear model performance
- Predictions are converted back to real price using `np.exp()`

---

## 📈 Exploratory Data Analysis (EDA)

Performed detailed univariate, bivariate, and correlation analysis to understand what drives laptop pricing:

- **Distribution plots** of `Price` before and after log transformation
- **Boxplots & swarmplots** of `Price` across categorical features (`Company`, `TypeName`, `fuel-type`-style categorical splits)
- **Bar plots** comparing average price across brands, CPU types, and GPU brands
- **Scatter plots** examining relationships between `ppi`, `Ram`, `Weight` and `Price`
- **Correlation heatmap** to identify which numeric features most strongly influence price, and to check for multicollinearity before modeling
- Identified and visually inspected **outliers** (e.g., extreme high-end laptops skewing distributions)

**Key insights from EDA:**
- RAM and SSD capacity showed strong positive correlation with price
- Touchscreen and IPS display laptops commanded a price premium
- Certain brands (e.g., Apple) had systematically higher prices independent of specs
- Gaming/Ultrabook categories priced significantly higher than standard notebooks

---

## 🤖 Model Building

### Preprocessing Pipeline
- **Encoding:** `OneHotEncoder` for categorical columns (`Company`, `TypeName`, `Cpu brand`, `Gpu brand`, `os`)
- **Column Transformer** to apply encoding only to categorical columns while passing numeric columns through
- **Train-Test Split** performed before scaling to avoid data leakage
- **Scaling** applied where required (fit on train, transform on test only)

### Models Trained & Compared

| Category | Algorithms |
|---|---|
| Linear Models | `LinearRegression`, `Ridge`, `Lasso` |
| Instance-based | `KNeighborsRegressor` |
| Tree-based | `DecisionTreeRegressor`, `RandomForestRegressor` |
| Ensemble Boosting | `GradientBoostingRegressor`, `AdaBoostRegressor`, `XGBRegressor` |
| Ensemble Bagging | `ExtraTreesRegressor` |
| Kernel-based | `SVR` |

Each model was evaluated using:
- **R² Score** (primary metric)
- **Mean Absolute Error (MAE)**

### Hyperparameter Tuning
- Used `GridSearchCV` with cross-validation to tune the best-performing models
- Compared tuned vs. default performance to select the final production model

### Final Model
The best-performing model was selected based on R² score and generalization on the test set, then wrapped inside a complete **Scikit-learn Pipeline** (`ColumnTransformer` + regressor) for seamless deployment.

---

## 🏆 Results

- Achieved an **R² accuracy of ~85–90%** on the test set with the best-performing model
- Model generalizes well with minimal overfitting between train and test performance
- Final pipeline handles raw categorical + numeric input directly — no manual preprocessing needed at inference time

---

## 🧰 Tech Stack

**Language:** Python

**Libraries:**
- Data Handling: `Pandas`, `NumPy`
- Visualization: `Matplotlib`, `Seaborn`
- Machine Learning: `Scikit-learn`, `XGBoost`
- Model Serialization: `Pickle`
- Deployment: `Streamlit`, Streamlit Community Cloud
- Version Control: `Git`, `GitHub`

---

## 📁 Project Structure

```
Laptop_Price_Predictor/
│
├── app.py                     # Streamlit web application
├── laptop_data.csv            # Raw dataset
├── cleaned_laptop_data.csv    # Cleaned dataset after preprocessing
├── df.pkl                     # Serialized cleaned DataFrame (used for dropdown options)
├── pipe.pkl                   # Serialized trained pipeline (preprocessing + model)
├── Laptop_Price_Predictor.ipynb  # Jupyter notebook with full analysis & model building
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Usage

### Run locally

```bash
# Clone the repository
git clone https://github.com/divay0407/Laptop_Price_Predictor.git
cd Laptop_Price_Predictor

# Install dependencies
pip install streamlit pandas numpy scikit-learn xgboost

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ How the App Works

1. User selects laptop specifications through dropdowns and input fields (Brand, RAM, CPU, GPU, Storage, Display, OS, etc.)
2. The app computes derived features like `ppi` from user-selected screen resolution and size
3. Input is passed through the saved pipeline (`pipe.pkl`), which handles encoding internally
4. The model outputs a **log-transformed price prediction**
5. The app applies `np.exp()` to convert it back into an actual price and displays it to the user

---

## 🚀 Future Improvements

- Add more granular CPU/GPU generation-level features for finer price sensitivity
- Experiment with stacking/ensembling top-performing models for higher accuracy
- Add confidence intervals around predictions instead of a single point estimate
- Expand dataset with more recent laptop listings to keep predictions current
- Add a feature importance visualization inside the app for explainability

---

## 👤 Author

**Divay**
B.Tech Electrical Engineering, NSUT
Transitioning into Data Analytics / ML

- GitHub: [@divay0407](https://github.com/divay0407)

---

⭐ If you found this project useful, consider giving it a star on GitHub!
