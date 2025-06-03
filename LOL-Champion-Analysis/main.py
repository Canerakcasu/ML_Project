import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # Dosya ve klasör işlemleri için eklendi

# Machine learning modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Example models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

# --- Plot Styling and Save Configuration ---
plt.style.use('dark_background')
sns.set_palette("Spectral")

# Türkçe Yorum: Grafiklerin kaydedileceği klasör yolu.
SAVE_DIRECTORY = r"C:\Users\caner\OneDrive\Desktop\ML_Project\ML_Project\LOL-Champion-Analysis\graps"

# Türkçe Yorum: Grafik kaydetme klasörünü oluştur (eğer yoksa).
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
    print(f"Directory created: {SAVE_DIRECTORY}")

def clean_filename_from_title(title):
    """Cleans a plot title to create a valid filename."""
    # Türkçe Yorum: Grafik başlıklarından dosya sistemine uygun isimler oluşturur. Özel karakterleri kaldırır, boşlukları alt çizgi yapar.
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    filename = ''.join(c for c in title if c in valid_chars)
    filename = filename.replace(' ', '_').replace(':', '').replace('/', '').replace('\\', '')
    return filename + ".png"

def save_current_plot(directory, plot_title_override=None):
    """Saves the current matplotlib plot to the specified directory."""
    # Türkçe Yorum: Mevcut grafiği belirtilen klasöre kaydeder.
    fig = plt.gcf() # Get current figure
    plot_title = plot_title_override if plot_title_override else plt.gca().get_title()

    if plot_title:
        filename = clean_filename_from_title(plot_title)
        save_path = os.path.join(directory, filename)
        try:
            fig.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot {save_path}: {e}")
    else:
        print("Plot title not found, cannot save plot.")

# --- 1. Data Loading and Initial Exploration ---
print("--- 1. Data Loading and Initial Exploration ---")
# === FILE PATH ===
# Türkçe Yorum: Kullandığınız CSV dosyasının tam yolunu buraya yazın. Örnekteki gibi mutlak yol kullanmanız önerilir.
file_path = r"C:\Users\caner\OneDrive\Desktop\ML_Project\ML_Project\LOL-Champion-Analysis\CSV\champions.csv"

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded '{os.path.basename(file_path)}'.\n") # Sadece dosya adını göster
except FileNotFoundError:
    print(f"ERROR: File '{file_path}' not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information (Column Types, Missing Values):")
df.info()
print("\nStatistical Summary of Numerical Features:")
print(df.describe())
print("\nNumber of Missing Values in Columns:")
print(df.isnull().sum())

# --- Placeholder for Target Variable Identification ---
# Türkçe Yorum: Tahmin etmeye çalışacağınız hedef değişkeni (sütun adı) ve problem tipini (classification/regression) tanımlayın.
target_column = None  # Example: 'Role', 'Class', 'HP', 'AttackDamage'
problem_type = None # 'classification' or 'regression' # Example: 'classification'

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n--- 2. Exploratory Data Analysis (EDA) ---")

numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Remove target column from feature lists if it's defined
if target_column:
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

# Histograms for numerical features
print("\nDistributions of Numerical Features:")
for col in numerical_cols:
    if col in df.columns: # Check if column still exists (e.g. not removed if it was target)
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=20)
        title = f'Distribution of {col}'
        plt.title(title, fontsize=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(alpha=0.3)
        save_current_plot(SAVE_DIRECTORY, title) # Grafiği kaydet
        plt.show()

# Correlation Matrix for numerical features
if len(numerical_cols) > 1:
    print("\nCorrelation Matrix of Numerical Features:")
    plt.figure(figsize=(14, 10)) # Increased size for better readability
    # Include target if numeric for correlation analysis
    corr_df_cols = numerical_cols[:] # Create a copy
    if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        corr_df_cols.append(target_column)

    valid_numerical_for_corr = [col for col in corr_df_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if len(valid_numerical_for_corr) > 1:
        correlation_matrix = df[valid_numerical_for_corr].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="Spectral", fmt=".2f", linewidths=.5, annot_kws={"size": 8}) # Reduced annot size
        title = 'Correlation Matrix of Numerical Features'
        plt.title(title, fontsize=15)
        plt.xticks(rotation=45, ha='right', fontsize=10) # Adjusted font size
        plt.yticks(rotation=0, fontsize=10) # Adjusted font size
        plt.tight_layout()
        save_current_plot(SAVE_DIRECTORY, title) # Grafiği kaydet
        plt.show()
    else:
        print("Not enough valid numerical features to create a correlation matrix.")

# Count plots for categorical features (Improved Readability)
print("\nDistributions of Categorical Features:")
MAX_CATEGORIES_TO_PLOT_FULLY = 35 # Bu eşik değerin altındaki kategori sayıları için tüm kategoriler çizilir
TOP_N_CATEGORIES = 25             # Bu eşik değerin üzerindeyse en sık görülen ilk N kategori çizilir

for col in categorical_cols:
    if col in df.columns: # Check if column still exists
        num_unique = df[col].nunique()
        value_counts = df[col].value_counts()
        plot_title = f'Distribution of {col}'

        if num_unique == 0:
            print(f"Feature '{col}' has no unique values. Skipping plot.")
            continue
        
        plt.figure(figsize=(12, max(7, num_unique * 0.25 if num_unique <= MAX_CATEGORIES_TO_PLOT_FULLY else TOP_N_CATEGORIES * 0.3)))

        if num_unique > MAX_CATEGORIES_TO_PLOT_FULLY:
            print(f"Feature '{col}' has {num_unique} unique values. Plotting top {TOP_N_CATEGORIES}.")
            plot_title = f'Top {TOP_N_CATEGORIES} Distribution of {col}'
            top_n_data = value_counts.nlargest(TOP_N_CATEGORIES)
            sns.barplot(x=top_n_data.values, y=top_n_data.index, palette="Spectral")
            plt.xlabel('Count', fontsize=12)
            plt.ylabel(col, fontsize=12)
        else:
            sns.countplot(y=df[col], order=value_counts.index, palette="Spectral")
            plt.xlabel('Count', fontsize=12)
            plt.ylabel(col, fontsize=12)

        plt.title(plot_title, fontsize=15)
        # plt.xticks(rotation=45, ha='right') # For horizontal bar plots, y-axis labels are more critical
        plt.yticks(fontsize=9) # Adjust y-tick font size for readability
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        save_current_plot(SAVE_DIRECTORY, plot_title) # Grafiği kaydet
        plt.show()


# --- 3. Data Cleaning and Preprocessing ---
print("\n--- 3. Data Cleaning and Preprocessing ---")
df_processed = df.copy()

for col in numerical_cols: # Use the updated numerical_cols (without target)
    if col in df_processed.columns and df_processed[col].isnull().any(): # Check existence
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in numerical feature '{col}' with median ({median_val}).")

for col in categorical_cols: # Use the updated categorical_cols (without target)
     if col in df_processed.columns and df_processed[col].isnull().any(): # Check existence
        mode_val = df_processed[col].mode()[0]
        df_processed[col].fillna(mode_val, inplace=True)
        print(f"Filled missing values in categorical feature '{col}' with mode ('{mode_val}').")

if target_column and target_column in df_processed.columns and df_processed[target_column].isnull().any():
    print(f"Warning: Target column '{target_column}' has {df_processed[target_column].isnull().sum()} missing values.")
    # Example: df_processed.dropna(subset=[target_column], inplace=True)
    # Or impute if appropriate for the target variable
    if pd.api.types.is_numeric_dtype(df_processed[target_column]):
        df_processed[target_column].fillna(df_processed[target_column].median(), inplace=True)
        print(f"Filled missing target values in '{target_column}' with median.")
    else:
        df_processed[target_column].fillna(df_processed[target_column].mode()[0], inplace=True)
        print(f"Filled missing target values in '{target_column}' with mode.")
    

if target_column and target_column in df_processed.columns:
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]

    if problem_type == 'classification' and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"\nTarget variable '{target_column}' encoded. Classes: {list(label_encoder.classes_)}")
else:
    print(f"ERROR: Target column '{target_column}' not defined or not found. Model training will be skipped.")
    X, y = None, None

if X is not None:
    # Update feature lists based on X after dropping target
    numerical_features_in_X = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features_in_X = X.select_dtypes(include='object').columns.tolist()

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)) # sparse_output=False for easier inspection if needed
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features_in_X),
            ('cat', categorical_pipeline, categorical_features_in_X)
        ],
        remainder='passthrough'
    )
    print("\nPreprocessor defined.")

# --- 4. Splitting Data ---
if X is not None and y is not None:
    print("\n--- 4. Splitting Data into Training and Test Sets ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=(y if problem_type == 'classification' and pd.Series(y).nunique() > 1 else None))
    print(f"Training set size: {X_train.shape[0]} samples, Test set size: {X_test.shape[0]} samples")

# --- 5. Model Training ---
if X is not None and y is not None:
    print("\n--- 5. Defining, Training, and Optimizing Models ---")
    model_instance = None
    param_grid = {}
    scoring_metric = ''

    if problem_type == 'classification':
        model_instance = RandomForestClassifier(random_state=42)
        param_grid = {'model__n_estimators': [50, 100], 'model__max_depth': [10, 20, None], 'model__min_samples_leaf': [1, 2, 4]}
        scoring_metric = 'accuracy'
    elif problem_type == 'regression':
        model_instance = LinearRegression()
        param_grid = {'model__fit_intercept': [True, False]} # Example
        scoring_metric = 'r2'
    else:
        print("Problem type not specified. Skipping model training.")

    if model_instance:
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_instance)])
        cv_folds = 3 if X_train.shape[0] < 1000 else 5
        grid_search = GridSearchCV(full_pipeline, param_grid, cv=cv_folds, scoring=scoring_metric, n_jobs=-1, verbose=1)
        print(f"\nTraining {model_instance.__class__.__name__} with GridSearchCV...")
        try:
            grid_search.fit(X_train, y_train)
            print(f"Best parameters for {model_instance.__class__.__name__}: {grid_search.best_params_}")
            best_model = grid_search.best_estimator_
        except Exception as e:
            print(f"Error during GridSearchCV: {e}. Model training may have failed.")
            best_model = None # Ensure best_model is None if training fails

# --- 6. Model Evaluation ---
if X is not None and y is not None and 'best_model' in locals() and best_model is not None:
    print("\n--- 6. Evaluating the Model ---")
    y_pred = best_model.predict(X_test)
    model_name_for_plot = best_model.named_steps['model'].__class__.__name__


    if problem_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        target_names_report = None
        if 'label_encoder' in locals() and hasattr(label_encoder, 'classes_'):
            try:
                # Ensure y_test and y_pred are in the original encoded form for report if target_names are used
                target_names_report = label_encoder.classes_
            except Exception as e:
                print(f"Could not get class names from label_encoder: {e}")
        # print(classification_report(y_test, y_pred, target_names=target_names_report if target_names_report is not None else None))
        print(classification_report(y_test, y_pred, target_names = [str(cls_name) for cls_name in target_names_report] if target_names_report is not None else None, zero_division=0))


        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7)) # Adjusted size
        sns.heatmap(cm, annot=True, fmt='d', cmap="Spectral_r",
                    xticklabels=target_names_report if target_names_report is not None else 'auto',
                    yticklabels=target_names_report if target_names_report is not None else 'auto',
                    annot_kws={"size": 10}) # Adjust annot size
        title = f'Confusion Matrix for {model_name_for_plot}'
        plt.title(title, fontsize=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        save_current_plot(SAVE_DIRECTORY, title) # Grafiği kaydet
        plt.show()

    elif problem_type == 'regression':
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f"\nMean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R2 Score): {r2:.4f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='w', linewidth=0.5)
        min_val = min(np.min(y_test) if isinstance(y_test, np.ndarray) else y_test.min(), np.min(y_pred) if isinstance(y_pred, np.ndarray) else y_pred.min())
        max_val = max(np.max(y_test) if isinstance(y_test, np.ndarray) else y_test.max(), np.max(y_pred) if isinstance(y_pred, np.ndarray) else y_pred.max())

        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', lw=2)
        title = f'Actual vs. Predicted Values for {model_name_for_plot}'
        plt.title(title, fontsize=15)
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.grid(alpha=0.3)
        save_current_plot(SAVE_DIRECTORY, title) # Grafiği kaydet
        plt.show()
else:
    print("\nSkipping model evaluation as model training was not completed, target not set, or 'best_model' not defined.")

print("\n--- Project Code Execution Completed ---")