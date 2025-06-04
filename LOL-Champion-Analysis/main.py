import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # Dosya ve klasör işlemleri için eklendi

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
# Türkçe Yorum: Bu dosya, LOL şampiyon verilerini analiz etmek ve makine öğrenimi modelleri oluşturmak için kullanılacaktır.
# Machine learning modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.multiclass import unique_labels
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
SAVE_DIRECTORY = r"C:\Users\caner\OneDrive\Desktop\ML_Project\ML_Project\LOL-Champion-Analysis\graps" # Kendi dosya yolunuzla değiştirin

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
file_path = r"C:\Users\caner\OneDrive\Desktop\ML_Project\ML_Project\LOL-Champion-Analysis\CSV\champions.csv" # Kendi dosya yolunuzla değiştirin

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
# Türkçe Yorum: Tahmin etmeye çalışacağınız hedef değişken ve problem tipi.
target_column ='Role' # Örnek: 'Role', 'Class', 'HP', 'AttackDamage'
problem_type = 'classification'  # 'classification' veya 'regression'

# LabelEncoder'ı daha üst bir kapsamda tanımla (Bölüm 6'da kullanılacak)
label_encoder = None

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
    if col in df.columns: # Check if column still exists
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=20)
        title = f'Distribution of {col}'
        plt.title(title, fontsize=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(alpha=0.3)
        save_current_plot(SAVE_DIRECTORY, title)
        plt.show()

# Correlation Matrix for numerical features
if len(numerical_cols) > 1:
    print("\nCorrelation Matrix of Numerical Features:")
    plt.figure(figsize=(14, 10))
    corr_df_cols = numerical_cols[:]
    if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        corr_df_cols.append(target_column)

    valid_numerical_for_corr = [col for col in corr_df_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if len(valid_numerical_for_corr) > 1:
        correlation_matrix = df[valid_numerical_for_corr].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="Spectral", fmt=".2f", linewidths=.5, annot_kws={"size": 8})
        title = 'Correlation Matrix of Numerical Features'
        plt.title(title, fontsize=15)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        save_current_plot(SAVE_DIRECTORY, title)
        plt.show()
    else:
        print("Not enough valid numerical features to create a correlation matrix.")

# Count plots for categorical features
print("\nDistributions of Categorical Features:")
MAX_CATEGORIES_TO_PLOT_FULLY = 35
TOP_N_CATEGORIES = 25

for col in categorical_cols:
    if col in df.columns:
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
        plt.yticks(fontsize=9)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        save_current_plot(SAVE_DIRECTORY, plot_title)
        plt.show()


# --- 3. Data Cleaning and Preprocessing ---
print("\n--- 3. Data Cleaning and Preprocessing ---")
df_processed = df.copy()

# Özelliklerdeki eksik değerleri doldurma
for col in numerical_cols:
    if col in df_processed.columns and df_processed[col].isnull().any():
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in numerical feature '{col}' with median ({median_val}).")

for col in categorical_cols:
     if col in df_processed.columns and df_processed[col].isnull().any():
        mode_val = df_processed[col].mode()[0]
        df_processed[col].fillna(mode_val, inplace=True)
        print(f"Filled missing values in categorical feature '{col}' with mode ('{mode_val}').")

# Hedef değişkendeki eksik değerleri doldurma
if target_column and target_column in df_processed.columns and df_processed[target_column].isnull().any():
    print(f"Warning: Target column '{target_column}' has {df_processed[target_column].isnull().sum()} missing values.")
    if pd.api.types.is_numeric_dtype(df_processed[target_column]):
        median_target_val = df_processed[target_column].median()
        df_processed[target_column].fillna(median_target_val, inplace=True)
        print(f"Filled missing target values in '{target_column}' with median ({median_target_val}).")
    else:
        mode_target_val = df_processed[target_column].mode()[0]
        df_processed[target_column].fillna(mode_target_val, inplace=True)
        print(f"Filled missing target values in '{target_column}' with mode ('{mode_target_val}').")
    

# X ve y'yi ayırma ve Label Encoding (gerekirse)
if target_column and target_column in df_processed.columns:
    X = df_processed.drop(target_column, axis=1)
    y = df_processed[target_column]

    if problem_type == 'classification' and y.dtype == 'object':
        label_encoder = LabelEncoder() # Üst kapsamdaki label_encoder'a atama yapılıyor
        y = label_encoder.fit_transform(y)
        y = pd.Series(y)  # Stratify uyumluluğu için Series'e dönüştür

        # Nadir sınıfları kaldırma (stratify <2 örnekli sınıfları işleyemez)
        class_counts = y.value_counts()
        rare_classes = class_counts[class_counts < 2].index
        if len(rare_classes) > 0:
            print(f"Removing classes with <2 samples (encoded): {list(rare_classes)}")
            mask = ~y.isin(rare_classes)
            X = X[mask]
            y = y[mask]
            y = y.reset_index(drop=True)
            X = X.reset_index(drop=True)
        
        if label_encoder: # label_encoder'ın fit edilip edilmediğini kontrol et
             print(f"\nTarget variable '{target_column}' encoded. Encoded Classes (first 5 if many): {list(label_encoder.classes_)[:5]}")
else:
    print(f"ERROR: Target column '{target_column}' not defined or not found. Model training will be skipped.")
    X, y = None, None

# --- Preprocessing Pipelines ---
if X is not None:
    numerical_features_in_X = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features_in_X = X.select_dtypes(include='object').columns.tolist()

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)) # sparse_output=False
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features_in_X),
            ('cat', categorical_pipeline, categorical_features_in_X)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )
    print("\nPreprocessor defined.")

# --- 4. Splitting Data ---
if X is not None and y is not None:
    print("\n--- 4. Splitting Data into Training and Test Sets ---")
    stratify_y = None
    if problem_type == 'classification' and y.nunique() > 1: # Stratify sadece birden fazla sınıf varsa ve problem classification ise mantıklı
        # y'nin Series olduğundan emin ol (yukarıda yapıldı)
        # Stratify için y'nin en az 2 örneği olan sınıflara sahip olması gerekir (yukarıda nadir sınıflar kaldırıldı)
        if y.value_counts().min() >= 2 : # Eğer en az örnek sayısı olan sınıf bile 2 veya daha fazlaysa
            stratify_y = y
        else:
            print("Warning: Not enough samples in some classes for stratification after rare class removal. Proceeding without stratification.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_y
    )
    print(f"Training set size: {X_train.shape[0]} samples, Test set size: {X_test.shape[0]} samples")

# --- 5. Model Training ---
best_model = None # best_model'ı burada ilklendir
if X is not None and y is not None and 'X_train' in locals(): # X_train'in varlığını kontrol et
    print("\n--- 5. Defining, Training, and Optimizing Models ---")
    model_instance = None
    param_grid = {}
    scoring_metric = ''

    if problem_type == 'classification':
        model_instance = RandomForestClassifier(random_state=42)
        param_grid = {'model__n_estimators': [50, 100], 'model__max_depth': [10, None], 'model__min_samples_leaf': [1, 2]}
        scoring_metric = 'accuracy'
    elif problem_type == 'regression':
        model_instance = LinearRegression()
        param_grid = {'model__fit_intercept': [True, False]} # Örnek
        scoring_metric = 'r2'
    else:
        print("Problem type not specified. Skipping model training.")

    if model_instance is not None:
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_instance)])
        cv_folds = 3 if X_train.shape[0] < 1000 else 5 # Daha küçük veri setleri için CV katlamasını azalt
        
        # Eğer çok az örnek varsa GridSearchCV'yi çalıştırma veya CV'yi ayarla
        min_samples_for_cv = cv_folds * max(y_train.value_counts().min() if problem_type == 'classification' else 1, 1) # Sınıflandırma için en küçük sınıfın boyutu
        
        if X_train.shape[0] < 20 or (problem_type == 'classification' and y_train.nunique() > 0 and y_train.value_counts().min() < cv_folds) : # Az örnek veya CV için yetersiz sınıf örneği
            print(f"Warning: Not enough samples or class diversity for GridSearchCV with cv={cv_folds}. Fitting model with default parameters.")
            try:
                best_model = full_pipeline.fit(X_train, y_train)
                print(f"{model_instance.__class__.__name__} trained with default parameters.")
            except Exception as e:
                print(f"Error during model fitting with default parameters: {e}")
                best_model = None
        else:
            grid_search = GridSearchCV(full_pipeline, param_grid, cv=cv_folds, scoring=scoring_metric, n_jobs=-1, verbose=1)
            print(f"\nTraining {model_instance.__class__.__name__} with GridSearchCV...")
            try:
                grid_search.fit(X_train, y_train)
                print(f"Best parameters for {model_instance.__class__.__name__}: {grid_search.best_params_}")
                best_model = grid_search.best_estimator_
            except Exception as e:
                print(f"Error during GridSearchCV: {e}. Model training may have failed.")
                best_model = None # Hata durumunda best_model'ı None yap
else:
     print("\nSkipping model training as data is not prepared (X, y, or X_train not available).")


# --- 6. Model Evaluation ---
if X is not None and y is not None and 'X_test' in locals() and best_model is not None: # X_test'in varlığını ve best_model'ın None olmadığını kontrol et
    print("\n--- 6. Evaluating the Model ---")
    y_pred = best_model.predict(X_test)
    model_name_for_plot = best_model.named_steps['model'].__class__.__name__

    if problem_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")

        labels_in_test = unique_labels(y_test, y_pred) # y_test ve y_pred'de bulunan benzersiz etiketler (sayısal)
        
        target_names_display = [str(i) for i in labels_in_test] # Varsayılan: sayısal etiketler
        if label_encoder is not None and hasattr(label_encoder, 'classes_'): # Eğer LabelEncoder kullanıldıysa
            try:
                # labels_in_test içindeki sayısal etiketleri orijinal metin etiketlerine dönüştür
                target_names_display = list(label_encoder.inverse_transform(labels_in_test))
            except ValueError:
                print(f"Warning: Could not decode all labels in `labels_in_test` using label_encoder. "
                      f"This might happen if y_pred contains labels not seen during training. "
                      f"Using numerical labels for the report.")
                # target_names_display [str(i) for i in labels_in_test] olarak kalır
            except Exception as e_le:
                print(f"Warning: An unexpected error occurred during label decoding: {e_le}. Using numerical labels.")
                # target_names_display [str(i) for i in labels_in_test] olarak kalır

        print(classification_report(
            y_test, y_pred,
            labels=labels_in_test, # Raporlanacak etiketler (sayısal)
            target_names=target_names_display, # Gösterilecek etiket isimleri (metin veya sayısal)
            zero_division=0
        ))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=labels_in_test) # labels_in_test'i burada da kullan
        plt.figure(figsize=(max(8, len(target_names_display)*0.8), max(6, len(target_names_display)*0.6))) # Boyutu etiket sayısına göre ayarla
        sns.heatmap(cm, annot=True, fmt='d', cmap="Spectral_r",
                    xticklabels=target_names_display,
                    yticklabels=target_names_display,
                    annot_kws={"size": 10})
        title = f'Confusion Matrix for {model_name_for_plot}'
        plt.title(title, fontsize=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha="right") # Uzun etiketler için
        plt.yticks(rotation=0)
        plt.tight_layout()
        save_current_plot(SAVE_DIRECTORY, title)
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
        min_val = min(np.min(y_test), np.min(y_pred))
        max_val = max(np.max(y_test), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', lw=2)
        title = f'Actual vs. Predicted Values for {model_name_for_plot}'
        plt.title(title, fontsize=15)
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.grid(alpha=0.3)
        save_current_plot(SAVE_DIRECTORY, title)
        plt.show()
else:
    print("\nSkipping model evaluation as model training was not completed, data not available, or 'best_model' is not defined/valid.")

print("\n--- Script End ---")