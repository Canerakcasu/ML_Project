import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ============================================================================

# Machine learning modules
from sklearn.model_selection import train_test_split, GridSearchCV  # Veri bölme ve hiperparametre optimizasyonu
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder  # Veri ön işleme
from sklearn.compose import ColumnTransformer  # Farklı sütun türleri için farklı işlemler
from sklearn.pipeline import Pipeline  # İşlem zinciri oluşturma
from sklearn.impute import SimpleImputer  # Eksik değer doldurma
from sklearn.utils.multiclass import unique_labels  # Benzersiz etiketleri bulma

# Makine öğrenimi modelleri
from sklearn.ensemble import RandomForestClassifier  # Sınıflandırma için Random Forest
from sklearn.linear_model import LinearRegression  # Regresyon için Linear Regression

# Değerlendirme metrikleri
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Sınıflandırma metrikleri
from sklearn.metrics import mean_squared_error, r2_score  # Regresyon metrikleri

# ============================================================================
plt.style.use('dark_background')  # Koyu tema kullan
sns.set_palette("Spectral")  # Renk paleti ayarla

# Grafiklerin kaydedileceği klasör yolu - KENDİ DOSYA YOLUNUZLA DEĞİŞTİRİN
SAVE_DIRECTORY = r"C:\Users\caner\OneDrive\Desktop\ML_Project\ML_Project\LOL-Champion-Analysis\graps"

# Grafik kaydetme klasörünü oluştur (eğer yoksa)
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
    print(f"Directory created: {SAVE_DIRECTORY}")

def clean_filename_from_title(title):
    """
    Grafik başlıklarından dosya sistemine uygun isimler oluşturur.
    Geçersiz karakterleri kaldırır ve dosya adını temizler.
    """
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    filename = ''.join(c for c in title if c in valid_chars)
    filename = filename.replace(' ', '_').replace(':', '').replace('/', '').replace('\\', '')
    return filename + ".png"

def save_current_plot(directory, plot_title_override=None):
    """
    Mevcut matplotlib grafiğini belirtilen klasöre kaydeder.
    """
    fig = plt.gcf()  # Mevcut figürü al
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

# ============================================================================
# 1. VERİ YÜKLEME VE İLK KEŞIF
# ============================================================================
print("--- 1. Data Loading and Initial Exploration ---")

# CSV dosyasının tam yolunu buraya yazın - KENDİ DOSYA YOLUNUZLA DEĞİŞTİRİN
file_path = r"C:\Users\caner\OneDrive\Desktop\ML_Project\ML_Project\LOL-Champion-Analysis\CSV\champions.csv"

# Veri setini yükleme ve hata kontrolü
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded '{os.path.basename(file_path)}'.\n")
except FileNotFoundError:
    print(f"ERROR: File '{file_path}' not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# Veri setinin temel bilgilerini görüntüleme
print("First 5 rows of the dataset:")
print(df.head())  # İlk 5 satırı göster

print("\nDataset Information (Column Types, Missing Values):")
df.info()  # Veri türleri ve eksik değer bilgileri

print("\nStatistical Summary of Numerical Features:")
print(df.describe())  # Sayısal özelliklerin istatistiksel özeti

print("\nNumber of Missing Values in Columns:")
print(df.isnull().sum())  # Her sütundaki eksik değer sayısı

# ============================================================================
# HEDEF DEĞİŞKEN TANIMI
# ============================================================================
# Tahmin etmeye çalışacağınız hedef değişken ve problem tipi
# Örnekler: 'Role', 'Class', 'HP', 'AttackDamage'
target_column = 'Role'
problem_type = 'classification'  # 'classification' veya 'regression'

# LabelEncoder'ı global olarak tanımla (sonraki bölümlerde kullanılacak)
label_encoder = None

# ============================================================================
# 2. KEŞİFSEL VERİ ANALİZİ (EDA)
# ============================================================================
print("\n--- 2. Exploratory Data Analysis (EDA) ---")

# Sayısal ve kategorik sütunları ayırma
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Hedef sütunu özellik listelerinden çıkarma
if target_column:
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

# Sayısal özelliklerin histogramlarını çizme
print("\nDistributions of Numerical Features:")
for col in numerical_cols:
    if col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=20)  # Histogram + Kernel Density Estimation
        title = f'Distribution of {col}'
        plt.title(title, fontsize=15)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(alpha=0.3)
        save_current_plot(SAVE_DIRECTORY, title)
        plt.show()

# Sayısal özellikler arasındaki korelasyon matrisi
if len(numerical_cols) > 1:
    print("\nCorrelation Matrix of Numerical Features:")
    plt.figure(figsize=(14, 10))
    
    # Korelasyon için kullanılacak sütunları belirleme
    corr_df_cols = numerical_cols[:]
    if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        corr_df_cols.append(target_column)

    # Geçerli sayısal sütunları filtreleme
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

# Kategorik özelliklerin dağılım grafikleri
print("\nDistributions of Categorical Features:")
MAX_CATEGORIES_TO_PLOT_FULLY = 35  # Tam olarak gösterilecek maksimum kategori sayısı
TOP_N_CATEGORIES = 25  # Çok kategori varsa gösterilecek en popüler N kategori

for col in categorical_cols:
    if col in df.columns:
        num_unique = df[col].nunique()  # Benzersiz değer sayısı
        value_counts = df[col].value_counts()  # Değer sayıları
        plot_title = f'Distribution of {col}'

        if num_unique == 0:
            print(f"Feature '{col}' has no unique values. Skipping plot.")
            continue
        
        # Grafik boyutunu kategori sayısına göre ayarlama
        plt.figure(figsize=(12, max(7, num_unique * 0.25 if num_unique <= MAX_CATEGORIES_TO_PLOT_FULLY else TOP_N_CATEGORIES * 0.3)))

        # Çok fazla kategori varsa sadece en popüler N tanesini göster
        if num_unique > MAX_CATEGORIES_TO_PLOT_FULLY:
            print(f"Feature '{col}' has {num_unique} unique values. Plotting top {TOP_N_CATEGORIES}.")
            plot_title = f'Top {TOP_N_CATEGORIES} Distribution of {col}'
            top_n_data = value_counts.nlargest(TOP_N_CATEGORIES)
            sns.barplot(x=top_n_data.values, y=top_n_data.index, palette="Spectral")
            plt.xlabel('Count', fontsize=12)
            plt.ylabel(col, fontsize=12)
        else:
            # Tüm kategorileri göster
            sns.countplot(y=df[col], order=value_counts.index, palette="Spectral")
            plt.xlabel('Count', fontsize=12)
            plt.ylabel(col, fontsize=12)

        plt.title(plot_title, fontsize=15)
        plt.yticks(fontsize=9)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        save_current_plot(SAVE_DIRECTORY, plot_title)
        plt.show()

# ============================================================================
# 3. VERİ TEMİZLEME VE ÖN İŞLEME
# ============================================================================
print("\n--- 3. Data Cleaning and Preprocessing ---")
df_processed = df.copy()  # Orijinal veriyi korumak için kopya oluştur

# Sayısal özelliklerdeki eksik değerleri medyan ile doldurma
for col in numerical_cols:
    if col in df_processed.columns and df_processed[col].isnull().any():
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in numerical feature '{col}' with median ({median_val}).")

# Kategorik özelliklerdeki eksik değerleri mod (en sık görülen değer) ile doldurma
for col in categorical_cols:
     if col in df_processed.columns and df_processed[col].isnull().any():
        mode_val = df_processed[col].mode()[0]
        df_processed[col].fillna(mode_val, inplace=True)
        print(f"Filled missing values in categorical feature '{col}' with mode ('{mode_val}').")

# Hedef değişkendeki eksik değerleri doldurma
if target_column and target_column in df_processed.columns and df_processed[target_column].isnull().any():
    print(f"Warning: Target column '{target_column}' has {df_processed[target_column].isnull().sum()} missing values.")
    if pd.api.types.is_numeric_dtype(df_processed[target_column]):
        # Sayısal hedef değişken için medyan kullan
        median_target_val = df_processed[target_column].median()
        df_processed[target_column].fillna(median_target_val, inplace=True)
        print(f"Filled missing target values in '{target_column}' with median ({median_target_val}).")
    else:
        # Kategorik hedef değişken için mod kullan
        mode_target_val = df_processed[target_column].mode()[0]
        df_processed[target_column].fillna(mode_target_val, inplace=True)
        print(f"Filled missing target values in '{target_column}' with mode ('{mode_target_val}').")

# ============================================================================
# X VE Y DEĞİŞKENLERİNİ AYIRMA VE LABEL ENCODING
# ============================================================================
if target_column and target_column in df_processed.columns:
    X = df_processed.drop(target_column, axis=1)  # Özellikler (features)
    y = df_processed[target_column]  # Hedef değişken (target)

    # Kategorik hedef değişken için Label Encoding uygulama
    if problem_type == 'classification' and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)  # Metinsel etiketleri sayısal değerlere çevir
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
        
        if label_encoder:
             print(f"\nTarget variable '{target_column}' encoded. Encoded Classes (first 5 if many): {list(label_encoder.classes_)[:5]}")
else:
    print(f"ERROR: Target column '{target_column}' not defined or not found. Model training will be skipped.")
    X, y = None, None

# ============================================================================
# ÖN İŞLEME PIPELINE'LARI OLUŞTURMA
# ============================================================================
if X is not None:
    # X'teki sayısal ve kategorik özellikleri belirleme
    numerical_features_in_X = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features_in_X = X.select_dtypes(include='object').columns.tolist()

    # Sayısal özellikler için pipeline: Eksik değer doldurma + Standardizasyon
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Eksik değerleri medyan ile doldur
        ('scaler', StandardScaler())  # Özellikleri standartlaştır (ortalama=0, std=1)
    ])
    
    # Kategorik özellikler için pipeline: Eksik değer doldurma + One-Hot Encoding
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Eksik değerleri en sık değer ile doldur
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))  # Kategorileri binary değişkenlere çevir
    ])
    
    # Tüm özellikleri işleyen ana preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features_in_X),
            ('cat', categorical_pipeline, categorical_features_in_X)
        ],
        remainder='passthrough'  # Diğer sütunları olduğu gibi bırak
    )
    print("\nPreprocessor defined.")

# ============================================================================
# 4. VERİYİ EĞİTİM VE TEST SETLERİNE BÖLME
# ============================================================================
if X is not None and y is not None:
    print("\n--- 4. Splitting Data into Training and Test Sets ---")
    stratify_y = None
    
    # Stratified sampling için koşulları kontrol etme
    if problem_type == 'classification' and y.nunique() > 1:
        # Her sınıfın en az 2 örneği olup olmadığını kontrol et
        if y.value_counts().min() >= 2:
            stratify_y = y  # Stratified sampling kullan
        else:
            print("Warning: Not enough samples in some classes for stratification after rare class removal. Proceeding without stratification.")

    # Veriyi %80 eğitim, %20 test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_y
    )
    print(f"Training set size: {X_train.shape[0]} samples, Test set size: {X_test.shape[0]} samples")

# ============================================================================
# 5. MODEL EĞİTİMİ VE HİPERPARAMETRE OPTİMİZASYONU
# ============================================================================
best_model = None  # En iyi modeli saklamak için
if X is not None and y is not None and 'X_train' in locals():
    print("\n--- 5. Defining, Training, and Optimizing Models ---")
    model_instance = None
    param_grid = {}
    scoring_metric = ''

    # Problem tipine göre model ve parametreleri seçme
    if problem_type == 'classification':
        model_instance = RandomForestClassifier(random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100],  # Ağaç sayısı
            'model__max_depth': [10, None],    # Maksimum derinlik
            'model__min_samples_leaf': [1, 2]  # Yaprakta minimum örnek sayısı
        }
        scoring_metric = 'accuracy'
    elif problem_type == 'regression':
        model_instance = LinearRegression()
        param_grid = {'model__fit_intercept': [True, False]}  # Kesim noktası kullanılsın mı
        scoring_metric = 'r2'
    else:
        print("Problem type not specified. Skipping model training.")

    if model_instance is not None:
        # Tam pipeline oluşturma: Preprocessing + Model
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_instance)])
        
        # Cross-validation fold sayısını veri boyutuna göre ayarlama
        cv_folds = 3 if X_train.shape[0] < 1000 else 5
        
        # Veri boyutu GridSearchCV için yeterli mi kontrol et
        min_samples_for_cv = cv_folds * max(y_train.value_counts().min() if problem_type == 'classification' else 1, 1)
        
        if X_train.shape[0] < 20 or (problem_type == 'classification' and y_train.nunique() > 0 and y_train.value_counts().min() < cv_folds):
            # Veri az ise varsayılan parametrelerle eğit
            print(f"Warning: Not enough samples or class diversity for GridSearchCV with cv={cv_folds}. Fitting model with default parameters.")
            try:
                best_model = full_pipeline.fit(X_train, y_train)
                print(f"{model_instance.__class__.__name__} trained with default parameters.")
            except Exception as e:
                print(f"Error during model fitting with default parameters: {e}")
                best_model = None
        else:
            # GridSearchCV ile hiperparametre optimizasyonu
            grid_search = GridSearchCV(
                full_pipeline, param_grid, cv=cv_folds, 
                scoring=scoring_metric, n_jobs=-1, verbose=1
            )
            print(f"\nTraining {model_instance.__class__.__name__} with GridSearchCV...")
            try:
                grid_search.fit(X_train, y_train)
                print(f"Best parameters for {model_instance.__class__.__name__}: {grid_search.best_params_}")
                best_model = grid_search.best_estimator_
            except Exception as e:
                print(f"Error during GridSearchCV: {e}. Model training may have failed.")
                best_model = None
else:
     print("\nSkipping model training as data is not prepared (X, y, or X_train not available).")

# ============================================================================
# 6. MODEL DEĞERLENDİRMESİ
# ============================================================================
if X is not None and y is not None and 'X_test' in locals() and best_model is not None:
    print("\n--- 6. Evaluating the Model ---")
    y_pred = best_model.predict(X_test)  # Test seti üzerinde tahmin yap
    model_name_for_plot = best_model.named_steps['model'].__class__.__name__

    if problem_type == 'classification':
        # Sınıflandırma metrikleri
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")

        # Test ve tahmin setindeki benzersiz etiketleri bul
        labels_in_test = unique_labels(y_test, y_pred)
        
        # Etiket isimlerini belirleme
        target_names_display = [str(i) for i in labels_in_test]  # Varsayılan: sayısal etiketler
        if label_encoder is not None and hasattr(label_encoder, 'classes_'):
            try:
                # Sayısal etiketleri orijinal metin etiketlerine dönüştür
                target_names_display = list(label_encoder.inverse_transform(labels_in_test))
            except ValueError:
                print(f"Warning: Could not decode all labels. Using numerical labels for the report.")
            except Exception as e_le:
                print(f"Warning: An unexpected error occurred during label decoding: {e_le}. Using numerical labels.")

        # Sınıflandırma raporu
        print(classification_report(
            y_test, y_pred,
            labels=labels_in_test,
            target_names=target_names_display,
            zero_division=0
        ))

        # Confusion Matrix (Karmaşıklık Matrisi)
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=labels_in_test)
        plt.figure(figsize=(max(8, len(target_names_display)*0.8), max(6, len(target_names_display)*0.6)))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Spectral_r",
                    xticklabels=target_names_display,
                    yticklabels=target_names_display,
                    annot_kws={"size": 10})
        title = f'Confusion Matrix for {model_name_for_plot}'
        plt.title(title, fontsize=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        save_current_plot(SAVE_DIRECTORY, title)
        plt.show()

    elif problem_type == 'regression':
        # Regresyon metrikleri
        mse = mean_squared_error(y_test, y_pred)  # Ortalama Kare Hatası
        rmse = np.sqrt(mse)  # Kök Ortalama Kare Hatası
        r2 = r2_score(y_test, y_pred)  # R-kare değeri
        
        print(f"\nMean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R2 Score): {r2:.4f}")

        # Gerçek vs Tahmin edilen değerler scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='w', linewidth=0.5)
        
        # Mükemmel tahmin çizgisi (45 derece)
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