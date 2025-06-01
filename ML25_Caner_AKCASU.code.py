
import pandas as pd  # Veri manipülasyonu ve CSV dosyası okuma/yazma için
import numpy as np   # Sayısal hesaplamalar için (özellikle RMSE için karekök)
import matplotlib.pyplot as plt  # Temel görselleştirmeler için
import seaborn as sns  # Gelişmiş istatistiksel görselleştirmeler için

# Makine öğrenmesi için scikit-learn kütüphanesinden modüller
from sklearn.model_selection import train_test_split, GridSearchCV  # Veri bölme ve hiperparametre optimizasyonu
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Özellik ölçeklendirme ve kategorik kodlama
from sklearn.compose import ColumnTransformer  # Farklı sütunlara farklı dönüşümler uygulamak için
from sklearn.pipeline import Pipeline  # İşlem adımlarını birleştirmek için

# Regresyon Modelleri
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Değerlendirme Metrikleri
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Görselleştirme için varsayılan stil ayarları (isteğe bağlı)
plt.style.use('ggplot')
sns.set_palette("viridis")

# -----------------------------------------------------------------------------
# 1. Data Loading and Basic Exploration
# -----------------------------------------------------------------------------
print("--- 1. Data Loading and Basic Exploration ---")

# === DOSYA YOLUNUZU KULLANARAK GÜNCELLENDİ ===
file_path = r"C:\Users\caner\OneDrive\Desktop\ML_Project\Pizza-Price.csv"

try:
    pizza_df = pd.read_csv(file_path) # DataFrame adını 'pizza_df' olarak değiştirdim
    print(f"'{file_path}' was successfully loaded.\n")
except FileNotFoundError:
    print(f"ERROR: File '{file_path}' not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    exit()

# !!! ÖNEMLİ: CSV DOSYANIZDAKİ GERÇEK SÜTUN ADLARINI KONTROL EDİN !!!
print("Column Names in your CSV file:")
print(pizza_df.columns)
print("Please carefully review the output above and ensure that the column names\n"
      "used in the code below (especially for size, price, etc.) exactly match\n"
      "the names in your file. Update the variables below if necessary.\n")

# Veri setine ilk bakış
print("First 5 rows of the dataset:")
print(pizza_df.head())

print("\nDataset Information (Column Types, Missing Values):")
pizza_df.info()

print("\nStatistical Summary of Numerical Features:")
print(pizza_df.describe())

print("\nNumber of Missing Values in Columns:")
print(pizza_df.isnull().sum())
# Genellikle bu veri setinde eksik değer bulunmaz, ancak bu kontrol önemlidir.

# -----------------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------
print("\n--- 2. Exploratory Data Analysis (EDA) ---")

# === BURAYI GÜNCELLEYİN: CSV'nizdeki gerçek pizza boyutu sütun adı ===
# print(pizza_df.columns) çıktısına bakarak doğru adı girin.
# Örnek: Eğer sütun adınız 'Pizza_Boyutu' ise bunu güncelleyin.
actual_size_column_name = 'Size by Inch'
target_column_name = 'Price' # Hedef değişken

# EDA için kullanılacak özellik listeleri
# Bu listelerdeki adların pizza_df.columns çıktısındaki adlarla eşleştiğinden emin olun.
numerical_features_for_eda = [actual_size_column_name, target_column_name]
categorical_features_for_eda = ['Restaurant', 'Extra Cheeze', 'Extra Mushroom', 'Extra Spicy']

# Sayısal özelliklerin dağılımı (Histogramlar)
print("\nDistributions of Numerical Features:")
for feature_name in numerical_features_for_eda:
    if feature_name in pizza_df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(pizza_df[feature_name], kde=True, bins=10)
        plt.title(f'Distribution of {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Frequency')
        plt.show()
    else:
        print(f"WARNING (EDA): Column '{feature_name}' not found in the dataset.")

# Kategorik özelliklerin dağılımı (Bar Grafikleri)
print("\nDistributions of Categorical Features:")
for feature_name in categorical_features_for_eda:
    if feature_name in pizza_df.columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=pizza_df, y=feature_name, order=pizza_df[feature_name].value_counts().index)
        plt.title(f'Distribution of {feature_name}')
        plt.xlabel('Count')
        plt.ylabel(feature_name)
        plt.tight_layout()
        plt.show()
    else:
        print(f"WARNING (EDA): Column '{feature_name}' not found in the dataset.")

# Boyut ile Fiyat ilişkisi (Saçılım Grafiği)
if actual_size_column_name in pizza_df.columns and target_column_name in pizza_df.columns:
    print(f"\nRelationship between {actual_size_column_name} and {target_column_name}:")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=pizza_df, x=actual_size_column_name, y=target_column_name)
    plt.title(f'{actual_size_column_name} vs {target_column_name}')
    plt.xlabel(actual_size_column_name)
    plt.ylabel(target_column_name)
    plt.show()

# Kategorik özellikler ile Fiyat ilişkisi (Kutu Grafikleri)
print(f"\nRelationship between Categorical Features and {target_column_name}:")
for feature_name in categorical_features_for_eda:
    if feature_name in pizza_df.columns and target_column_name in pizza_df.columns:
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=pizza_df, x=feature_name, y=target_column_name)
        plt.title(f'{feature_name} vs {target_column_name}')
        plt.xlabel(feature_name)
        plt.ylabel(target_column_name)
        plt.xticks(rotation=20, ha='right') # Etiketlerin okunabilirliği için
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------
# 3. Data Preprocessing
# -----------------------------------------------------------------------------
print("\n--- 3. Data Preprocessing ---")

# Hedef değişken ve özelliklerin varlığını son bir kez kontrol et
if target_column_name not in pizza_df.columns:
    print(f"ERROR: Target variable '{target_column_name}' not found in the dataset.")
    exit()

# Bağımsız değişkenler (X) ve hedef değişken (y)
features_X = pizza_df.drop(target_column_name, axis=1)
target_y = pizza_df[target_column_name]

# Model için kullanılacak sayısal ve kategorik özelliklerin adları
# Bu adların features_X.columns (yani pizza_df'den target_column_name çıkarıldıktan sonraki sütunlar)
# içinde bulunduğundan emin olun.
numerical_features_for_model = [actual_size_column_name]
categorical_features_for_model = ['Restaurant', 'Extra Cheeze', 'Extra Mushroom', 'Extra Spicy']

# Özelliklerin X'te var olup olmadığını kontrol etme
missing_numerical_model_features = [col for col in numerical_features_for_model if col not in features_X.columns]
missing_categorical_model_features = [col for col in categorical_features_for_model if col not in features_X.columns]

if missing_numerical_model_features:
    print(f"ERROR (Preprocessing): Numerical feature(s) not found in X: {missing_numerical_model_features}")
    exit()
if missing_categorical_model_features:
    print(f"ERROR (Preprocessing): Categorical feature(s) not found in X: {missing_categorical_model_features}")
    exit()

# ColumnTransformer: Sayısal özelliklere ölçeklendirme, kategoriklere One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical_transformer', StandardScaler(), numerical_features_for_model),
        ('categorical_transformer', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features_for_model)
        # drop='first' kukla değişken tuzağını engellemek için
        # sparse_output=False sonucu yoğun (dense) matris olarak verir, küçük veri setlerinde yönetimi kolaylaştırır
    ],
    remainder='passthrough' # Belirtilmeyen sütunları olduğu gibi bırakır (bu projede olmamalı)
)
print("Data preprocessing steps (ColumnTransformer) defined.")

# -----------------------------------------------------------------------------
# 4. Splitting Data into Training and Test Sets
# -----------------------------------------------------------------------------
print("\n--- 4. Splitting Data into Training and Test Sets ---")
X_train, X_test, y_train, y_test = train_test_split(features_X, target_y, test_size=0.2, random_state=42)
# test_size=0.2: Verinin %20'si test için ayrılır
# random_state=42: Sonuçların her çalıştırmada aynı olması için (tekrarlanabilirlik)

print(f"Training set size: {X_train.shape[0]} samples, Test set size: {X_test.shape[0]} samples")

# -----------------------------------------------------------------------------
# 5. Defining, Training, and Optimizing Models
# -----------------------------------------------------------------------------
print("\n--- 5. Defining, Training, and Optimizing Models ---")

# Modeller
ml_models = { # 'models' is too generic, changed to 'ml_models'
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVR": SVR()
}

# Hiperparametre aralıkları (Bu küçük veri seti için basit tutulmuştur)
parameter_grids = { # 'param_grids' is fine
    "Linear Regression": {}, # Özel bir hiperparametresi yok
    "Decision Tree": {
        'model__max_depth': [3, 5, None], # None: Sınırsız derinlik
        'model__min_samples_leaf': [1, 2, 3] # Bir yaprakta olması gereken min örnek
    },
    "Random Forest": {
        'model__n_estimators': [50, 100], # Ağaç sayısı
        'model__max_depth': [3, 5, None],
        'model__min_samples_leaf': [1, 2]
    },
    "SVR": {
        'model__C': [0.1, 1, 10], # Ceza parametresi
        'model__kernel': ['linear', 'rbf'] # Çekirdek tipi
    }
}

model_results = {} # Sonuçları saklamak için

# Her bir model için pipeline oluşturma, GridSearchCV ile eğitme ve değerlendirme
for model_name, model_instance in ml_models.items():
    print(f"\nProcessing model: {model_name}...")

    # Pipeline: Ön işlemci + Model
    # Bu satır Türkçe yorumlu kalabilir, çünkü pipeline mantığını açıklıyor
    # İşlem hattı: önişlemci (ColumnTransformer) + model (mevcut regresyon modeli)
    processing_pipeline = Pipeline(steps=[('preprocessor', preprocessor), # 'pipeline' is fine
                                          ('model', model_instance)])

    # Hiperparametre optimizasyonu için GridSearchCV
    # Not: Çok küçük veri setlerinde (örn: 20 satır) GridSearchCV ve çapraz doğrulama
    # kararlı sonuçlar vermeyebilir veya hatalara yol açabilir.
    # Eğitim setindeki örnek sayısı (örn: ~16) cv değerinden büyük olmalıdır.
    cross_validation_folds = min(3, X_train.shape[0] // 2 if X_train.shape[0] // 2 >= 2 else 2) # cv değeri için basit bir kontrol

    if model_name in parameter_grids and parameter_grids[model_name]:
        try:
            grid_search_cv = GridSearchCV(processing_pipeline, # 'grid_search' is fine
                                          parameter_grids[model_name],
                                          cv=cross_validation_folds,
                                          scoring='r2', # R-kare skoruna göre optimizasyon
                                          n_jobs=-1, # Tüm işlemcileri kullan
                                          verbose=0) # Çıktıyı azalt
            grid_search_cv.fit(X_train, y_train)
            best_model_estimator = grid_search_cv.best_estimator_ # 'best_model' is fine
            print(f"  Best parameters for {model_name}: {grid_search_cv.best_params_}")
        except ValueError as e: # Eğer cv değeriyle ilgili bir sorun olursa
            print(f"  GridSearchCV ERROR ({model_name}): {e}. Training model with default parameters.")
            processing_pipeline.fit(X_train, y_train)
            best_model_estimator = processing_pipeline
    else:
        processing_pipeline.fit(X_train, y_train) # Optimizasyon yoksa doğrudan eğit
        best_model_estimator = processing_pipeline

    # Test seti üzerinde tahminler
    y_predictions = best_model_estimator.predict(X_test) # 'y_pred' is fine

    # Performans metrikleri
    mse_score = mean_squared_error(y_test, y_predictions) # 'mse' is fine
    rmse_score = np.sqrt(mse_score) # 'rmse' is fine
    mae_score = mean_absolute_error(y_test, y_predictions) # 'mae' is fine
    r2_metric = r2_score(y_test, y_predictions) # 'r2' is fine

    model_results[model_name] = { # 'results' is fine
        'Model': best_model_estimator, # Eğitilmiş en iyi modeli de saklayalım
        'MSE': mse_score,
        'RMSE': rmse_score,
        'MAE': mae_score,
        'R2 Score': r2_metric,
        'Predictions': y_predictions
    }

    print(f"  Metrics for {model_name}:")
    print(f"    MSE      : {mse_score:.4f}")
    print(f"    RMSE     : {rmse_score:.4f}")
    print(f"    MAE      : {mae_score:.4f}")
    print(f"    R2 Score : {r2_metric:.4f}")

# -----------------------------------------------------------------------------
# 6. Comparing Results
# -----------------------------------------------------------------------------
print("\n--- 6. Comparing Results ---")

# DataFrame from 'model_results' dict comprehension
results_dataframe = pd.DataFrame([(model_key, model_value['R2 Score'], model_value['RMSE'], model_value['MAE'], model_value['MSE'])
                                  for model_key, model_value in model_results.items()],
                                 columns=['Model Name', 'R2 Score', 'RMSE', 'MAE', 'MSE'])
results_dataframe = results_dataframe.sort_values(by='R2 Score', ascending=False).reset_index(drop=True) # 'results_df' is fine

print("\nPerformance Summary of All Models (sorted by R2 Score):")
print(results_dataframe)

if not results_dataframe.empty:
    best_model_name_from_summary = results_dataframe.loc[0, 'Model Name'] # 'best_model_name' is fine
    print(f"\nBest performing model (based on R2 Score): {best_model_name_from_summary}")

    # -----------------------------------------------------------------------------
    # 7. Visualizing Predictions of the Best Model
    # -----------------------------------------------------------------------------
    if best_model_name_from_summary in model_results:
        print(f"\n--- 7. Predictions of {best_model_name_from_summary} Model ---")
        best_model_y_predictions = model_results[best_model_name_from_summary]['Predictions'] # 'best_y_pred' is fine

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, best_model_y_predictions, alpha=0.7, edgecolors='k', s=50)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # İdeal çizgi
        plt.xlabel(f'Actual {target_column_name} Values')
        plt.ylabel(f'Predicted {target_column_name} Values')
        plt.title(f'{best_model_name_from_summary}: Actual vs. Predicted')
        plt.show()

        # Artıkların (Hataların) Dağılımı
        prediction_residuals = y_test - best_model_y_predictions # 'residuals' is fine
        plt.figure(figsize=(10, 6))
        sns.histplot(prediction_residuals, kde=True, bins=10)
        plt.xlabel('Residuals (Error = Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.title(f'{best_model_name_from_summary}: Distribution of Residuals')
        plt.axvline(0, color='red', linestyle='--') # Sıfır hata çizgisi
        plt.show()
else:
    print("No model results were generated to summarize or visualize.")

print("\nProject Code Execution Completed.")