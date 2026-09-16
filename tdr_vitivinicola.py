"""
TD R: Aplicació de la IA en el sector vitivinícola
Model predictiu amb dades reals històriques del conjunt de dades de qualitat del vi.

Aquest programa:
  - descarrega el dataset públic UCI Wine Quality si no està disponible localment
  - prepara les dades i genera categories de qualitat
  - entrena un model de classificació i un de regressió
  - avalua el rendiment i exporta el model entrenat
  - ofereix exemples de predicció per a nous mostres de vinya
"""

import argparse
import os
import pathlib
import urllib.request

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATA_DIR = pathlib.Path(__file__).resolve().parent
RED_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
RED_FILENAME = DATA_DIR / "winequality-red.csv"
WHITE_FILENAME = DATA_DIR / "winequality-white.csv"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def download_if_missing(url: str, destination: pathlib.Path):
    if destination.exists():
        print(f"  - fitxer existent: {destination.name}")
        return
    print(f"  - descarregant: {url}")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"    descarrega completa: {destination.name}")
    except Exception as exc:
        raise RuntimeError(f"No s'ha pogut descarregar {url}: {exc}")


def load_real_wine_data() -> pd.DataFrame:
    """Carrega i concatena els datasets de vi negre i blanc del repositori UCI."""
    print("Carregant dades reals de qualitat del vi...")
    download_if_missing(RED_WINE_URL, RED_FILENAME)
    download_if_missing(WHITE_WINE_URL, WHITE_FILENAME)

    df_red = pd.read_csv(RED_FILENAME, sep=';')
    df_red['wine_type'] = 'red'
    df_white = pd.read_csv(WHITE_FILENAME, sep=';')
    df_white['wine_type'] = 'white'

    df = pd.concat([df_red, df_white], ignore_index=True)
    print(f"  - total registres: {len(df)}")
    return df


def add_quality_category(df: pd.DataFrame) -> pd.DataFrame:
    """Converteix la qualitat numèrica en categories Low / Medium / High."""
    df = df.copy()
    df['quality_category'] = pd.cut(
        df['quality'], bins=[0, 5, 7, 11], labels=['low', 'medium', 'high']
    )
    df['quality_category'] = df['quality_category'].astype(str)
    return df


def preprocess(df: pd.DataFrame, task: str = 'classification') -> tuple:
    """Preprocessa dades per a modelatge, retorna X, y, encoder i scaler."""
    df = add_quality_category(df)

    features = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]

    X = df[features].copy()
    y = df['quality_category'] if task == 'classification' else df['quality']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    label_encoder = None
    if task == 'classification':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    return X_scaled, y, label_encoder, scaler, features


def get_models(task: str = 'classification') -> dict:
    if task == 'classification':
        return {
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=120, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }
    return {
        'RandomForestRegressor': RandomForestRegressor(n_estimators=200, random_state=42)
    }


def evaluate_classification(model, X_test, y_test, label_encoder: LabelEncoder):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=label_encoder.classes_, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    return acc, report, cm, preds


def evaluate_regression(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, mae, r2, preds


def train_and_evaluate(task: str = 'classification') -> dict:
    df = load_real_wine_data()
    X, y, label_encoder, scaler, feature_names = preprocess(df, task=task)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task == 'classification' else None
    )

    models = get_models(task=task)
    results = {}

    for name, model in models.items():
        print(f"Entrenant model: {name}")
        model.fit(X_train, y_train)

        result = {'model': model}
        if task == 'classification':
            acc, report, cm, preds = evaluate_classification(model, X_test, y_test, label_encoder)
            cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            result.update({
                'accuracy': acc,
                'classification_report': report,
                'confusion_matrix': cm,
                'cv_accuracy_mean': cv.mean(),
                'cv_accuracy_std': cv.std(),
                'predictions': preds,
            })
        else:
            mse, mae, r2, preds = evaluate_regression(model, X_test, y_test)
            cv = cross_val_score(model, X, y, cv=5, scoring='r2')
            result.update({
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'cv_r2_mean': cv.mean(),
                'cv_r2_std': cv.std(),
                'predictions': preds,
            })

        result['feature_names'] = feature_names
        result['scaler'] = scaler
        result['label_encoder'] = label_encoder
        result['X_test'] = X_test
        result['y_test'] = y_test
        results[name] = result

    return results, df


def save_model(model, scaler, label_encoder, task: str, name: str):
    model_path = MODEL_DIR / f"{name}_{task}.joblib"
    joblib.dump({'model': model, 'scaler': scaler, 'label_encoder': label_encoder}, model_path)
    print(f"Model guardat a: {model_path}")
    return model_path


def plot_confusion_matrix(cm, labels, output_path: pathlib.Path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('Verdaders')
    plt.xlabel('Predits')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Gràfic guardat: {output_path}")


def example_prediction(model, scaler, label_encoder, feature_names, task: str):
    sample = {
        'fixed acidity': 7.4,
        'volatile acidity': 0.70,
        'citric acid': 0.00,
        'residual sugar': 1.9,
        'chlorides': 0.076,
        'free sulfur dioxide': 11.0,
        'total sulfur dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4,
    }
    X_new = scaler.transform(np.array([[sample[feat] for feat in feature_names]]))
    prediction = model.predict(X_new)[0]

    if task == 'classification':
        label = label_encoder.inverse_transform([prediction])[0]
        probability = model.predict_proba(X_new)[0] if hasattr(model, 'predict_proba') else None
        print(f"Exemple de predicció (qualitat categoria): {label}")
        if probability is not None:
            scores = {label_encoder.inverse_transform([i])[0]: f"{prob:.1%}" for i, prob in enumerate(probability)}
            print(f"Probabilitats: {scores}")
    else:
        print(f"Exemple de predicció (qualitat numèrica): {prediction:.2f}")


def print_summary(results: dict, task: str):
    print("\nResum dels models entrenats:")
    for name, info in results.items():
        if task == 'classification':
            print(f"  - {name}: accuracy test={info['accuracy']:.4f}, cv_mean={info['cv_accuracy_mean']:.4f}")
        else:
            print(f"  - {name}: R² test={info['r2']:.4f}, RMSE={np.sqrt(info['mse']):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Model predictiu IA per al sector vitivinícola')
    parser.add_argument('--task', choices=['classification', 'regression'], default='classification',
                        help='Tipus de model a entrenar: classificació de qualitat o regressió del valor de qualitat')
    args = parser.parse_args()

    print('Inici del programa de predicció vitivinícola amb dades reals històriques')
    results, df = train_and_evaluate(task=args.task)
    print_summary(results, task=args.task)

    best_model_name = max(results, key=lambda name: results[name]['accuracy'] if args.task == 'classification' else results[name]['r2'])
    best_info = results[best_model_name]

    print(f"\nMillor model: {best_model_name}")
    if args.task == 'classification':
        print(best_info['classification_report'])
        plot_confusion_matrix(best_info['confusion_matrix'], best_info['label_encoder'].classes_, DATA_DIR / f"confusion_{best_model_name}.png")
    else:
        print(f"MSE: {best_info['mse']:.2f}")
        print(f"MAE: {best_info['mae']:.2f}")
        print(f"R²: {best_info['r2']:.4f}")

    save_model(best_info['model'], best_info['scaler'], best_info['label_encoder'], args.task, best_model_name)
    example_prediction(best_info['model'], best_info['scaler'], best_info['label_encoder'], best_info['feature_names'], args.task)

    summary_path = DATA_DIR / 'dataset_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as handle:
        handle.write('Estadístiques del dataset de qualitat del vi\n')
        handle.write(df.describe().to_string())
    print(f"Resum del dataset exportat a: {summary_path}")


if __name__ == '__main__':
    main()
