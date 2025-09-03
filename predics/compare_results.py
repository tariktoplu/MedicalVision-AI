# compare_results.py

import json
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compare_json_files(ground_truth_path, prediction_path):
    """
    İki JSON dosyasını karşılaştırır, metrikleri hesaplar ve confusion matrix çizer.
    
    :param ground_truth_path: Gerçek etiketleri içeren JSON dosyasının yolu (MR_Son.json).
    :param prediction_path: Modelin tahminlerini içeren JSON dosyasının yolu (model.json).
    """
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
        
        with open(prediction_path, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Hata: Dosya bulunamadı - {e.filename}")
        return
    except json.JSONDecodeError:
        print("Hata: JSON dosyalarından biri bozuk veya hatalı formatta.")
        return

    # Eşleştirmeyi hızlandırmak için gerçek etiketleri bir sözlüğe (map) dönüştür
    # Key: ImageId, Value: LessionName
    truth_map = {item['ImageId']: item['LessionName'] for item in ground_truth_data}
    
    y_true = [] # Gerçek etiketler listesi
    y_pred = [] # Tahmin edilen etiketler listesi
    
    unmatched_predictions = 0

    # Tahmin dosyasındaki her bir kaydı gez
    for prediction in prediction_data:
        image_id = prediction.get('ImageId')
        
        if image_id in truth_map:
            # Eğer ImageId gerçek etiketler dosyasında varsa, iki listeye de ekle
            y_true.append(truth_map[image_id])
            y_pred.append(prediction['LessionName'])
        else:
            unmatched_predictions += 1
            
    if not y_true:
        print("Hata: İki dosyada da eşleşen hiçbir 'ImageId' bulunamadı. Karşılaştırma yapılamıyor.")
        return

    if unmatched_predictions > 0:
        print(f"Uyarı: Tahmin dosyasındaki {unmatched_predictions} adet kayıt, gerçek etiketler dosyasında bulunamadığı için değerlendirme dışı bırakıldı.")

    # Tüm olası etiketleri bul ve sırala
    labels = sorted(list(set(y_true)))

    print("\n" + "="*50)
    print(" MODEL PERFORMANS RAPORU")
    print("="*50 + "\n")
    
    # Classification Report (Precision, Recall, F1-Score)
    try:
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Classification report oluşturulurken hata oluştu: {e}")
        return
        
    # Confusion Matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
        plt.title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=16)
        plt.ylabel('Gerçek Etiket (Actual)', fontsize=12)
        plt.xlabel('Tahmin Edilen Etiket (Predicted)', fontsize=12)
        
        # Grafiği kaydet
        output_filename = "comparison_confusion_matrix.png"
        plt.savefig(output_filename)
        
        print("\n" + "="*50)
        print(f"Karmaşıklık matrisi '{output_filename}' olarak kaydedildi.")
        print("="*50)

    except Exception as e:
        print(f"Confusion matrix oluşturulurken hata oluştu: {e}")


if __name__ == '__main__':
    # Komut satırından argümanları almak için bir parser oluştur
    parser = argparse.ArgumentParser(description="Model tahminleri ile gerçek sonuçları karşılaştırır.")
    
    # Gerekli argümanları tanımla
    parser.add_argument("ground_truth_json", help="Gerçek sonuçları içeren JSON dosyasının yolu (örn: MR_Son.json).")
    parser.add_argument("predictions_json", help="Modelin tahminlerini içeren JSON dosyasının yolu (örn: model_tahminleri.json).")

    # Argümanları ayrıştır
    args = parser.parse_args()
    
    # Ana fonksiyonu çağır
    compare_json_files(args.ground_truth_json, args.predictions_json)