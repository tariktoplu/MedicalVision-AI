# compare_mr_results.py

import json
import argparse
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score, precision_score, recall_score
from collections import defaultdict

# --- Sabitler (Sonn.py'den) ---
CLASS_NAMES = ["hyperacute_acute", "subacute", "normal_chronic"]
LABEL_MAP_FROM_JSON = {"HiperakutAkut": "hyperacute_acute", "Subakut": "subacute", "NormalKronik": "normal_chronic"}

def build_ground_truth(gt_data):
    """
    Dilim bazlı gerçek etiketleri, hasta bazlı multi-label etiketlere dönüştürür.
    Bir hastanın herhangi bir diliminde bir bulgu varsa, o hasta o bulguya sahip kabul edilir.
    """
    gt_dict = defaultdict(lambda: {k: 0 for k in CLASS_NAMES})
    
    for entry in gt_data:
        if str(entry.get("Modality", "")).upper() != "MR":
            continue
        
        pid = str(entry.get("PatientId"))
        lesion = entry.get("LessionName")
        
        if not pid or not lesion or lesion not in LABEL_MAP_FROM_JSON:
            continue
            
        # JSON'daki etiket ismini bizim standart anahtar ismimize çevir
        class_name = LABEL_MAP_FROM_JSON[lesion]
        
        # O hasta için o sınıfı 1 olarak işaretle
        gt_dict[pid][class_name] = 1
        
    return dict(gt_dict)


def compare_patient_level_json(ground_truth_path, prediction_path):
    """
    Hasta bazlı multi-label tahminleri, dilim bazlı gerçeklerle karşılaştırır.
    """
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
        
        with open(prediction_path, 'r', encoding='utf-8') as f:
            prediction_data_full = json.load(f)
            # Yarışma formatına uygun olarak 'tahminler' listesini al
            prediction_data = prediction_data_full.get("tahminler", [])

    except FileNotFoundError as e:
        print(f"Hata: Dosya bulunamadı - {e.filename}")
        return
    except json.JSONDecodeError as e:
        print(f"Hata: JSON dosyası bozuk veya hatalı formatta. Hata: {e}")
        return

    # Gerçek etiketleri hasta bazlı hazırla
    gt_dict = build_ground_truth(ground_truth_data)
    
    # Modelin tahminlerini hasta bazlı bir sözlüğe dönüştür
    pred_dict = {str(e["PatientID"]): {k: int(e[k]) for k in CLASS_NAMES} for e in prediction_data}
    
    y_true = [] # Gerçek etiketler (multi-hot formatında)
    y_pred = [] # Tahmin edilen etiketler (multi-hot formatında)
    
    # Sadece her iki dosyada da bulunan ortak hastaları değerlendir
    common_pids = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))

    if not common_pids:
        print("Hata: İki dosyada da eşleşen hiçbir 'PatientID' bulunamadı. Karşılaştırma yapılamıyor.")
        return
        
    print(f"Toplam {len(gt_dict)} gerçek etiketli hasta ve {len(pred_dict)} tahmin edilmiş hasta bulundu.")
    print(f"{len(common_pids)} ortak hasta üzerinden değerlendirme yapılıyor...\n")
    
    for pid in common_pids:
        y_true.append([gt_dict[pid][c] for c in CLASS_NAMES])
        y_pred.append([pred_dict[pid][c] for c in CLASS_NAMES])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print("\n" + "="*60)
    print(" HASTA BAZLI PERFORMANS RAPORU (MULTI-LABEL)")
    print("="*60 + "\n")
    
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, digits=4)
    print(report)
    
    print("\n--- Genel Ortalama Skorlar ---")
    for avg in ["micro", "macro", "weighted", "samples"]:
        prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
        rec  = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1   = f1_score(y_true, y_pred, average=avg, zero_division=0)
        print(f"{avg.capitalize():<10} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
        
    cm = multilabel_confusion_matrix(y_true, y_pred)
    print("\n" + "="*60)
    print("SINIF BAZLI KARMAŞIKLIK MATRİSLERİ")
    print("="*60)
    for i, label in enumerate(CLASS_NAMES):
        print(f"\nSınıf: {label}")
        print("         Tahmin: Negatif   Tahmin: Pozitif")
        print(f"Gerçek: Negatif   {cm[i, 0, 0]:<15} {cm[i, 0, 1]:<15}")
        print(f"Gerçek: Pozitif    {cm[i, 1, 0]:<15} {cm[i, 1, 1]:<15}")
        print(f"(TN: {cm[i, 0, 0]}, FP: {cm[i, 0, 1]}, FN: {cm[i, 1, 0]}, TP: {cm[i, 1, 1]})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hasta bazlı multi-label MR tahminlerini gerçek sonuçlarla karşılaştırır.")
    parser.add_argument("ground_truth_json", help="Gerçek sonuçları içeren (dilim bazlı) JSON dosyasının yolu (örn: MR_Son.json).")
    parser.add_argument("predictions_json", help="Modelin (hasta bazlı) tahminlerini içeren JSON dosyasının yolu.")

    args = parser.parse_args()
    compare_patient_level_json(args.ground_truth_json, args.predictions_json)

    #python compare_mr_results.py MR_Son.json 987654_TUSEB_SYZ_MR_Yarisma.jsonpython