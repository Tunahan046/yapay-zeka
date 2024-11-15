import pandas as pd

# 1. Yorumlar.csv dosyasını yükle
df_comments = pd.read_csv('yorumlar.csv', encoding='utf-8')

# 2. Yorumlar_tahmin.csv dosyasını yükle (sadece 'Predicted_Class' kolonu alınacak)
df_predictions = pd.read_csv('yorumlar_tahmin.csv', encoding='utf-8')

# 3. Yorumlar ve tahmin verilerini birleştir
df_comments['Predicted_Class'] = df_predictions['Predicted_Class']

# 4. Yeni CSV dosyasını kaydet
df_comments.to_csv('yorumlar_ve_tahminler.csv', index=False)

print("Yeni CSV dosyası oluşturuldu: yorumlar_ve_tahminler.csv")
