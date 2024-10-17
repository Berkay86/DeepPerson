import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, log_loss
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 1. Veriyi yükle
data = pd.read_excel('C:/Users/Berkay/Desktop/SONN/Data/englishYorum_star.xlsx')

# 2. Yorum ve Duygu sütunlarını seç
X = data['Yorum']  # Yorumlar
y = data['Duygu']  # Duygu sınıfları

# 3. Etiket eşleştirmesini tanımla
label_map = {
    '1 star': 0,  # Negatif
    '2 stars': 0,  # Negatif
    '3 stars': None,  # 3 yıldızları hariç tut
    '4 stars': 1,  # Pozitif
    '5 stars': 1   # Pozitif
}

# 4. Duygu etiketlerini eşleştirme ile dönüştür
y_mapped = y.map(label_map)

# 5. 3 yıldızlı yorumları hariç tut
data_filtered = data[~y_mapped.isna()]
X_filtered = data_filtered['Yorum']
y_filtered = y_mapped.dropna()

# 6. Metin Ön İşleme
sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def preprocess_texts(texts):
    texts = [str(text) if not pd.isna(text) else '' for text in texts]  # NaN veya dize olmayanları yönet
    texts = [text.lower() for text in texts]  # Büyük/küçük harf normalizasyonu
    texts = [re.sub(r'[^\w\s]', '', text) for text in texts]  # Noktalama işaretlerini kaldır
    texts = [re.sub(r'\d', '', text) for text in texts]  # Sayıları kaldır
    texts = [" ".join(word for word in text.split() if word not in sw) for text in texts]  # Stopwords kaldır
    texts = [" ".join([lemmatizer.lemmatize(word) for word in text.split()]) for text in texts]  # Lemmatizasyon
    return texts

# Eğitim verisine ön işleme uygula
X_filtered = preprocess_texts(X_filtered)
# 7. Kelime dağarcığını oluştur
from tensorflow.keras.preprocessing.text import Tokenizer

max_words = 5000  # Kelime sayısını sınırlayın
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_filtered)
X_sequences = tokenizer.texts_to_sequences(X_filtered)

# 8. Dizileri pad et
max_len = 100  # Maksimum dizinin uzunluğu
X_pad = pad_sequences(X_sequences, maxlen=max_len)

# 9. Etiketleri sayısal hale getir
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_filtered)
# 10. Eğitim-Test Bölmesi
train_x, test_x, train_y, test_y = train_test_split(X_pad, y_encoded, test_size=0.33, random_state=42)
# 11. Modeli oluştur
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# 12. Modeli derle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 13. Modeli eğit
history = model.fit(train_x, train_y, epochs=5, batch_size=64, validation_data=(test_x, test_y), verbose=2)
# 14. Modeli değerlendirin
y_pred = model.predict(test_x)
y_pred_classes = (y_pred > 0.5).astype(int)

print("Sonuçlar:")
print(classification_report(test_y, y_pred_classes))

# Eğitim ve test doğruluğunu hesapla
train_accuracy = accuracy_score(train_y, model.predict(train_x) > 0.5)
test_accuracy = accuracy_score(test_y, y_pred_classes)
test_loss = log_loss(test_y, y_pred)

print(f'Yapay Sinir Ağı - Eğitim Doğruluğu: {train_accuracy:.2f}, Test Doğruluğu: {test_accuracy:.2f}, Test Kaybı: {test_loss:.2f}')
# 15. Modeli kaydet
model.save('sentiment_analysis_model.h5')
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Modeli yükle
model = load_model('sentiment_analysis_model.h5')

# Örnek yorumları tanımla
sample_reviews = [
    "This product is amazing!",
    "I had a terrible experience with this item.",
    "It's okay, not the best but not the worst.",
    "Absolutely loved it! Will buy again.",
    "The product is really nice even though the shipping service is terrible"
]

# Metin ön işleme fonksiyonu
sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def preprocess_texts(texts):
    texts = [str(text) if not pd.isna(text) else '' for text in texts]
    texts = [text.lower() for text in texts]
    texts = [re.sub(r'[^\w\s]', '', text) for text in texts]
    texts = [re.sub(r'\d', '', text) for text in texts]
    texts = [" ".join(word for word in text.split() if word not in sw) for text in texts]
    texts = [" ".join([lemmatizer.lemmatize(word) for word in text.split()]) for text in texts]
    return texts

# Örnek yorumları ön işleme uygula
sample_reviews_processed = preprocess_texts(sample_reviews)

# Kelime dizilerini oluştur
sample_sequences = tokenizer.texts_to_sequences(sample_reviews_processed)

# Dizileri pad et
max_len = 100  # Max length you used during training
sample_pad = pad_sequences(sample_sequences, maxlen=max_len)

# Tahmin yap
sample_predictions = model.predict(sample_pad)

# Tahmin sonuçlarını sınıf etiketlerine dönüştür
sample_pred_classes = (sample_predictions > 0.5).astype(int)

# Sonuçları yazdır
for review, pred in zip(sample_reviews, sample_pred_classes):
    sentiment = "Positive" if pred[0] == 1 else "Negative"
    print(f"Review: '{review}' => Sentiment: {sentiment}")
