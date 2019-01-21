
import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, GRU , Embedding
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model

#PANDAS ARACILIĞIYLA VERİSETİ DATAFRAME OLARAK OKUNDU
dataset = pd.read_csv('hepsiburada.csv')

#PUAN VE YORUM SÜTUNLARI AYRILDI

yorum = dataset['Review'].values.tolist()
puan = dataset['Rating'].values.tolist()



#VERİLERİN %80'İ EĞİTİM İÇİN %20'Sİ TEST İÇİN KULLANILMAK ÜZERE AYRILDI
egitimYuzde = len(yorum) * 0.80
egitimYuzde = int(egitimYuzde)
x_train, x_test = yorum[:egitimYuzde], yorum[egitimYuzde:]
y_train, y_test = puan[:egitimYuzde], puan[egitimYuzde:]


# VERİ SETİNDEKİ 10.000 TANE FARKLI KELİMEYİ TOKENLEŞTİRME İŞLEMİ
# TOKEN İŞLEMİ BU 10000 TANE KELİMEYİ TOPLAYIP ONLARA KARŞILIK GELEN SAYIYI ÜRETİYOR
# BU SAYEDE EĞİTİMDE KULLANMAK ÜZERE STRING İFADELERİ SAYISAL İFADELERE ÇEVİRİYORUZ
# VERİSETİNDEKİ HER KELİMEYE KARŞILIK GELEN BİR SAYI OLMAYABİLİYOR, ÇÜNKÜ 10000 TANE BELİRLEDİK
# EĞER KARŞILIĞI YOKSA 0 OLARAK KABUL EDİLİYOR

unique_num_words = 20000
tokenizer = Tokenizer(num_words=unique_num_words)
tokenizer.fit_on_texts(yorum)
x_train_tokens, x_test_tokens = tokenizer.texts_to_sequences(x_train), tokenizer.texts_to_sequences(x_test)
num_tokens = np.array([len(tokens) for tokens in x_train_tokens + x_test_tokens])



#BÜTÜN YORUMLARDAKİ KELİME SAYISI EŞİT OLMALI, FARKLI MİKTARLARDA KELİME SAYISI OLMAMASI LAZIM
#TOPLAM TOKEN SAYISININ ORTALAMASINI ALIP STANDART SAPMASINI 2 İLE ÇARPIP EKLİYORUZ BU SAYEDE MAKSİMUM TOKEN'İ BULUYORUZ
max_tokens = int(np.mean(num_tokens) + 3.5 * np.std(num_tokens))


#BULUNAN BU MAKSİMUM TOKEN SAYISI HER CÜMLE İÇİN AYNI HALE GETİRİLİYOR
#ÖRNEK OLARAK 50 TOKEN ÇIKTIYSA BÜTÜN CÜMLELER 50 KELİMEYE DÖNÜŞTÜRÜLÜYOR, 50'DEN FAZLA İSE FAZLALIKLAR SİLİNİYOR
#EĞER 50'DEN AZ İSE 0 EKLENİYOR
x_train_pad, x_test_pad = pad_sequences(x_train_tokens, maxlen=max_tokens), pad_sequences(x_test_tokens, maxlen=max_tokens)


model = Sequential()



#Embedding vektörler için kullanılıyor, her kelimeye karşılık 50 vektör denk gelicek

embedding_size = 50
model.add(Embedding(input_dim=unique_num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='embedding_layer'))
#RNN yapısı return_sequences yazınca output olarak tamamı döndürülüyor, çünkü sonraki layerımız GRU ve 1'den fazla nörona sahip
model.add(GRU(units=32, return_sequences=True))
model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train_pad, y_train, epochs=5, batch_size=256, validation_data=(x_test_pad,y_test))

#model = load_model('analysis_v2')


while(True):
 text = input("Metin giriniz...  ")
 tokens = tokenizer.texts_to_sequences([text])
 tokens_pad = pad_sequences(tokens,maxlen=max_tokens)
 print(model.predict(tokens_pad))


#model.save('analysis_v2')
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#plt.savefig('acc.jpg')
#summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#plt.savefig('loss.jpg')

#Teslimat çok hızlı ama ürün güzel değil 0.5
#Ne iyi ne kötü 0.5
#Eğer gündelik kullanım için alıyorsanız almayın, fakat diğer türlü kullanımlar için gayet iyi 0.5
