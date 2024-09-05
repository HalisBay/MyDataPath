import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Veri oluşturma fonksiyonu
def create_data():
    # Kullanıcı sayılarını ekleme
    users = 70000

    # A ve B grupları için kullanıcı sayısını ikiye bölüyoruz
    half_users = users // 2

    # Her iki grup için; tıklama, sepete ekleme ve satın alım oranlarını veriyoruz.
    # Deneyim grubu için; tıklama oranı %65, sepete ekleme oranı %60, satın alım oranı %40.
    click_rate_exp = 0.65
    add_to_basket_rate_exp = 0.60
    purchase_rate_exp = 0.40

    # Kontrol grubu için; tıklama oranı %50, sepete ekleme oranı %45, satın alım oranı %20.
    click_rate_cont = 0.50
    add_to_basket_rate_cont = 0.45
    purchase_rate_cont = 0.20

    # Şimdi her kullanıcı için Id ve grupları oluşturalım
    user_ids = np.arange(1, users + 1)
    groups = ['Experience'] * half_users + ['Control'] * half_users

    # Pandas ile tablo oluşturuyoruz
    df = pd.DataFrame({'Id': user_ids, 'Group': groups})

    # Şimdi tıklama oranlarını hesaplayıp tabloya sütun olarak ekliyoruz
    df['Click'] = df.apply(
        lambda x: np.random.binomial(1, click_rate_exp) 
        if x['Group'] == 'Experience'
        else np.random.binomial(1, click_rate_cont),
        axis=1
    )

    # Ardından sepete ekleme oranlarını hesaplayıp tabloya sütun olarak ekliyoruz
    df['Basket'] = df.apply(
        lambda x: np.random.binomial(1, add_to_basket_rate_exp) 
        if x['Group'] == 'Experience' and x['Click'] == 1
        else (np.random.binomial(1, add_to_basket_rate_cont) 
              if x['Group'] == 'Control' and x['Click'] == 1 
              else 0), 
        axis=1
    )

    # Ardından satın alma oranlarını hesaplayıp tabloya sütun ekliyoruz
    df['Purchase'] = df.apply(
        lambda x: np.random.binomial(1, purchase_rate_exp)
        if x['Group'] == 'Experience' and x['Basket'] == 1
        else (np.random.binomial(1, purchase_rate_cont) 
              if x['Group'] == 'Control' and x['Basket'] == 1 else 0),
        axis=1
    )

    # Zaman aralığını oluşturuyoruz ve zaman aralığı hesabı yapıyoruz
    
    start_time = datetime(2024, 7, 1, 0, 0, 0)
    end_time = datetime(2024, 7, 15, 0, 0, 0)
    total_seconds = (end_time - start_time).total_seconds()
    number_of_users = len(df) / 2
    time_interval = total_seconds / number_of_users

    # Zaman damgalarını oluşturup tabloya ekliyoruz
    
    def generate_random_timestamps(size):
        timestamps = []
        for _ in range(size):
            random_time = start_time + timedelta(seconds=np.random.randint(0, total_seconds))
            timestamps.append(random_time)
        return timestamps

    df['Timestamp'] = generate_random_timestamps(len(df))

    # Son olarak verileri CSV dosyasına yazıyoruz
    try:
        df.to_csv('data2.csv', index=False)
        print("Data saved successfully")
    except Exception as e:
        print(f"Error occurred: {e}")

