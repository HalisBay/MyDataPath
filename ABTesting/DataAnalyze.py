import pandas as pd
import numpy as np

def analyze_data(file_name):
    df = pd.read_csv(file_name)

    print("Genel bilgiler")
    print(df.describe())

    clickCounts = df['Click'].value_counts()
    print(f"Toplam tıklama sayısı: {clickCounts.get(1, 0)}")

    experienceClickCounts = df[df['Group'] == 'Experience']['Click'].value_counts()
    print(f"\t Experience grubu tıklama sayısı: {experienceClickCounts.get(1, 0)}")

    controlClickCounts = df[df['Group'] == 'Control']['Click'].value_counts()
    print(f"\t Control grubu tıklama sayısı: {controlClickCounts.get(1, 0)}")

    basketCounts = df['Basket'].value_counts()
    print(f"Toplam sepete ekleme sayısı: {basketCounts.get(1, 0)}")

    experienceBasketCounts = df[df['Group'] == 'Experience']['Basket'].value_counts()
    print(f"\t Experience grubu sepete ekleme sayısı: {experienceBasketCounts.get(1, 0)}")

    controlBasketCounts = df[df['Group'] == 'Control']['Basket'].value_counts()
    print(f"\t Control grubu sepete ekleme sayısı: {controlBasketCounts.get(1, 0)}")

    purchaseCounts = df['Purchase'].value_counts()
    print(f"Toplam satın alma sayısı: {purchaseCounts.get(1, 0)}")

    experiencePurchaseCounts = df[df['Group'] == 'Experience']['Purchase'].value_counts()
    print(f"\t Experience grubu satın alma sayısı: {experiencePurchaseCounts.get(1, 0)}")

    controlPurchaseCounts = df[df['Group'] == 'Control']['Purchase'].value_counts()
    print(f"\t Control grubu satın alma sayısı: {controlPurchaseCounts.get(1, 0)}")
