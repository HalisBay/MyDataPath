import pandas as pd
import numpy as np
from scipy.stats import norm

# Ortalama (mu) ve Standart Sapma (sigma) hesabı
def calculate_mu_sigma(data):
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    return mu, sigma

# Güven aralığı hesabı
def calculate_confidence_interval(rate_diff, standardError, zCrit):
    CI_lower = rate_diff - (zCrit * standardError)
    CI_upper = rate_diff + (zCrit * standardError)
    return CI_lower, CI_upper

def LossAnalysis(experience_df = None,control_df = None ):

    # Tıklayan veya sepete ekleyen ancak  satın almayan kullanıcılar için kayıp analizi
    if experience_df is None or control_df is None:
        raise ValueError("DataFrames for experience and control groups must be provided.")
    
    #Genel metrikler
    experienceClicks = experience_df['Click'].sum()
    experienceBaskets = experience_df['Basket'].sum()
    experiencePurchases = experience_df['Purchase'].sum()
    controlClicks = control_df['Click'].sum()
    controlBaskets = control_df['Basket'].sum()
    controlPurchases = control_df['Purchase'].sum()

    #Deneyim Grubu
    print("\n\nEXPERİENCE GROUP")
    print(f"Total number of users who clicked: {experienceClicks}")
    print(f"Number of users adding to cart: {experienceBaskets}")
    print(f"Total number of users who purchased: {experiencePurchases}")

    #Kullanıcıların kayıp oranlarının hesaplanması
    lostAfterClickExperience = experienceClicks - experiencePurchases
    lostAfterClickAndBeforeBasketExperience = experienceClicks - experienceBaskets
    lostAfterBasketExperience = experienceBaskets - experiencePurchases
    
    lostPercentageAfterClickExperience = (lostAfterClickExperience / experienceClicks ) * 100
    lostPercentageAfterClickAndBeforeBasketExperience = (lostAfterClickAndBeforeBasketExperience / experienceClicks) * 100
    lostPercentageAfterBasketExperience = (lostAfterBasketExperience / experienceBaskets ) * 100

    print(f"Loss of users clicking but not purchasing: {lostAfterClickExperience}, Rate: {lostPercentageAfterClickExperience:.2f}%")
    print(f"Loss of users clicking but not adding to cart: {lostAfterClickAndBeforeBasketExperience}, Rate: {lostPercentageAfterClickAndBeforeBasketExperience:.2f}%")
    print(f"Loss of users adding to cart but not purchasing: {lostAfterBasketExperience}, Rate: {lostPercentageAfterBasketExperience:.2f}%")


    #Kontrol grubu
    print("\n\nCONTROL GROUP")
    print(f"Total number of users who clicked: {controlClicks}")
    print(f"Number of users adding to cart: {controlBaskets}")
    print(f"Total number of users who purchased: {controlPurchases}")

    #Kullanıcıların kayıp oranlarının hesaplanması
    lostAfterClickControl = controlClicks - controlPurchases
    lostAfterClickAndBeforeBasketControl = controlClicks - controlBaskets
    lostAfterBasketControl = controlBaskets - controlPurchases
    
    lostPercentageAfterClickControl = (lostAfterClickControl / controlClicks ) * 100
    lostPercentageAfterClickAndBeforeBasketControl = (lostAfterClickAndBeforeBasketControl / controlClicks) * 100
    lostPercentageAfterBasketControl = (lostAfterBasketControl / controlBaskets ) * 100
    print(f"Loss of users clicking but not purchasing: {lostAfterClickControl}, Rate: {lostPercentageAfterClickControl:.2f}%")
    print(f"Loss of users clicking but not adding to cart: {lostAfterClickAndBeforeBasketControl}, Rate: {lostPercentageAfterClickAndBeforeBasketControl:.2f}%")
    print(f"Loss of users adding to cart but not purchasing: {lostAfterBasketControl}, Rate: {lostPercentageAfterBasketControl:.2f}%")


    return {
        'experience': {
            'lostAfterClick': lostAfterClickExperience,
            'lostAfterBasket': lostAfterBasketExperience,
            'lostAfterClickAndBeforeBasket': lostAfterClickAndBeforeBasketExperience,
            'lostPercentageAfterClick': lostPercentageAfterClickExperience,
            'lostPercentageAfterBasket': lostPercentageAfterBasketExperience,
            'lostPercentageAfterClickAndBeforeBasket': lostPercentageAfterClickAndBeforeBasketExperience
        },
        'control': {
            'lostAfterClick': lostAfterClickControl,
            'lostAfterBasket': lostAfterBasketControl,
            'lostAfterClickAndBeforeBasket': lostAfterClickAndBeforeBasketControl,
            'lostPercentageAfterClick': lostPercentageAfterClickControl,
            'lostPercentageAfterBasket': lostPercentageAfterBasketControl,
            'lostPercentageAfterClickAndBeforeBasket': lostPercentageAfterClickAndBeforeBasketControl
        }
    }

    
def ABTest(file_name):
    
    df = pd.read_csv(file_name)
    
    # Zaman'ın gün bazında dönüşümü
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date

    # Deneyim ve kontrol gruplarını ayırma
    experience_df = df[df['Group'] == 'Experience']
    control_df = df[df['Group'] == 'Control']

    # Günlük bazda analiz
    experience_df_date = experience_df.copy()
    control_df_date = control_df.copy()

    experience_df_date['Date'] = experience_df_date['Timestamp'].dt.date
    control_df_date['Date'] = control_df_date['Timestamp'].dt.date

    # Genel metrikler
    daily_experience = experience_df_date.groupby('Date').agg(
        Clicks=('Click', 'sum'),
        Baskets=('Basket', 'sum'),
        Purchases=('Purchase', 'sum'),
        Users=('Id', 'count')
    )

    daily_control = control_df_date.groupby('Date').agg(
        Clicks=('Click', 'sum'),
        Baskets=('Basket', 'sum'),
        Purchases=('Purchase', 'sum'),
        Users=('Id', 'count')
    )
    experienceCounts = experience_df.shape[0]
    controlCounts = control_df.shape[0]

    #Kontrol ve Deney grupları için tıklama (Click) analizleri
    
    # Kontrol ve deney grubundaki toplam tıklama sayısını hesaplanması
    experienceClickCounts = experience_df['Click'].sum()
    controlClickCounts = control_df['Click'].sum()

    print(df.groupby('Group')['Group'].count())
    print("\n\nExperience Click Counts:", experienceClickCounts)
    print("Control Click Counts:", controlClickCounts)

    # Kontrol ve deney grubundaki toplam kullanıcı sayısını hesaplanması
    experienceCounts = df[df['Group'] == 'Experience'].shape[0]
    controlCounts = df[df['Group'] == 'Control'].shape[0]

    # Tıklama oranlarının hesaplanması
    experienceClickRate = experienceClickCounts / experienceCounts
    controlClickRate = controlClickCounts / controlCounts

    print("Experience Click Rate: {:.2%}".format(experienceClickRate))
    print("Control Click Rate: {:.2%}".format(controlClickRate))

    # Birleşik tahmin ve varyans hesaplanması
    pooledClickRate = (experienceClickCounts + controlClickCounts) / (experienceCounts + controlCounts)
    pooledVariance = pooledClickRate * (1 - pooledClickRate)
    print("Pooled Click Rate: {:.2%}".format(pooledClickRate))
    print("Pooled Variance: {:.10}".format(pooledVariance))
    
    standardError = np.sqrt(pooledVariance * (1 / controlCounts + 1 / experienceCounts))
    print("Standard Error: {:.4f}".format(standardError))
    
    # Test istatistiği hesaplanması (Z testi)
    testStatistic = (experienceClickRate - controlClickRate) / standardError
    print("Test Statistic: {:.4f}".format(testStatistic))

    # Z kritik değeri ve P-değeri hesaplanması
    # Alpha testteki hata payını belirler ve genellikle %5(0.05) olarak ayarlanır.
    alpha = 0.05
    zCrit = norm.ppf(1 - alpha / 2)
    pValueClick = 2 * norm.sf(abs(testStatistic))

    print("P Value: {:.10f}".format(pValueClick))
    
    # Güven aralığı hesaplanması
    CI_lower, CI_upper = calculate_confidence_interval(experienceClickRate - controlClickRate, standardError, zCrit)
    
    print("95% Confidence Interval: ({:.4f}, {:.4f})".format(CI_lower, CI_upper))
    
    # Deneyim ve kontrol gruplarının Tıklama verileri
    clickData = df['Click'].values
    
    # Tıklama için; Ortalama ve  standart sapma hesapları
    muClick, sigmaClick = calculate_mu_sigma(clickData)
    print(f" Click Mean: {muClick:.2f}")
    print(f" Click Sigma: {sigmaClick:.2f}")
    

    #------------------------------------------------------------------------
    # Kontrol ve Deney grupları için sepet analizleri

    # Kontrol ve deney grubundaki toplam sepete ekleme sayısını hesaplanması

    experienceBasketCounts = experience_df['Basket'].sum()
    controlBasketCounts = control_df['Basket'].sum()

    print("\n\nNumber of people adding to cart:",df['Basket'].sum())
    print("Experience Basket Counts:", experienceBasketCounts)
    print("Control Basket Counts:", controlBasketCounts)

    # Sepete ekleme oranlarını hesaplanması
    experienceBasketRate = experienceBasketCounts / experienceCounts
    controlBasketRate = controlBasketCounts / controlCounts
    print("Experience Basket Rate: {:.2%}".format(experienceBasketRate))
    print("Control Basket Rate: {:.2%}".format(controlBasketRate))

    # Birleşik tahmin ve varyans hesaplanması
    pooledBasketRate = (experienceBasketCounts + controlBasketCounts) / (experienceCounts + controlCounts)
    pooledVarianceBasket = pooledBasketRate * (1 - pooledBasketRate)
    print("Pooled Basket Rate: {:.2%}".format(pooledBasketRate))
    print("Pooled Variance: {:.5}".format(pooledVarianceBasket))

    #Standart Hata hesaplanması
    standardErrorBasket = np.sqrt(pooledVarianceBasket * (1 / controlCounts + 1 /experienceCounts))
    print("Standard Error: {:.4f}".format(standardErrorBasket))

    # Test istatistiği(Z testi) hesaplanması
    testStatisticBasket = (experienceBasketRate - controlBasketRate) / standardErrorBasket
    print("Test Statistic: {:.4f}".format(testStatisticBasket))

    # P-Değeri hesaplanması
    pValueBasket = 2 * norm.sf(abs(testStatisticBasket))
    print("P Value: {:.3f}".format(pValueBasket))

    # Güven aralığı hesaplanması
    CI_lowerBasket, CI_upperBasket = calculate_confidence_interval(experienceBasketRate - controlBasketRate, standardErrorBasket, zCrit)

    print("95% Confidence Interval for Basket Rate: ({:.4f}, {:.4f})".format(CI_lowerBasket, CI_upperBasket))
    
    # Deneyim ve kontrol gruplarının sepet verileri
    basketData = df['Basket'].values

    # Sepet için; Ortalama ve  standart sapma hesapları
    muBasket, sigmaBasket = calculate_mu_sigma(basketData)
    print(f" Basket Mean: {muBasket:.2f}")
    print(f" Basket Sigma: {sigmaBasket:.2f}")
    #------------------------------------------------------------------------
    # Kontrol ve Deney grupları için Satın alma analizleri

    # Kontrol ve deney grubundaki toplam Satın alma sayısının hesaplanması
    experiencePurchaseCounts = experience_df['Purchase'].sum()
    controlPurchaseCounts = control_df['Purchase'].sum()

    print("\n\nNumber of people Purchase:",df['Purchase'].sum())
    print("Experience Purchase Counts:", experiencePurchaseCounts)
    print("Control Purchase Counts:", controlPurchaseCounts)

    # Satın alma oranlarının hesaplanması
    experiencePurchaseRate = experiencePurchaseCounts / experienceCounts
    controlPurchaseRate = controlPurchaseCounts / controlCounts
    print("Experience Purchase Rate: {:.2%}".format(experiencePurchaseRate))
    print("Control Purchase Rate: {:.2%}".format(controlPurchaseRate))

    # Birleşik tahmin ve varyans hesaplanması
    pooledPurchaseRate = (experiencePurchaseCounts + controlPurchaseCounts) / (experienceCounts + controlCounts)
    pooledVariancePurchase = pooledPurchaseRate * (1 - pooledPurchaseRate)
    print("Pooled Purchase Rate: {:.2%}".format(pooledPurchaseRate))
    print("Pooled Variance (Purchase): {:.5}".format(pooledVariancePurchase))
    
    #Standart Hata hesaplanması
    standardErrorPurchase = np.sqrt(pooledVariancePurchase * (1 / controlCounts + 1 / experienceCounts))
    print("Standard Error (Purchase): {:.4f}".format(standardErrorPurchase))

    # Satın alma için test istatistiği (Z testi) hesaplanması
    testStatisticPurchase = (experiencePurchaseRate - controlPurchaseRate) / standardErrorPurchase
    print("Purchase Test Statistic: {:.4f}".format(testStatisticPurchase))

    pValuePurchase = 2 * norm.sf(abs(testStatisticPurchase))
    print("Purchase P Value: {:.3f}".format(pValuePurchase))

    # Güven aralığı hesaplanması
    CI_lowerPurchase, CI_upperPurchase = calculate_confidence_interval(experiencePurchaseRate - controlPurchaseRate, standardErrorPurchase, zCrit)

    print("95% Confidence Interval for Purchase Rate: ({:.4f}, {:.4f})".format(CI_lowerPurchase, CI_upperPurchase))
    
    purchaseData = df['Purchase'].values

    # Satın alma için; Ortalama ve  standart sapma hesapları
    muPurchase, sigmaPurchase = calculate_mu_sigma(purchaseData)
    print(f" Purchase Mean: {muPurchase:.2f}")
    print(f" Purchase Sigma: {sigmaPurchase:.2f}")

    print("\nZ Critical Value(General): {:.4f}".format(zCrit))
    #------------------------------------------------------------------------------------------------------------------------
    LossAnalysisResults=LossAnalysis(experience_df,control_df)
    
    return {
        'metrics': {
            'experienceDF' : experience_df,
            'controlDF' : control_df,
            'DailyExperience' : daily_experience,
            'DailyControl' : daily_control,
            'zCrit': zCrit
        },
        'PooledRates':{
            'PooledClickRate': pooledClickRate,
            'PooledBasketRate': pooledBasketRate,
            'PooledPurchaseRate': pooledPurchaseRate
        },
        'Clicks':{
          'pooledVariance':pooledVariance,
            'standartError':standardError,
            'testStatistic':testStatistic,
            'pValue':pValueClick,
            'CI_lower':CI_lower,
            'CI_upper':CI_upper,
            'muClick':muClick,
            'sigmaClick':sigmaClick
            
        },
        'Baskets':{
            'pooledVarianceBasket':pooledVarianceBasket,
            'standartErrorBasket':standardErrorBasket,
            'testStatistic':testStatisticBasket,
            'pValue':pValueBasket,
            'CI_lower':CI_lowerBasket,
            'CI_upper':CI_upperBasket,
            'muBasket':muBasket,
            'sigmaBasket':sigmaBasket
        },
        'Purchases':{
            'pooledVariancePurchase':pooledVariancePurchase,
            'standartErrorPurchase':standardErrorPurchase,
            'testStatistic':testStatisticPurchase,
            'pValue':pValuePurchase,
            'CI_lower':CI_lowerPurchase,
            'CI_upper':CI_upperPurchase,
            'muPurchase':muPurchase,
            'sigmaPurchase':sigmaPurchase
        },
        'LossAnalysis': LossAnalysisResults
    }
    