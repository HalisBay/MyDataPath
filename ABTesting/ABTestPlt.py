from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def SaveToJPg(outputDirectory, jpgName):
    # Eğer mevcut değilse, çıktı dizinini oluşturuyoruz
    outputDirect = outputDirectory
    os.makedirs(outputDirect, exist_ok=True)

    # Çıktı dosya yolunu oluşturuyoruz
    outputFilePath = os.path.join(outputDirect, jpgName)

    # Grafiği belirtilen dosya yoluna kaydediyoruz
    plt.savefig(outputFilePath, format='jpg')
    plt.close()

def plot_pdf(mu, sigma, test_statistic, z_crit, p_value, ci_lower, ci_upper, name='default'):
    # Normal dağılımın PDF'ini oluşturmak için x değerlerini belirliyoruz
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
    y = norm.pdf(x, mu, sigma)

    # Grafik için boyutları verip, bir figür oluşturuyoruz
    plt.figure(figsize=(12, 8))

    # Normal dağılımın PDF'ini çiziyoruz
    plt.plot(x, y, label='PDF of Normal Distribution')

    # Kayıp bölgeyi kırmızı ile işaretliyoruz
    plt.fill_between(x, y, where=(x > z_crit) | (x < -z_crit), color='red', alpha=0.5, label='Rejection Region')

    # Test istatistiğini yeşil kesikli çizgi ile gösteriyoruz
    plt.axvline(test_statistic, color='green', linestyle='dashed', linewidth=2, label=f'Test Statistic = {test_statistic:.2f}')

    # Z-kritik değerini mavi kesikli çizgi ile gösteriyoruz
    plt.axvline(z_crit, color='blue', linestyle='dashed', linewidth=1, label=f'Z-critical = {z_crit:.2f}')
    plt.axvline(-z_crit, color='blue', linestyle='dashed', linewidth=1)

    # Güven aralığını mor renkte işaretliyoruz
    plt.fill_between(x, 0, y, where=(x >= ci_lower) & (x <= ci_upper), color='purple', alpha=0.5, label='95% Confidence Interval')

    # Grafik başlığını ve etiketleri ayarlıyoruz
    plt.xlabel('Z-value')
    plt.ylabel('Probability Density')
    plt.title(f'Gaussian Distribution - PDF, for: {name}\nP-value = {p_value:.4f}')
    plt.legend()
    plt.grid(False)

    # Grafiği kaydediyoruz
    SaveToJPg('graphs/PDFandCDF', f'{name}PDF.jpg')

def plot_cdf(mu, sigma, test_statistic, z_crit, p_value, ci_lower, ci_upper, name='default'):
    # Normal dağılımın CDF'ini oluşturmak için x değerlerini belirliyoruz
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
    y = norm.cdf(x, mu, sigma)

    # Grafik için boyutları verip, bir figür oluşturuyoruz
    plt.figure(figsize=(12, 8))

    # Normal dağılımın CDF'ini çiziyoruz
    plt.plot(x, y, label='CDF of Normal Distribution')

    # Test istatistiğini yeşil kesikli çizgi ile gösteriyoruz
    plt.axvline(test_statistic, color='green', linestyle='dashed', linewidth=2, label=f'Test Statistic = {test_statistic:.2f}')

    # Z-kritik değerini mavi kesikli çizgi ile gösteriyoruz
    plt.axvline(z_crit, color='blue', linestyle='dashed', linewidth=1, label=f'Z-critical = {z_crit:.2f}')
    plt.axvline(-z_crit, color='blue', linestyle='dashed', linewidth=1)

    # Güven aralığını mor renkte işaretliyoruz
    plt.fill_between(x, 0, y, where=(x >= ci_lower) & (x <= ci_upper), color='purple', alpha=0.5, label='95% Confidence Interval')

    # Grafik başlığını ve etiketleri ayarlıyoruz
    plt.xlabel('Z-value')
    plt.ylabel('Cumulative Probability')
    plt.title(f'Gaussian Distribution - CDF, for: {name}\nP-value = {p_value:.4f}')
    plt.legend()
    plt.grid(False)

    # Grafiği kaydediyoruz
    SaveToJPg('graphs/PDFandCDF', f'{name}CDF.jpg')

def PDF_and_CDF_for_click(abTestResults, z_crit):
    # Tıklama metrikleri için PDF ve CDF grafiklerini oluşturuyoruz
    mu = abTestResults['Clicks']['muClick']
    sigma = abTestResults['Clicks']['sigmaClick']
    test_statistic = abTestResults['Clicks']['testStatistic']
    p_value = abTestResults['Clicks']['pValue']
    ci_lower = abTestResults['Clicks']['CI_lower']
    ci_upper = abTestResults['Clicks']['CI_upper']

    plot_pdf(mu, sigma, test_statistic, z_crit, p_value, ci_lower, ci_upper, "Click")
    plot_cdf(mu, sigma, test_statistic, z_crit, p_value, ci_lower, ci_upper, "Click")

def PDF_and_CDF_for_basket(abTestResults, z_crit):
    # Sepet metrikleri için PDF ve CDF grafiklerini oluşturuyoruz
    muB = abTestResults['Baskets']['muBasket']
    sigmaB = abTestResults['Baskets']['sigmaBasket']
    test_statisticB = abTestResults['Baskets']['testStatistic']
    p_valueB = abTestResults['Baskets']['pValue']
    ci_lowerB = abTestResults['Baskets']['CI_lower']
    ci_upperB = abTestResults['Baskets']['CI_upper']

    plot_pdf(muB, sigmaB, test_statisticB, z_crit, p_valueB, ci_lowerB, ci_upperB, "Basket")
    plot_cdf(muB, sigmaB, test_statisticB, z_crit, p_valueB, ci_lowerB, ci_upperB, "Basket")

def PDF_and_CDF_for_purchase(abTestResults, z_crit):
    # Satın alma metrikleri için PDF ve CDF grafiklerini oluşturuyoruz
    muP = abTestResults['Purchases']['muPurchase']
    sigmaP = abTestResults['Purchases']['sigmaPurchase']
    test_statisticP = abTestResults['Purchases']['testStatistic']
    p_valueP = abTestResults['Purchases']['pValue']
    ci_lowerP = abTestResults['Purchases']['CI_lower']
    ci_upperP = abTestResults['Purchases']['CI_upper']

    plot_pdf(muP, sigmaP, test_statisticP, z_crit, p_valueP, ci_lowerP, ci_upperP, "Purchase")
    plot_cdf(muP, sigmaP, test_statisticP, z_crit, p_valueP, ci_lowerP, ci_upperP, "Purchase")

def GraphsOfPDFAndCDF(abTestResults):
    # Z-kritik değerini alıyoruz ve tüm PDF & CDF grafiklerini oluşturuyoruz
    z_crit = abTestResults['metrics']['zCrit']
    PDF_and_CDF_for_click(abTestResults, z_crit)
    PDF_and_CDF_for_basket(abTestResults, z_crit)
    PDF_and_CDF_for_purchase(abTestResults, z_crit)

def plot_combined_loss_analysis(experience_data, control_data):
    # Kayıp metriklerini belirliyoruz
    groups = ['Experience', 'Control']
    lost_after_click = [experience_data['lostAfterClick'], control_data['lostAfterClick']]
    lost_after_basket = [experience_data['lostAfterBasket'], control_data['lostAfterBasket']]
    lost_after_click_and_before_basket = [experience_data['lostAfterClickAndBeforeBasket'], control_data['lostAfterClickAndBeforeBasket']]
    lost_percentage_after_click = [experience_data['lostPercentageAfterClick'], control_data['lostPercentageAfterClick']]
    lost_percentage_after_click_and_before_basket = [experience_data['lostPercentageAfterClickAndBeforeBasket'], control_data['lostPercentageAfterClickAndBeforeBasket']]
    lost_percentage_after_basket = [experience_data['lostPercentageAfterBasket'], control_data['lostPercentageAfterBasket']]

    # Bir figür ve iki eksen oluşturuyoruz
    fig, ax1 = plt.subplots(figsize=(15, 8))
    bar_width = 0.25
    index = np.arange(len(groups))

    # Kayıp sayıları için bar grafik oluşturuyoruz
    ax1.set_xlabel('Groups')
    ax1.set_ylabel('Number of Users', color='tab:red')
    ax1.bar(index - bar_width, lost_after_click, bar_width, color='tab:blue', label='Lost after Click')
    ax1.bar(index, lost_after_click_and_before_basket, bar_width, color='tab:green', label='Lost after Click & Before Basket')
    ax1.bar(index + bar_width, lost_after_basket, bar_width, color='tab:orange', label='Lost after Basket')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xticks(index)
    ax1.set_xticklabels(groups)
    ax1.set_ylim(0, max(max(lost_after_click), max(lost_after_basket), max(lost_after_click_and_before_basket)) * 1.2)
    ax1.legend(loc='upper left')
    ax1.grid(False)

    # Kayıp yüzdeleri için ikinci eksen oluşturuyoruz
    ax2 = ax1.twinx()
    color = 'purple'
    ax2.set_ylabel('Percentage (%)', color=color)
    ax2.plot(groups, lost_percentage_after_click, color=color, marker='o', linestyle='-.', linewidth=2, label='Lost Percentage After Click')
    ax2.plot(groups, lost_percentage_after_click_and_before_basket, color='black', marker='o', linestyle='-.', linewidth=2, label='Lost Percentage After Click & Before Basket')
    ax2.plot(groups, lost_percentage_after_basket, color='tab:red', marker='o', linestyle='-.', linewidth=2, label='Lost Percentage After Basket')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(max(lost_percentage_after_click), max(lost_percentage_after_click_and_before_basket), max(lost_percentage_after_basket)) * 1.2)
    ax2.legend(loc='upper right')
    ax2.grid(False)

    # Grafik düzenini sıkıştırıyoruz ve başlık ekliyoruz
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.title('Combined Loss Analysis with Side-by-Side Bars and Percentages')

    # Grafiği kaydediyoruz
    SaveToJPg('graphs', 'combined_loss_analysis.jpg')

def NumberOfDailyUser(daily_experience, daily_control):
    # Günlük kullanıcı sayılarını grafikleştiriyoruz
    plt.figure(figsize=(15, 8))

    # Günlük tıklama sayılarını grafikleştiriyoruz
    plt.subplot(2, 2, 1)
    plt.plot(daily_experience.index, daily_experience['Clicks'], marker='o', linestyle='-', label='Deneyim')
    plt.plot(daily_control.index, daily_control['Clicks'], marker='o', linestyle='--', label='Kontrol')
    plt.ylim(bottom=0)
    plt.title('Number Of Daily Click')
    plt.xlabel('Date', color='gray')
    plt.ylabel('Click Rate', color='gray')
    plt.legend()
    plt.grid(True)

    # Günlük sepete ekleme sayılarını grafikleştiriyoruz
    plt.subplot(2, 2, 2)
    plt.plot(daily_experience.index, daily_experience['Baskets'], marker='o', linestyle='-', label='Deneyim')
    plt.plot(daily_control.index, daily_control['Baskets'], marker='o', linestyle='--', label='Kontrol')
    plt.ylim(bottom=0)
    plt.title('Number Of Daily Add To Carts')
    plt.xlabel('Date', color='gray')
    plt.ylabel('Click Rate', color='gray')
    plt.legend()
    plt.grid(True)

    # Günlük satın alma sayılarını grafikleştiriyoruz
    plt.subplot(2, 2, 3)
    plt.plot(daily_experience.index, daily_experience['Purchases'], marker='o', linestyle='-', label='Deneyim')
    plt.plot(daily_control.index, daily_control['Purchases'], marker='o', linestyle='--', label='Kontrol')
    plt.ylim(bottom=0)
    plt.title('Number Of Daily Purchases')
    plt.xlabel('Date', color='gray')
    plt.ylabel('Click Rate', color='gray')
    plt.legend()
    plt.grid(True)

    # Grafik düzenini sıkıştırıyoruz
    plt.tight_layout()

    # Grafiği kaydediyoruz
    SaveToJPg('graphs', 'DailyNumbersCBP.jpg')

def TimeIntervalAnalysis(daily_experience, daily_control, pooled_rates):
    # Günlük tıklama oranlarını hesaplıyoruz
    daily_experience['ClickRate'] = daily_experience['Clicks'] / daily_experience['Users']
    daily_control['ClickRate'] = daily_control['Clicks'] / daily_control['Users']

    # Günlük sepete ekleme oranlarını hesaplıyoruz
    daily_experience['BasketRate'] = daily_experience['Baskets'] / daily_experience['Users']
    daily_control['BasketRate'] = daily_control['Baskets'] / daily_control['Users']

    # Günlük satın alma oranlarını hesaplıyoruz
    daily_experience['PurchaseRate'] = daily_experience['Purchases'] / daily_experience['Users']
    daily_control['PurchaseRate'] = daily_control['Purchases'] / daily_control['Users']

    # Grafiğin arka planını ayarlıyoruz
    sns.set_theme(style="whitegrid")

    # Tıklama oranlarını için grafik oluşturuyoruz
    plt.figure(figsize=(15, 8))
    plt.subplot(3, 1, 1)
    sns.lineplot(data=daily_experience, x=daily_experience.index, y='ClickRate', label='Experience', color='blue')
    sns.lineplot(data=daily_control, x=daily_control.index, y='ClickRate', label='Control', color='red')
    plt.axhline(y=pooled_rates['PooledClickRate'], color='green', linestyle='--', label='Pooled Click Rate')
    plt.title('Daily Click Rate')
    plt.xlabel('Date', color='gray')
    plt.ylabel('Click Rate', color='gray')
    plt.legend()

    # Sepete ekleme oranlarını için grafik oluşturuyoruz
    plt.subplot(3, 1, 2)
    sns.lineplot(data=daily_experience, x=daily_experience.index, y='BasketRate', label='Experience', color='blue')
    sns.lineplot(data=daily_control, x=daily_control.index, y='BasketRate', label='Control', color='red')
    plt.axhline(y=pooled_rates['PooledBasketRate'], color='green', linestyle='--', label='Pooled Basket Rate')
    plt.title('Daily Basket Rate')
    plt.xlabel('Date', color='gray')
    plt.ylabel('Basket Rate', color='gray')
    plt.legend()

    # Satın alma oranlarını için grafik oluşturuyoruz
    plt.subplot(3, 1, 3)
    sns.lineplot(data=daily_experience, x=daily_experience.index, y='PurchaseRate', label='Experience', color='blue')
    sns.lineplot(data=daily_control, x=daily_control.index, y='PurchaseRate', label='Control', color='red')
    plt.axhline(y=pooled_rates['PooledPurchaseRate'], color='green', linestyle='--', label='Pooled Purchase Rate')
    plt.title('Daily Purchase Rate')
    plt.xlabel('Date', color='gray')
    plt.ylabel('Purchase Rate', color='gray')
    plt.legend()

    # Grafik düzenini sıkıştırıyoruz
    plt.tight_layout()

    # Grafiği kaydediyoruz
    SaveToJPg('graphs', 'DailyRatesCBP.jpg')

def GeneralFunctions(file_name, abTestResults):
    # Analiz fonksiyonlarını çağırarak gerekli grafikleri oluşturuyoruz
    TimeIntervalAnalysis(abTestResults['metrics']['DailyExperience'], abTestResults['metrics']['DailyControl'], abTestResults['PooledRates'])
    NumberOfDailyUser(abTestResults['metrics']['DailyExperience'], abTestResults['metrics']['DailyControl'])
    plot_combined_loss_analysis(abTestResults['LossAnalysis']['experience'], abTestResults['LossAnalysis']['control'])
    GraphsOfPDFAndCDF(abTestResults)
