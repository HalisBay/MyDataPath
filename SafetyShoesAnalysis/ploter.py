import pandas as pd
import matplotlib.pyplot as plt
import calendar

def draw_bar_plot(csv_file):

    df = pd.read_csv(csv_file)
    df['month'] = pd.to_datetime(df['month'])
    df['year'] = df['month'].dt.year

    numeric_cols = ['iş ayakkabısı', 'safety shoes', 'work boots']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')


    df.dropna(inplace=True)
    df_yearly = df.groupby('year')[numeric_cols].sum()
    fig, ax = plt.subplots(figsize=(12, 8))
    df_yearly.plot(kind='bar', ax=ax)

    ax.set_xlabel('Yıl')
    ax.set_ylabel('Toplam değer (%)')
    ax.set_title('Google üzerinde (Global) yıllık trend değerleri')
    ax.legend(title='Categories', labels=numeric_cols)

    note = "Not: Tablodaki veriler 2024 yılının Haziran ayına kadar olan dönemi kapsamaktadır."
    fig.text(0.5, 0.95, note, ha='center', va='center', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
    fig.savefig('bar_plot_yearly.png')
    plt.show()
