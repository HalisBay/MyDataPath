import pandas as pd

def filterOutliers(df, lower_threshold=5, upper_threshold=140):
    return df[(df['AVERAGE_SPEED'] > lower_threshold) & (df['AVERAGE_SPEED'] <= upper_threshold)]

def loadData(file_path):
    return pd.read_csv(file_path)

def preprocessData():
    csv_file1 = "Data/traffic_density_202401.csv"
    csv_file2 = "Data/traffic_density_202402.csv"
    csv_file12 = "Data/traffic_density_202312.csv"

    dfDec = loadData(csv_file12)
    dfJan = loadData(csv_file1)
    dfFeb = loadData(csv_file2)

    dfall = pd.concat([dfDec,dfJan,dfFeb])
    dfall['DATE_TIME'] = pd.to_datetime(dfall['DATE_TIME'], errors='coerce')
    dfall.dropna(inplace=True)
    cleanDfAll = filterOutliers(dfall)
    # print("Veri Sayısı:", len(dfall))
    # print(dfall.head())
    # print(dfall.tail())
    # print(dfall.isnull().sum())

    return cleanDfAll

