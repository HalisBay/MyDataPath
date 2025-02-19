from LoadData import preprocessData
from TraficAnalysis import allAnalysis

def main():
    df = preprocessData()

    allAnalysis(df)

if __name__ == "__main__":
    main()