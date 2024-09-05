import ABTest
import CreateData
import DataAnalyze
import ABTestPlt


#CreateData.create_data()

#DataAnalyze.analyze_data('data3.csv')
abTestResults = ABTest.ABTest('data3.csv')

ABTestPlt.GeneralFunctions('data3.csv',abTestResults)

