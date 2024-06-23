import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def draw_plot():
    # Read data from file
    df = pd.read_csv('epa-sea-level.csv')

    # Create scatter plot
    x = df['Year']
    y = df['CSIRO Adjusted Sea Level']
    fig, ax = plt.subplots(figsize = (8,6))
    ax = plt.scatter(x, y)
    
    # Create first line of best fit
    slope,intercept,_,_,_ = linregress(x,y)
    x_values = pd.Series([i for i in range(1880,2051)])
    y_values = slope*x_values + intercept
    plt.plot(x_values,y_values,'r',label = 'Linear fit (1880-2050)')
    
    # Create second line of best fit
    newDf = df.loc[df['Year'] >= 2000]
    newX= newDf['Year']
    newY = newDf['CSIRO Adjusted Sea Level']
    newSlope,newIntercept,_,_,_ = linregress(newX,newY)
    newxPredict = pd.Series([i for i in range(2000,2051)])
    newyPredict = newSlope*newxPredict + newIntercept
    plt.plot(newxPredict,newyPredict,'g',label = 'Linear fit (2000-2050)')
    
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')
    plt.legend()
    
    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plot.png')
    return plt.gca()