import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import calendar
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Import data (Make sure to parse dates. Consider setting index column to 'date'.)
df = pd.read_csv('fcc-forum-pageviews.csv', index_col='date', parse_dates=True)
# Clean data
bottomLimit = df['value'].quantile(0.025)
topLimit = df['value'].quantile(0.975)
df = df[(df['value'] >= bottomLimit) & (df['value'] <= topLimit)]


def draw_line_plot():
    # Draw line plot
    fig , ax = plt.subplots(figsize=(20,6))

    ax.plot(df.index, df['value'], color='red')
    ax.set_title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019')
    ax.set_xlabel('Date')
    ax.set_ylabel('Page Views')

    

    # Save image and return fig (don't change this part)
    fig.savefig('line_plot.png')
    return fig

def draw_bar_plot():
    # Copy and modify data for monthly bar plot
    df_bar = df.groupby([df.index.year, df.index.month]).mean()
    df_bar = df_bar.unstack()

    # Draw bar plot
    fig,ax = plt.subplots(figsize=(10,6))
    df_bar.plot(kind='bar', ax=ax)
    ax.set_xlabel('Years')
    ax.set_ylabel('Average Page Views')
    ax.legend(title='Months', labels=[calendar.month_name[i] for i in range(1, 13)])



    # Save image and return fig (don't change this part)
    fig.savefig('bar_plot.png')
    return fig

def draw_box_plot():
    # Prepare data for box plots (this part is done!)
    df_box = df.copy()
    df_box.reset_index(inplace=True)
    df_box['year'] = [d.year for d in df_box.date]
    df_box['month'] = [d.strftime('%b') for d in df_box.date]

    # Draw box plots (using Seaborn)
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    sns.boxplot(x='year', y='value', data=df_box, ax=axs[0], palette='Set1', hue='year', legend=False)
    axs[0].set_title('Year-wise Box Plot (Trend)')
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Page Views')

    sns.boxplot(x='month', y='value', data=df_box, ax=axs[1], order=[calendar.month_abbr[i] for i in range(1, 13)], palette='Set3', hue='month', legend=False)
    axs[1].set_title('Month-wise Box Plot (Seasonality)')
    axs[1].set_xlabel('Month')
    axs[1].set_ylabel('Page Views')


    
    # Save image and return fig (don't change this part)
    fig.savefig('box_plot.png')
    return fig
