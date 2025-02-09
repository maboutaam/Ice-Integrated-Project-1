# # Project Name: Global Platfoms and Video Games Data Analysis
# ## The aim of this project is to analyze all the gaming platforms and video games accross all the regions in order to conclude the populars ones and the reasons behind this conclusion.

# In[1]:


# Importing Libarires

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import numpy as np
import datetime


# In[2]:


# Data Loading

df = pd.read_csv("/datasets/games.csv")


# ### The dataframe is for the Excel document of Videogames of gaming platforms and their global information.

# In[3]:


df.info()


# ### There are 16715 entries and 11 columns in this dataframe.

# In[4]:


df.dtypes


# ### There are two data types: Float, Category and Object

# In[5]:


df.describe()

# ### The above information describe the data of each column in the Games dataframe.

# In[6]:


# Prepare Data

# Replace the column names (make them lowercase)

df.columns = df.columns.str.lower()


# ### All columns' names are now lower cased.

# In[7]:


# Prepare Data

# Convert missing values to NaN
df = df.replace("", np.nan)


# ### Missing values are converted to NaN to make it more convenient to analyze the data.

# In[8]:


# Prepare Data

# Convert 'year_of_release' to integer
df["year_of_release"] = pd.to_numeric(df["year_of_release"], errors='coerce').astype('Int64')

# Convert sales columns to float
sales_columns = ["na_sales", "eu_sales", "jp_sales", "other_sales"]
df[sales_columns] = df[sales_columns].apply(pd.to_numeric, errors='coerce')

# Convert 'user_score' to float
df["user_score"] = pd.to_numeric(df["user_score"], errors='coerce')

# Convert 'critic_score' to integer
df["critic_score"] = pd.to_numeric(df["critic_score"], errors='coerce').astype('Int64')

# Convert 'platform', 'genre', 'publisher', 'developer', 'rating' to categorical
categorical_columns = ["platform", "genre", "critic_score", "user_score"]
df[categorical_columns] = df[categorical_columns].astype('category')

# Convert 'name' to string
df["name"] = df["name"].astype(str)


# <div class="alert alert-block alert-success"> <b>Reviewer's comment</b> <a 
# class="tocSkip"></a>
# Perfect primary data preprocessing</div>

# ### Data is being prepared for cleaning for better analysis and data visualization.



# In[9]:


# Prepare Data 

# Filling missing values in the 'rating' column with 'unknown'

df['rating'] = df['rating'].fillna('unknown')


# In[10]:


# Prepare Data

# Drop duplicate rows

df = df.drop_duplicates()


# ### Dataframe is cleaned from duplicates.

# In[11]:


# Prepare Data

# Calculate the total sales (the sum of sales in all regions) for each game and put these values in a separate column.

# New Total Sales column

sales_columns = ["na_sales", "eu_sales", "jp_sales", "other_sales"]

# Calculate the total sales for each game and add it as a new column
df["total_sales"] = df[sales_columns].sum(axis=1)


# In[12]:


# Prepare Data

df.head()


# In[13]:


# Analyze Data

# Look at how many games were released in different years. Is the data for every period significant?

# Group by year and count the number of games released in each year
year_counts = df.groupby("year_of_release")["name"].count()

print("Number of games released in each year:")
print(year_counts)

# Data visualization
plt.figure(figsize=(10, 6))
plt.bar(year_counts.index, year_counts.values)
plt.xlabel("Year of Release")
plt.ylabel("Number of Games")
plt.title("Number of Games Released Each Year")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ### Ever since the year 1980 there are releases of new games every year. This number would obviously increase each year because of the evolution in technology. For example, in the year 1995 less than 200 games were released, this number increased to more than 1400 new released game in the year 2008 but then decreased afterwords to 500 games in 2015.


# In[14]:


# Analyze Data

# Check if the data for every period is significant
min_games_per_year = 350
significant_years = year_counts[year_counts >= min_games_per_year]

print("\nYears with significant data (at least", min_games_per_year, "games released):")
print(significant_years)
significant_years.sum()


# ### We have set the threshold at 350 for the minimum number of games per year. It shows that from the year 1998 to 2016 there are at least 350 games released every year except in 1999.

# In[15]:


# Analyze Data

# Look at how sales varied from platform to platform. Choose the platforms with the greatest total sales and build a distribution based on data for each year. Find platforms that used to be popular but now have zero sales. How long does it generally take for new platforms to appear and old ones to fade?

# Calculate total sales for each platform
platform_sales = df.groupby("platform")["total_sales"].sum().sort_values(ascending=False)
print("Total sales by platform:")
print(platform_sales.head(5))

# Top 5 platforms according to Sales

top_platforms = platform_sales.head(5).index


# In[16]:


# Analyze Data

# Sales distribution for each year for the top platforms
for platform in top_platforms:
    platform_data = df[df["platform"] == platform]
    platform_year_sales = platform_data.groupby("year_of_release")["total_sales"].sum()
    print(f"\nSales distribution for {platform}:")
    print(platform_year_sales)
    
# Data Visualization
plt.figure(figsize=(10, 6))
plt.plot(platform_year_sales.index, platform_year_sales.values, marker='o')
plt.xlabel("Year of Release")
plt.ylabel("Total Sales")
plt.title(f"Sales Distribution for {platform}")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ### The DS was not as popular when it was first released as it became recently. In 1990 the sales were almost 0 but then increased dramatically to more than 140 in the year 2008. This number would proceed to decrease until it reach 0 in the year 2011.

# In[17]:


# Filtering
ps2_data = df[df["platform"] == "PS2"]

# Grouping
ps2_year_sales = ps2_data.groupby("year_of_release")["total_sales"].sum()

# Data Visualization
plt.figure(figsize=(10, 6))
plt.plot(ps2_year_sales.index, ps2_year_sales.values, marker='o', color='blue')
plt.xlabel("Year of Release")
plt.ylabel("Total Sales")
plt.title("Sales Distribution for PS2")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ### PS2 have always been a popular platform in the market. In the year 2000 the total sales was around 45 and then increased to reached it's peak in 2004 by more than 200. This number decreased to reach 0 in the year 2012. This is most probably because of the release of the new PS3.

# In[18]:


# Filtering
x360_data = df[df["platform"] == "X360"]

# Grouping
x360_year_sales = x360_data.groupby("year_of_release")["total_sales"].sum()

print("\nSales distribution for X360:")
print(x360_year_sales)

# Data Visuzaliton
plt.figure(figsize=(10, 6))
plt.plot(x360_year_sales.index, x360_year_sales.values, marker='o', color='green')
plt.xlabel("Year of Release")
plt.ylabel("Total Sales")
plt.title("Sales Distribution for X360")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ### The sales distribution for X360 was around 5 prior to 2006, this number increased to reach it's peak in 2010 by around 173 and then decreased to reach almost 0 in the year 2016. This is probably because of the new release of other gaming platform by the manufacturer of X360.

# In[19]:


# Filtering
ps3_data = df[df["platform"] == "PS3"]

# Grouping
ps3_year_sales = ps3_data.groupby("year_of_release")["total_sales"].sum()

print("\nSales distribution for PS3:")
print(ps3_year_sales)

# Data Visualization
plt.figure(figsize=(10, 6))
plt.plot(ps3_year_sales.index, ps3_year_sales.values, marker='o', color='red')
plt.xlabel("Year of Release")
plt.ylabel("Total Sales")
plt.title("Sales Distribution for PS3")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ### The number of PS3 sales when it was first released in 2006 was 20 and then increased sharply to around 155 in 2011. The total sales then decreased to around 5 in the year 2016. The reason behind it could the new release of the PS4.

# In[20]:


# Filtering
wii_data = df[df["platform"] == "Wii"]

# Grouping
wii_year_sales = wii_data.groupby("year_of_release")["total_sales"].sum()

print("\nSales distribution for Wii:")
print(wii_year_sales)

# Data Visualization
plt.figure(figsize=(10, 6))
plt.plot(wii_year_sales.index, wii_year_sales.values, marker='o', color='purple')
plt.xlabel("Year of Release")
plt.ylabel("Total Sales")
plt.title("Sales Distribution for Wii")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ### The sales distribution for Wii was around 130 in 2006, this number increased to 210 in the year 2009 and then decreased dramatically to 0 in 2016. This means that demand on Wii decreased because of the customer behaviour change for purchasing Wii.

# <div class="alert alert-block alert-danger">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
#     
# ~~All calculations are correct here But you need to visulize data as well~~</div>

# <div class="alert alert-block alert-success"> <b>Reviewer's comment v2</b> <a 
# class="tocSkip"></a>
# Well done</div>

# In[21]:


# Identify platforms with zero sales in the latest year

# The latest year in the dataset
latest_year = df['year_of_release'].max()

# Filter the data for the latest year
latest_year_data = df[df['year_of_release'] == latest_year]

# Group the data by platform and sum the sales for the latest year
platform_sales = latest_year_data.groupby('platform')['total_sales'].sum().reset_index()

# Platforms with zero sales
zero_sales_platforms = platform_sales[platform_sales['total_sales'] == 0]['platform']

# Print the platforms with zero sales
print(f"Platforms with zero sales in {latest_year}:")
print(zero_sales_platforms.values)


# ### There is a total of 22 platforms with zero sales in 2016 such as 2600, 3DO, DC, DS, GB and others.

# <div class="alert alert-block alert-success"> <b>Reviewer's comment</b> <a 
# class="tocSkip"></a>
# Super</div>

# In[22]:


# Analyze Data

# Analyze the time it takes for new platforms to appear and old ones to fade

# Identifying unique platforms and their first and last appearance years
platform_years = df.groupby('platform')['year_of_release'].agg(['min', 'max']).reset_index()
platform_years.columns = ['platform', 'first_year', 'last_year']

# Calculating the time difference between first and last appearance years
platform_years['lifespan'] = platform_years['last_year'] - platform_years['first_year'] + 1

# Print the lifespan for each platform
print(platform_years[['platform', 'lifespan']])

# Analyzing the distribution of platform lifespans
print(f"\nSummary statistics for platform lifespans:")
print(platform_years['lifespan'].describe())


# <div class="alert alert-block alert-success"> <b>Reviewer's comment</b> <a 
# class="tocSkip"></a>
# Well done</div>

# In[23]:


# Analyze Data

# Determine what period you should take data for. To do so, look at your answers to the previous questions. The data should allow you to build a model for 2017.

# Which platforms are leading in sales? Which ones are growing or shrinking? Select several potentially profitable platforms.

# Filtering the data for the years 2014 to 2016
df_2014_2016 = df[(df["year_of_release"] >= 2014) & (df["year_of_release"] <= 2016)]

# Grouping
platform_sales = df_2014_2016.groupby("platform")["total_sales"].sum()

# Sorting
platform_sales_sorted = platform_sales.sort_values(ascending=False)

print("Platform sales for 2014-2016:")
print(platform_sales_sorted)

# Top 5 platforms by sales
top_platforms = platform_sales_sorted.head(5)

print("\nTop 5 platforms by sales for 2014-2016:")
print(top_platforms)

# Data Visualization
plt.figure(figsize=(10, 6))
plt.bar(top_platforms.index, top_platforms.values)
plt.xlabel("Platform")
plt.ylabel("Total Sales")
plt.title("Top 5 Platforms by Sales (2014-2016)")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ### The above bar chart shows the top 5 platforms by sales in the order from highest to lowest, PS4, XOne, 3DS, PS3, X360. The data is taken from the years between 2014 and 2016. 

# <div class="alert alert-block alert-danger">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
#     
# ~~It is uncommon to use data for more than 2-3 years when forecasting next year's sales, even in case of traditional businesses. And in the dynamic computer games industry, taking longer time intervals should be avoided as it will definitely lead to tracking some obsolete trends.~~
#     
# ~~Please choose shorter period for relevant data~~
#     
# 
# </div>

# ### The top 4 growing platforms according to sales betweent the years 2010 and 2016 are PS4, XOne, PSV, WiiU, 3DS.
# 

# In[24]:


# Analyze Data

# Which platforms are leading in sales? Which ones are growing or shrinking? Select several potentially profitable platforms.

# Grouping
platform_year_sales = df_2014_2016.groupby(["platform", "year_of_release"])["total_sales"].sum().unstack()

# Calculating the percentage change in sales from the previous year for each platform
pct_change = platform_year_sales.pct_change(axis=1) * 100

# Calculating the average percentage change for each platform
avg_pct_change = pct_change.mean(axis=1)

# Sorting the platforms by average percentage change in ascending order
shrinking_platforms = avg_pct_change.sort_values()

# Selecting the top 5 shrinking platforms
top_shrinking_platforms = shrinking_platforms.head(5)

print("Top 5 shrinking platforms by sales (2014-2016):")
print(top_shrinking_platforms)

# Data Visualization
plt.figure(figsize=(10, 6))
plt.barh(top_shrinking_platforms.index, top_shrinking_platforms.values, color='red')
plt.xlabel("Average Percentage Change in Sales")
plt.ylabel("Platform")
plt.title("Top 5 Shrinking Platforms by Sales (2014-2016)")
plt.grid(True)
plt.tight_layout()
plt.show()


# ### The above figure shows the platforms with the lowest percentage of sales. The highest is Wii with -76.90% then X360 with -76.43%, then PSP -75.00%, then PS3 -71.68%, then WiiU with -48.82%. 

# Analyze Data

# Total sales for each platform between 2014 and 2016
platform_sales_2014_2016 = df_2014_2016.groupby('platform')['total_sales'].sum().reset_index()
platform_sales_2014_2016 = platform_sales_2014_2016.sort_values('total_sales', ascending=False)

print("Leading platforms (2014-2016):\n", platform_sales_2014_2016.head(5))

# Total sales for each platform over the entire timeframe
total_platform_sales = df.groupby('platform')['total_sales'].sum().reset_index()
total_platform_sales = total_platform_sales.sort_values('total_sales', ascending=False)

print("\nLeading platforms (overall):\n", total_platform_sales.head(5))


# In[26]:


# Data Visualization

# Build a box plot for the global sales of all games, broken down by platform. Are the differences in sales significant? What about average sales on various platforms? Describe your findings.

# Filter data from 2014 to 2016
df_relevant = df[(df["year_of_release"] >= 2014) & (df["year_of_release"] <= 2016)]

# Create a box plot for global sales by platform
plt.figure(figsize=(12, 6))
plt.boxplot([df_relevant[df_relevant["platform"] == platform]["total_sales"] for platform in df_relevant["platform"].unique()], labels=df_relevant["platform"].unique())
plt.xlabel("Platform")
plt.ylabel("Global Sales")
plt.title("Global Sales by Platform (2014-2016)")
plt.xticks(rotation=45)
plt.show()


# <div class="alert alert-block alert-warning"> <b>Reviewer's comment</b> <a 
# class="tocSkip"></a>
# Please use plt.ylim(0,3) that we can see boxplots and not outliers</div>

# ### According to the above figure, the differences in sales is significant between the games of the different platforms. The highest platforms by global sales between 2014 and 2016 is PS4 with more than 14 million dollars and the lowest is PSP with approximately 1 million dollars.

# In[27]:


# Analyze Data

# Calculate average global sales for each platform
platform_avg_sales = df_relevant.groupby("platform")["total_sales"].mean().sort_values(ascending=False)

print("Average Global Sales by Platform (2014-2016):")
print(platform_avg_sales)


# ### The top 5 Average Global Sales by Platforms between the years 2014 and 2016 are X360, PS4, PS3, XOne, WiiU.

# In[28]:


# Data Visualization

# Filter data for PS4 platform
df_ps4 = df[df["platform"] == "PS4"]

# Scatter plot for user reviews vs. total sales
plt.figure(figsize=(8, 6))
sns.scatterplot(x="user_score", y="total_sales", data=df_ps4)
plt.xlabel("User Review Score")
plt.ylabel("Total Sales")
plt.title("User Reviews vs. Total Sales (PS4)")
plt.show()


# ### By analyzing the above scatterplot, there is a strong relationship between user reviews and total sales. For instance, when the user reviews are high, the total sales is high. 

# In[29]:


# Data Visualization

# Scatter plot for professional reviews vs. total sales
plt.figure(figsize=(8, 6))
sns.scatterplot(x="critic_score", y="total_sales", data=df_ps4)
plt.xlabel("Professional Review Score")
plt.ylabel("Total Sales")
plt.title("Professional Reviews vs. Total Sales (PS4)")
plt.show()

# ### The relationship between total sales and professional reviews is strong than that with user reviews. If the reviews are less than 70/100 it drives much less sales if it was higher than that score.

# In[30]:


# Analyze Data

# Take a look at how user and professional reviews affect sales for one popular platform (you choose). Build a scatter plot and calculate the correlation between reviews and sales. Draw conclusions.

# working code

corr = df['user_score'].corr(df['total_sales'])
print(f"Correlation between user_score and total_sales: {corr}")


# ### The correlation between user score and total sales is 0.08 which indicates that there is a positive relations which means as user score increases, total sales tends to increase slightly as well but it is a very weak relation because it is closer to 0.

# In[31]:


# working code

print(df['critic_score'].isnull().sum())


# In[32]:


# working code

print(df['total_sales'].isnull().sum())


# In[33]:


# Convert critic_score to integer data type, preserving NaN values

# working code

df['critic_score'] = pd.to_numeric(df['critic_score'], errors='coerce')


# In[34]:


# working code

print(df['critic_score'].dtype)  # Should print 'int64'
print(df['critic_score'].isna().sum())  # Should print the number of NaN values in the column


# In[35]:


# Analyze Data

# working code

corr = df['critic_score'].corr(df['total_sales'])

print(f"Correlation between critic_score and total_sales: {corr}")


# ### The correlation between critic score and total sales is 0.24 which indicates a positive relationship between them but it is a weak relation. This means that as the critic score increases the total sales increase.

# In[36]:


# Analyze data

# Keeping your conclusions in mind, compare the sales of the same games on other platforms from year 2014 to 2016.

# working code

# Filtering the data for the years 2014 to 2016
df_2014_2016 = df[(df['year_of_release'] >= 2014) & (df['year_of_release'] <= 2016)]

# Grouping
game_sales_by_platform = df_2014_2016.groupby(['name', 'platform'])[['na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'total_sales']].sum().reset_index()

# Sorting
game_sales_by_platform = game_sales_by_platform.sort_values('total_sales', ascending=False)

# Printing the top 10 games by total sales
print("Top 10 games by total sales (2014-2016):")
print(game_sales_by_platform.head(10))


# In[37]:


# Analyze data

# working code

game_sales_by_name = game_sales_by_platform.groupby('name')
print(game_sales_by_name.get_group('Call of Duty: Black Ops 3'))


# ### The highest total sales for the game Call of Duty: Black Ops 3 comes from the PS4 platform with total sales of 14.63 then with XOne with 7.39.

# In[38]:


# Analyze data

# working code

game_sales_by_name = game_sales_by_platform.groupby('name')
print(game_sales_by_name.get_group('Grand Theft Auto V'))


# ### The highest total sales for the game Grand Theft Auto V comes from the Wii platform with total sales of 12.62.

# In[39]:


# Analyze data

# working code

game_sales_by_name = game_sales_by_platform.groupby('name')
print(game_sales_by_name.get_group('Pokemon Omega Ruby/Pokemon Alpha Sapphire'))


# ### The highest total sales for the game Pokemon Omega Ruby/Pokemon Alpha Sapphire comes from the 3DS platform with total sales of 11.68.

# In[40]:


# Analyze data

# working code

game_sales_by_name = game_sales_by_platform.groupby('name')
print(game_sales_by_name.get_group('FIFA 16'))


# ### The highest total sales for the game FIFA 16 comes from the PS4 platform with total sales of 8.85.

# In[41]:


# Analyze data

# working code

game_sales_by_name = game_sales_by_platform.groupby('name')
print(game_sales_by_name.get_group('Star Wars Battlefront (2015)'))


# ### The highest total sales for the game Star Wars Battlefront (2015) comes from the PS4 platform with 7.98.

# In[42]:


# Analyze data

# Take a look at the general distribution of games by genre. What can we say about the most profitable genres? Can you generalize about genres with high and low sales?

# From year 2014 to 2016

# Grouping by genre and calculating the total sales
genre_sales = df_2014_2016.groupby('genre')['total_sales'].sum().reset_index()

# Sorting the genres by total sales in descending order
genre_sales = genre_sales.sort_values('total_sales', ascending=False)

print("Total sales by genre (2014-2016):")
print(genre_sales)

# Boxplot Data Visualization
plt.figure(figsize=(10, 6))

plt.boxplot([df_2014_2016[df_2014_2016['genre'] == genre]['total_sales'] for genre in genre_sales['genre']])

plt.xticks(range(1, len(genre_sales['genre']) + 1), genre_sales['genre'], rotation=45)
plt.xlabel('Genre')
plt.ylabel('Total Sales')
plt.title('Total Sales by Genre (2014-2016)')
plt.grid(True)
plt.tight_layout()
plt.show()


# ### The Boxplot shows that the genre with the highest total sales is Action with 199.36 million dollars then comes Shooter with 170.94 million dollars and the least is Puzzle genre with 2.21 million dollars.

# In[43]:


# Step 4. Create a user profile for each region for the time between 2014 and 2016.

# For each region (NA, EU, JP), determine: The top five platforms. Describe variations in their market shares from region to region

# Grouping
platform_sales = df_2014_2016.groupby('platform')[['na_sales', 'eu_sales', 'jp_sales']].sum().reset_index()

# Top 5 platforms for each region
na_top_platforms = platform_sales.nlargest(5, 'na_sales')
eu_top_platforms = platform_sales.nlargest(5, 'eu_sales')
jp_top_platforms = platform_sales.nlargest(5, 'jp_sales')

print("Top 5 platforms in NA (2014-2016):\n", na_top_platforms[['platform', 'na_sales']])
print("\nTop 5 platforms in EU (2014-2016):\n", eu_top_platforms[['platform', 'eu_sales']])
print("\nTop 5 platforms in JP (2014-2016):\n", jp_top_platforms[['platform', 'jp_sales']])

# Calculating the market share for each top platform in each region
na_total_sales = na_top_platforms['na_sales'].sum()
eu_total_sales = eu_top_platforms['eu_sales'].sum()
jp_total_sales = jp_top_platforms['jp_sales'].sum()

na_top_platforms['market_share'] = (na_top_platforms['na_sales'] / na_total_sales) * 100
eu_top_platforms['market_share'] = (eu_top_platforms['eu_sales'] / eu_total_sales) * 100
jp_top_platforms['market_share'] = (jp_top_platforms['jp_sales'] / jp_total_sales) * 100

print("\nMarket shares of top platforms in NA (2014-2016):\n", na_top_platforms[['platform', 'market_share']])
print("\nMarket shares of top platforms in EU (2014-2016):\n", eu_top_platforms[['platform', 'market_share']])
print("\nMarket shares of top platforms in JP (2014-2016):\n", jp_top_platforms[['platform', 'market_share']])


# ### According to the above data, customer's taste differs in other regions. The only common platform between the 3 regions is PS3.

# In[44]:


# The top five genres from the year 2014 to 2016. Explain the difference.

# Group the data by 'genre' and sum the regional sales
genre_sales = df_2014_2016.groupby('genre')[['na_sales', 'eu_sales', 'jp_sales']].sum().reset_index()

# Get the top 5 genres for each region
na_top_genres = genre_sales.nlargest(5, 'na_sales')
eu_top_genres = genre_sales.nlargest(5, 'eu_sales')
jp_top_genres = genre_sales.nlargest(5, 'jp_sales')

print("Top 5 genres in NA (2014-2016):\n", na_top_genres[['genre', 'na_sales']])
print("\nTop 5 genres in EU (2014-2016):\n", eu_top_genres[['genre', 'eu_sales']])
print("\nTop 5 genres in JP (2014-2016):\n", jp_top_genres[['genre', 'jp_sales']])


# ### Action genre appears to be the most common popular genre among other genres in all regions. Action comes second in North America, first in Europe, and second in Japan.

# In[45]:


# Group the data by 'rating' and sum the regional sales from 2014 to 2016
rating_sales = df_2014_2016.groupby('rating')[['na_sales', 'eu_sales', 'jp_sales']].sum().reset_index()

print("Sales by ESRB Rating in NA (2014-2016):\n", rating_sales[['rating', 'na_sales']])
print("\nSales by ESRB Rating in EU (2014-2016):\n", rating_sales[['rating', 'eu_sales']])
print("\nSales by ESRB Rating in JP (2014-2016):\n", rating_sales[['rating', 'jp_sales']])


# ### Although these findings imply that ESRB ratings can have some effect on sales, it's crucial to remember that other elements like player preferences, game popularity, marketing, and genre can all have a big impact on sales in a given area. Furthermore, the interpretation of these data may be impacted by the fact that the ESRB rating system is largely utilised in North America and that other rating systems may be in use elsewhere.
# 

# In[46]:


# Step 5. 

# Test the following hypotheses: Average user ratings of the Xbox One and PC platforms are the same from the year 2014 to 2016.

# Filter the data for Xbox One and PC platforms
xbox_pc_data = df_2014_2016[(df_2014_2016['platform'] == 'XOne') | (df_2014_2016['platform'] == 'PC')]

# Create New Arrays
xbox_ratings = xbox_pc_data[xbox_pc_data['platform'] == 'XOne']['user_score'].dropna()
pc_ratings = xbox_pc_data[xbox_pc_data['platform'] == 'PC']['user_score'].dropna()

# Perform the two-sample t-test
t_stat, p_value = ttest_ind(xbox_ratings, pc_ratings)

# Print
print(f"p-value: {p_value}")

# Interpret
alpha = 0.05
if p_value < alpha:
    print("We reject the null hypothesis. The average user ratings of the Xbox One and PC platforms are significantly different.")
else:
    print("We fail to reject the null hypothesis. There is no significant difference between the average user ratings of the Xbox One and PC platforms.")


# ### We picked alpha = 0.05 because it is the most popular significance level. H0 hypothesis is that the average The average user ratings of the Xbox One and PC platforms are significantly different. The p-value is higher than alpha which means that we fail to reject the hypothesis and are unable to prove it. The average user ratings of the Xbox One and PC platforms are not significantly different.

# In[47]:


# working code

# Test the following hypotheses: Average user ratings for the Action and Sports genres are different from the year 2014 to 2016.

# Filter the data for Action and Sports genres
action_sports_data = df_2014_2016[(df_2014_2016['genre'] == 'Action') | (df_2014_2016['genre'] == 'Sports')]

# Create New Arrays
action_ratings = action_sports_data[action_sports_data['genre'] == 'Action']['user_score'].dropna()
sports_ratings = action_sports_data[action_sports_data['genre'] == 'Sports']['user_score'].dropna()

# Perform the two-sample t-test
t_stat, p_value = ttest_ind(action_ratings, sports_ratings)

# Print
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Interpret
alpha = 0.05
if p_value < alpha:
    print("We reject the null hypothesis. The average user ratings for the Action and Sports genres are significantly different.")
else:
    print("We fail to reject the null hypothesis. There is no significant difference between the average user ratings for the Action and Sports genres.")


# ### We chose alpha = 0.05 because it is the most popular significance level. H0 hypothesis is that the average user ratings for the Action and Sports genres are significantly differentt. The p-value is less than alpha which means that we to reject the null hypothesis. There is a significant difference between the average user ratings for the Action and Sports genres.

# In[48]:


# Platforms with the highest total sales from year 2014 to 2016.
top_platforms = df_2014_2016.groupby('platform')['total_sales'].sum().reset_index().sort_values('total_sales', ascending=False).head(5)['platform']

# Filtering the data to include only the top platforms
top_platforms_data = df_2014_2016[df_2014_2016['platform'].isin(top_platforms)]

# Calculating mean, standard deviation, and variance of user ratings for each top platform
for platform in top_platforms:
    platform_data = top_platforms_data[top_platforms_data['platform'] == platform]
    user_ratings = platform_data['user_score'].astype(float).dropna()  # Convert to float
    
    if len(user_ratings) > 0:
        mean_rating = user_ratings.mean()
        std_dev_rating = user_ratings.std()
        var_rating = user_ratings.var()
        
        print(f"Platform: {platform}")
        print(f"Mean User Rating: {mean_rating:.2f}")
        print(f"Standard Deviation of User Ratings: {std_dev_rating:.2f}")
        print(f"Variance of User Ratings: {var_rating:.2f}")
        print()
    else:
        print(f"Platform: {platform}")
        print("No user ratings available.")
        print()

# # Conlusion
# 
# ### The top 5 platforms in terms of overall total sales are: PS2, X360, PS3, Wii, DS
# ### There is a total of 22 platforms with zero sales in 2016 such as 2600, 3DO, DC, DS, GB and others.
# ### Top 5 growing platforms (2014-2016):PS4, XOne,  3DS, PS3, X360.
# 
# ### Top 5 shrinking platforms (2014-2016): Wii, X360, PSP, PS3, WiiU
# 
# ### There is a strong relationship between user reviews and total sales but the relationship between professional reviews and total sales is stronger.
# 
# ### The highest platforms by global sales between 2014 and 2016 is PS4 with more than 0.76 million dollars and the lowest is PSP with approximately 0.02 million dollars.
# 
# ### The lowest genre by total sales is Puzzle with approximately 0.2 million dollars.
# 
# ### The only common platform with high sales between the 3 regions is PS3.
#  
# ### The average user ratings of the Xbox One and PC platforms are not significantly different.
# 
# ### There is a significant difference between the average user ratings for the Action and Sports genres.





