#!/usr/bin/env python
# coding: utf-8

# # YouTube and Streamer Insights: A Closer Look

#  ## Introduction

# In the introduction, we explore a datasheet packed with extensive information on YouTube streamers, presenting a detailed snapshot of their online profiles

# ## Quick recap on the Dataset

#  This dataset comprises approximately 1000 entries related to YouTube and streamers, featuring 9 key columns.They are 
# 
# 1. Rank
# 2. Username
# 3. Categories
# 4. Suscribers
# 5. Country
# 6. Visits
# 7. Likes
# 8. Comments
# 9. Links

# The Task is to perform a comprehensive analysis of the dataset to extract insights about the top YouTube content creators.
# LET'S GET STARTED

# In[54]:


# importing necessay libraies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # 1. Data Exploration
# - Start by exploring the dataset to understand its structure and identify key variables.
# - Check for missing data and outliers.

# In[7]:


# Importing the dataset

df=pd.read_csv("C:\\Users\\Lenovo\\OneDrive\\Desktop\\youtubers_final_raw_data.csv")
df.head()


# In[5]:


print(df.info())


# In[6]:


df.describe()


# - The dataset comprises 1000 records distributed across 9 columns. 
# - Notably, key columns of interest encompass metrics such as Likes, Views, Subscriptions, and Comments

# # Data Cleaning

# - In the 'Country' column, I have translated the country names and categories from Spanish to English using Excel.
# - Additionally, for countries with a single entry, I have transformed them into 'Others' for consolidation.

# In[10]:


# For Example let me show you the Country column alone

country_column = df["Country"]
print(country_column)


# ## Searching of NULL values 

# In[21]:


# To find NULL values
null_values = df.isnull()
# Count the number of missing values in each column
null_count_per_column = df.isnull().sum()
print(null_count_per_column)


# In[20]:


# To remove the NULL values in the dataset 
df = pd.DataFrame(data)
df.dropna(inplace=True)


# ## Searching of Duplicate values using "Links" column 

# In[11]:


# To find Duplicate values
duplicates = df[df.duplicated(subset='Links', keep=False)]
print(duplicates)


# In[23]:


# to drop duplicate values

df_no_duplicates = df.drop_duplicates(subset='Links', keep='first')
df.head()


# # 2. Trend Analysis

# - Identify trends among the top YouTube streamers. Which categories are the most popular?
# - Is there a correlation between the number of subscribers and the number of likes or comments?

# In[29]:


# to find correlation between no.of.Subscribers and no.of.likes
subset_df = df[['Suscribers', 'Likes', 'Comments','Rank']]
correlation_matrix = subset_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap: Subscribers, Likes, Comments,Rank')
plt.show()


# ## Correlation Insights  
# - The heatmap shows that there is a strong positive correlation between subscribers and likes.
# - A weaker positive correlation between subscribers and comments. 
# - There is also a positive correlation between likes and comments.
# - Rank has no correlation with any of these columns. 

# In[32]:


subset_df = df[['Suscribers', 'Likes', 'Comments','Rank']]
sns.pairplot(subset_df)
plt.show()


# - Similarly in heatmap here we can conclude the points, That is
# - A weaker positive correlation is observed between subscribers and comments.
# - A positive correlation exists between likes and comments.
# - Rank does not show any correlation with these metrics.

# In[34]:


plt.figure(figsize=(10, 6))
sns.countplot(y='Categories', data=df, order=df['Categories'].value_counts().index, palette='viridis')
plt.title('Top YouTube Streamer Categories')
plt.xlabel('Number of Streamers')
plt.ylabel('Categories')
plt.show()


# ## Categories Insights
# - The top categories are Dance and Music, Movies, Animation, Videogames, and Others.
# - The bottom categories are Daily Vlogs, News and Politics, Education, Science and Technology, Food, Comedy, and Sports.
# - Dance and Music, Movies, and Animation these categories have high no.of.subscribers.
# - Daily Vlogs, News and Politics, Education, Science and Technology, Food, Comedy, and Sports are the least popular categories on YouTube
# 

# ## 3. Audience Study
# - Analyze the distribution of streamers' audiences by country. Are there regional preferences for specific content categories?

# In[55]:


grouped_data = df.groupby(['Country', 'Categories']).size().reset_index(name='Streamer Count')
plt.figure(figsize=(12, 8))
sns.barplot(x='Country', y='Streamer Count', hue='Categories', data=grouped_data, palette='deep')
plt.title('Distribution of Streamers\' Audiences by Country and Category')
plt.xlabel('Country')
plt.ylabel('Number of Streamers')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# - The United States stands out with the highest global audience distribution.
# - The "Dance and Music" category consistently engages audiences prominently.
# - India follows as the second-largest audience hub, with a notable focus on the "Dance and Music" category.

# In[59]:


# piechart display for specific country (Example:- India)
grouped_data = df.groupby(['Country', 'Categories']).size().reset_index(name='Streamer Count')
selected_country = 'India'
country_data = grouped_data[grouped_data['Country'] == selected_country]
plt.figure(figsize=(10, 8))
plt.pie(country_data['Streamer Count'], labels=country_data['Categories'], autopct='%1.1f%%', startangle=140)
plt.title(f'Distribution of Streamers\' Audiences in {selected_country} by Category')
plt.show()


# In[22]:


df_result=df.groupby(['Country', 'Categories']).sum().sort_values(by='Visits', ascending=False).drop([ 'Links', 'Rank','Username', 'Comments'], axis=1)
display(df_result)


# In[60]:


df.groupby(['Username']).sum().sort_values(by='Visits', ascending=False)[:10].drop(['Categories', 'Country', 'Links', 'Rank', 'Comments'], axis=1)


# ## 4.Performance Metrics
# - Calculate and visualize the average number of subscribers, visits, likes, and comments.
# - Are there patterns or anomalies in these metrics?

# In[27]:


average_metrics = df[['Suscribers', 'Visits', 'Likes', 'Comments']].mean()
plt.figure(figsize=(10, 6))
average_metrics.plot(kind='bar', color=['blue', 'orange', 'green', 'red'])
plt.title('Average Metrics')
plt.xlabel('Metrics')
plt.ylabel('Average Value')
plt.show()


# ##  Performance Metrics Insights
# - Indeed, the data exhibits anomalies. It is peculiar that the subscriber count surpasses the number of likes,Comments and visitors.

# ## 5. Content Categories:
# 
# - Explore the distribution of content categories. Which categories have the highest number ofstreamers?
# - Are there specific categories with exceptional performance metrics? 

# In[32]:


category_distribution = df['Categories'].value_counts()
plt.figure(figsize=(12, 8))
category_distribution.plot(kind='barh', color='skyblue')
plt.title('Distribution of Content Categories')
plt.xlabel('Categories')
plt.ylabel('Number of Streamers')
plt.show()


# In[33]:


category_likes_distribution = df.groupby('Categories')['Likes'].sum()

# Plot the distribution of content categories based on Likes
plt.figure(figsize=(12, 8))
category_likes_distribution.sort_values(ascending=False).plot(kind='barh', color='skyblue')
plt.title('Distribution of Content Categories Based on Likes')
plt.xlabel('Number of Likes')
plt.ylabel('Categories')
plt.show()


# ## Content Categories Insights 
# - "Dance and Music" category having highest no.of.streamers.
# - The "Videogames" category has the highest average number of Likes, followed by the "Animation" and "Daily Vlogs" categories.
# - The "Dance and Music" category ranks fourth in terms of the number of likes, but it holds the top position among all categories when considering the number of streamers.

# ## 6. Brands and Collaborations
# - Analyze whether streamers with high performance metrics receive more brand collaborations and marketing campaigns.

# #### Streamers with high performance metrics may not receive more brand collaborations or marketing campaigns.It's essential to acknowledge that the available data might be insufficient for a comprehensive analysis
# 

# ## 7. Benchmarking
# - Identify streamers with above-average performance in terms of subscribers, visits, likes, and comments.
# - Who are the top-performing content creators?

# In[38]:


# Calculate average values for each metric
average_metrics = df[['Suscribers', 'Visits', 'Likes', 'Comments']].mean()

# Identify streamers with above-average performance
above_average_streamers = df[
    (df['Suscribers'] > average_metrics['Suscribers']) &
    (df['Visits'] > average_metrics['Visits']) &
    (df['Likes'] > average_metrics['Likes']) &
    (df['Comments'] > average_metrics['Comments'])
]

# Determine the top-performing content creators
top_performers = above_average_streamers.nlargest(5,'Likes')  # Adjust the number as needed or choose a different metric
print("Top-Performing Content Creators:")
print(top_performers[['Username', 'Suscribers', 'Visits', 'Likes', 'Comments']])


# In[43]:


above_average_streamers.groupby('Username').sum().sort_values(by = 'Likes', ascending = False).drop([ 'Links', 'Rank'], axis=1)


# ## Benchmarking Insights
# - Streamers with above-average performance in terms of subscribers, visits, likes, and comments are MrBeast,MrBeast2,DaFuqBoom,alanbecker,fedevigevani
# - The 7 top-performing content creators,they are
# - 1. MrBeast
# - 2. MrBeast2
# - 3. DaFuqBoom
# - 4. alanbecker
# - 5. fedevigevani
# - 6. souravjoshivlogs7028
# - 7. AboFlah
# - 8. A4a4a4a4
# - 9. dream
# - 10. TaylorSwift 

# ## 8. Content Recommendations
# - Propose a system for enhancing content recommendations to YouTube users based on streamers' categories and performance metrics.

# In[49]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

recommendation_data = df[['Username', 'Categories', 'Suscribers', 'Visits', 'Likes', 'Comments']]

scaler = MinMaxScaler()
performance_metrics = recommendation_data[['Suscribers', 'Visits', 'Likes', 'Comments']]
normalized_metrics = scaler.fit_transform(performance_metrics)
recommendation_data[['Suscribers', 'Visits', 'Likes', 'Comments']] = normalized_metrics

user_item_matrix = pd.pivot_table(recommendation_data, values=['Suscribers', 'Visits', 'Likes', 'Comments'],
                                  index='Username', columns='Categories', aggfunc='mean', fill_value=0)

cosine_sim = cosine_similarity(user_item_matrix, user_item_matrix)

similarity_df = pd.DataFrame(cosine_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_content_recommendations(username, top_n=5):
    similar_streamers = similarity_df[username].sort_values(ascending=False)[1:top_n+1].index
    return recommendation_data[recommendation_data['Username'].isin(similar_streamers)]

recommended_content = get_content_recommendations('TaylorSwift', top_n=5)
print("Content Recommendations for TaylorSwift:")
print(recommended_content[['Username', 'Categories', 'Suscribers', 'Visits', 'Likes', 'Comments']])


# ## Content Recommendations Insights
# - 5 Top most accounts similar to "TaylorSwift".They are
# - 1. nickiminaj 
# - 2. Zayn
# - 3. RauwAlejandroTv
# - 4. AvrilLavigne
# - 5. CNCOMusic

# In[ ]:




