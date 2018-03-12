
# Relax Challenge
***

In order to perform EDA on this dataset, I had to create a function that allows you to create labels for defining an 'adopted_user'.

**Interesting findings**
1. Adopted Users accounted for 12% of the dataset
2. Source Creation - Adopted Users were mostly invited by Org_Invite and Guest_Invite
3. Marketing Drip and Mailing List Opt didn't show anything to interesting
4. Majority of the non-adopted users registered from Jan-May

**Most Interesting Finding:**
1. The most interesting finding is that the least amount of registration occurred on May for adopted users, but it's the most registered in non-adopted users.

**Further Investigation:** It would be great if we can have more data about user activity. One EDA that I would like to perform more on is the difference betweeen registrations of adopted vs non-adopted users during the month of May.

Defining an **"adopted user"** as a user who has logged into the product on **three seperate days in at least one seven-day period**, identify which factors predict future user adoption.

**Please send us:**
- a brief writeup of your findings (the more conscise, thebetter -- no more than one page)
- summary tables
- graphs
- code
- queries that can help us understand your approach. 

Please not any factors you considered or investigation you did, even if they did not pan out. Feel free to identify any further research or data you think would be valuable. 


```python
# Import necessary libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
%matplotlib inline
```

## Create Active User Function 
***
Apply this function to the dataset to create the 'adopted_user' labels, where an adopted user represents people who logged in at least 3 times within a 7 day period.


```python
def active_users(period, days_logged, user):

    import pandas as pd
    from pandas import DataFrame, Series

    visited = len(user.index) #get the number of times the user logged in
    i, count = 0, 1
    active_user = False

    while count < days_logged:
        if (i+2) < visited: #needs to be at least 3 entries left	
            if (user['time_stamp'].iloc[i + 1] - user['time_stamp'].iloc[i]) <= pd.Timedelta(days=period) and (user['time_stamp'].iloc[i + 1] - user['time_stamp'].iloc[i]) > pd.Timedelta(days=1) :
                count += 1 #logged in twice within a 7 day period
                new_timeframe = pd.Timedelta(days=7) - ((user['time_stamp'].iloc[i + 1] - user['time_stamp'].iloc[i]))
                if (user['time_stamp'].iloc[i + 2] - user['time_stamp'].iloc[i + 1]) <= new_timeframe and (user['time_stamp'].iloc[i + 2] - user['time_stamp'].iloc[i + 1]) > pd.Timedelta(days=1):
                    active_user = True #they logged in three times within a 7 period window
                    count += 1
                else: 
                    i += 1
                    count = 1
            else:
                i += 1
                count = 1
        else:
            count = days_logged
    return active_user
```

## Load 'takehome_user_engagement.csv'


```python
df_eng = pd.read_csv('takehome_user_engagement.csv')
df_eng.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time_stamp</th>
      <th>user_id</th>
      <th>visited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-04-22 03:53:30</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-11-15 03:45:04</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-11-29 03:45:04</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-09 03:45:04</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-12-25 03:45:04</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_eng['time_stamp'] = pd.to_datetime(df_eng.time_stamp)
```


```python
df3 = df_eng.groupby('user_id').filter(lambda x: (len(x) > 2) & (active_users(period=7, days_logged=3, user=x) ==True))
```


```python
# Now we can create our labels with these unique users that logged in 3 times or more within 7 days
unique_users = df3.user_id.unique()
unique_users
```




    array([    2,    10,    33, ..., 11969, 11975, 11988], dtype=int64)



## Load 'takehome_users.csv'


```python
# Load json file into pandas dataframe
df_user = pd.read_csv('takehome_users.csv', encoding ='latin1')
df_user.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>object_id</th>
      <th>creation_time</th>
      <th>name</th>
      <th>email</th>
      <th>creation_source</th>
      <th>last_session_creation_time</th>
      <th>opted_in_to_mailing_list</th>
      <th>enabled_for_marketing_drip</th>
      <th>org_id</th>
      <th>invited_by_user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2014-04-22 03:53:30</td>
      <td>Clausen August</td>
      <td>AugustCClausen@yahoo.com</td>
      <td>GUEST_INVITE</td>
      <td>1.398139e+09</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>10803.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2013-11-15 03:45:04</td>
      <td>Poole Matthew</td>
      <td>MatthewPoole@gustr.com</td>
      <td>ORG_INVITE</td>
      <td>1.396238e+09</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>316.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2013-03-19 23:14:52</td>
      <td>Bottrill Mitchell</td>
      <td>MitchellBottrill@gustr.com</td>
      <td>ORG_INVITE</td>
      <td>1.363735e+09</td>
      <td>0</td>
      <td>0</td>
      <td>94</td>
      <td>1525.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2013-05-21 08:09:28</td>
      <td>Clausen Nicklas</td>
      <td>NicklasSClausen@yahoo.com</td>
      <td>GUEST_INVITE</td>
      <td>1.369210e+09</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5151.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2013-01-17 10:14:20</td>
      <td>Raw Grace</td>
      <td>GraceRaw@yahoo.com</td>
      <td>GUEST_INVITE</td>
      <td>1.358850e+09</td>
      <td>0</td>
      <td>0</td>
      <td>193</td>
      <td>5240.0</td>
    </tr>
  </tbody>
</table>
</div>




## Create "adopted_user" label

**Summary:**
- About 88% of the users are not adopted users
- About 12% of the users are adopted users


```python
df_user['adopted_user'] = df_user.object_id.isin(unique_users)
```


```python
df_user.adopted_user.value_counts(1)
```




    0    0.875583
    1    0.124417
    Name: adopted_user, dtype: float64




```python
df_user.dtypes
```




    object_id                       int64
    creation_time                  object
    name                           object
    email                          object
    creation_source                object
    last_session_creation_time    float64
    opted_in_to_mailing_list        int64
    enabled_for_marketing_drip      int64
    org_id                          int64
    invited_by_user_id            float64
    adopted_user                     bool
    dtype: object




```python
# Convert the boolean label into an int
df_user.adopted_user = df_user.adopted_user.astype(int)
```


```python
df_user.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>object_id</th>
      <th>creation_time</th>
      <th>name</th>
      <th>email</th>
      <th>creation_source</th>
      <th>last_session_creation_time</th>
      <th>opted_in_to_mailing_list</th>
      <th>enabled_for_marketing_drip</th>
      <th>org_id</th>
      <th>invited_by_user_id</th>
      <th>adopted_user</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2014-04-22 03:53:30</td>
      <td>Clausen August</td>
      <td>AugustCClausen@yahoo.com</td>
      <td>GUEST_INVITE</td>
      <td>1.398139e+09</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>10803.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2013-11-15 03:45:04</td>
      <td>Poole Matthew</td>
      <td>MatthewPoole@gustr.com</td>
      <td>ORG_INVITE</td>
      <td>1.396238e+09</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>316.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2013-03-19 23:14:52</td>
      <td>Bottrill Mitchell</td>
      <td>MitchellBottrill@gustr.com</td>
      <td>ORG_INVITE</td>
      <td>1.363735e+09</td>
      <td>0</td>
      <td>0</td>
      <td>94</td>
      <td>1525.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2013-05-21 08:09:28</td>
      <td>Clausen Nicklas</td>
      <td>NicklasSClausen@yahoo.com</td>
      <td>GUEST_INVITE</td>
      <td>1.369210e+09</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5151.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2013-01-17 10:14:20</td>
      <td>Raw Grace</td>
      <td>GraceRaw@yahoo.com</td>
      <td>GUEST_INVITE</td>
      <td>1.358850e+09</td>
      <td>0</td>
      <td>0</td>
      <td>193</td>
      <td>5240.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Explore Creation Source

**Summary:**
- Majority of the users were invited to an organization as a full member
- Doesn't seem to have any difference between adopted users and non-adopted users


```python
df_user.creation_source.value_counts(1)
```




    ORG_INVITE            0.354500
    GUEST_INVITE          0.180250
    PERSONAL_PROJECTS     0.175917
    SIGNUP                0.173917
    SIGNUP_GOOGLE_AUTH    0.115417
    Name: creation_source, dtype: float64




```python
f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y="creation_source", hue='adopted_user', data=df_user).set_title('Creation Source Distribution');
```


![png](output_22_0.png)


## Explore Marketing Drip


```python
f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y="enabled_for_marketing_drip", hue='adopted_user', data=df_user).set_title('Marketing Drip Distribution');
```


![png](output_24_0.png)


## Explore Mailing List Opt


```python
f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y="opted_in_to_mailing_list", hue='adopted_user', data=df_user).set_title('Opt Mailing List Distribution');
```


![png](output_26_0.png)


## Create new features (Month and Year)


```python
df_user['creation_time'] = pd.to_datetime(df_user.creation_time)
```


```python
df_user['year'] = df_user['creation_time'].dt.year
df_user['month'] = df_user['creation_time'].dt.month
df_user['day'] = df_user['creation_time'].dt.day
```


```python
df_user.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>object_id</th>
      <th>creation_time</th>
      <th>name</th>
      <th>email</th>
      <th>creation_source</th>
      <th>last_session_creation_time</th>
      <th>opted_in_to_mailing_list</th>
      <th>enabled_for_marketing_drip</th>
      <th>org_id</th>
      <th>invited_by_user_id</th>
      <th>adopted_user</th>
      <th>weekday</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2014-04-22 03:53:30</td>
      <td>Clausen August</td>
      <td>AugustCClausen@yahoo.com</td>
      <td>GUEST_INVITE</td>
      <td>1.398139e+09</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>10803.0</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>4</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2013-11-15 03:45:04</td>
      <td>Poole Matthew</td>
      <td>MatthewPoole@gustr.com</td>
      <td>ORG_INVITE</td>
      <td>1.396238e+09</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>316.0</td>
      <td>1</td>
      <td>4</td>
      <td>2013</td>
      <td>11</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2013-03-19 23:14:52</td>
      <td>Bottrill Mitchell</td>
      <td>MitchellBottrill@gustr.com</td>
      <td>ORG_INVITE</td>
      <td>1.363735e+09</td>
      <td>0</td>
      <td>0</td>
      <td>94</td>
      <td>1525.0</td>
      <td>0</td>
      <td>1</td>
      <td>2013</td>
      <td>3</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2013-05-21 08:09:28</td>
      <td>Clausen Nicklas</td>
      <td>NicklasSClausen@yahoo.com</td>
      <td>GUEST_INVITE</td>
      <td>1.369210e+09</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5151.0</td>
      <td>0</td>
      <td>1</td>
      <td>2013</td>
      <td>5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2013-01-17 10:14:20</td>
      <td>Raw Grace</td>
      <td>GraceRaw@yahoo.com</td>
      <td>GUEST_INVITE</td>
      <td>1.358850e+09</td>
      <td>0</td>
      <td>0</td>
      <td>193</td>
      <td>5240.0</td>
      <td>0</td>
      <td>3</td>
      <td>2013</td>
      <td>1</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



## Explore Non-Adopted Users by Month Registration

**Summary:**
- It looks like majority of the non-adopted users registered from Jan-May


```python
# Count Plot (a.k.a. Bar Plot)
sns.countplot(x='month', data=df_user[df_user['adopted_user']==0]).set_title('Account Creation for Non-Adtoped Users(Month)');
 
# Rotate x-labels
plt.xticks(rotation=-45)
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]),
     <a list of 12 Text xticklabel objects>)




![png](output_32_1.png)


## Explore Adopted Users by Month Registration

**Summary:**
- The most interesting finding is that the least amount of registration occurred on May for adopted users, but it's the most registered in non-adopted users. 


```python
# Count Plot (a.k.a. Bar Plot)
sns.countplot(x='month', data=df_user[df_user['adopted_user']==1]).set_title('Account Creation for Adopted Users (Month)');
 
# Rotate x-labels
plt.xticks(rotation=-45)
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]),
     <a list of 12 Text xticklabel objects>)




![png](output_34_1.png)


## Explore Account Creation by Year


```python
# Count Plot (a.k.a. Bar Plot)
sns.countplot(x='year', data=df_user).set_title('Account Creation (Year)');
 
# Rotate x-labels
plt.xticks(rotation=-45)
```




    (array([0, 1, 2]), <a list of 3 Text xticklabel objects>)




![png](output_36_1.png)



```python

```
