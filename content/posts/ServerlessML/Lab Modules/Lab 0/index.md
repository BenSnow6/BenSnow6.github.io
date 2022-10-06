---
title: "Serverless Machine Learning Module 0"
date: 2022-10-05T08:06:25+06:00
description: Following along with the serverless-ml online course by Jim Dowling and Hopsworks.
menu:
  sidebar:
    name: Lab 0
    identifier: ServerlessML-module0
    weight: 10
tags: ["Serverless", "Pandas"]
categories: ["Basic"]
---

05/10/2022
## Intro
This course aims to allow a data scientist to deploy machine learning and predictive services online without the use of a server. This is useful in creating powerful business applications to show the predictive effect of a machine learning model quickly and effectively.

## Labs
### Pandas
We started with a simple pandas introduction lab involving converting to datetime from a given date field. After, we looked at applying functions to a series within the dataframe with a few different methods. We used lambda functions to create a new column based on an amount being greater than a specified value. We also used the `.apply` function to apply a user defined function to a series. Along with this, rolling windows were introduced to compute the highest value over a given time period, in this case, the highest amount spent in the past day.
Additionally, we looked at how vectorized operations are much (around 50x) faster than `.apply` functions when operating on dataframes. And example of a vectorized function is
```python
df['a'] ** 2 # raise all values in series 'a' to the power of two
```
Finally we looked at applying power transformations to a near exponentially distributed dataset to convert it to a near gaussian distribution, therefore making it more suitable for deep learning. In the notebook I wrote that 'We wish for distributions that are more Gaussian for machine learning algorithms since they operate on the assumption that the distributions are smoothly varying, are differentiable, and have the possibility for infinite surprise.'

### Apples and oranges
In these two notebooks, we create a simple decision tree classifier and a Logistic regression model to predict the label of apples and oranges based on their red-green colour touples. We found that the logistic regression model performed well but mis-classified one of the test examples hinting at overfitting and poor generalisation of the model. This was attributed to an out of sample test feature being used. See the notebook re-and-green-apples-vs-oranges.ipynb for more details.

### Feature store intro
In this notebook, we set up a Hopsworks account online and began by creating an API key with which we can access the Hopsworks online platform.

Motivation for using Hopsworks is to create feature stores, online databases that store artefacts useful for training and inference with machine learning models. We wish to create a feature group that is a table of features that are written to the feature store as a dataframe.

Feature groups require a name and are usually provided with a version, description, primary key, and event time. We defined a feature group as follows:
```python
fg = fs.get_or_create_feature_group(
     name="credit_card_transactions",
     version=1,
     description="Credit Card Transaction data",
     primary_key=['credit_card_number'],
     event_time='trans_datetime'
)
```
This particular use case is creating features for a credit card fraud detection model.

Features can be written to the feature group by inserting a dataframe:
```python
fg.insert(df) # adds the dataframe to the feature group with a hopsworks job
```

We can see the online feature group[ here](https://c.app.hopsworks.ai/p/2289/fs/2234/fg/2883) which also includes plenty of metadata about the group.

#### Reading from the feature store
We can query the feature group by selecting columns within the dataset and using the `fg.select(['column1, column2'])` function.
Additionally, a feature view can be made with the returned selection query which allows for model inference (not training). We can define which of the columns are selected as features (query) and which of the columns is the label:

```python
query = fg.select(["amount", "location", "fraud"])

fv = fs.create_feature_view(name="credit_card_transactions",
                            version=1,
                            description="Features from the credit_card_transactions FG",
                            labels=["fraud"],
                            query=query)
```
Again, we can see this online at Hopsworks [here](https://c.app.hopsworks.ai:443/p/2289/fs/2234/fv/credit_card_transactions/version/1)

#### Using feature views
We can use a feature view to set training and test features and labels like we would a dataframe. Hopsworks has its own train test split function that we can apply to a feature view to return this split.

```python
X_train, X_test, y_train, y_test = fv.train_test_split(0.5)
```
I love when it's so simple!

We can also save some of the training data to disk if we wish and then use these to read in during training. This can be done by writing the feature view data to a .csv file. Very useful for large datasets.

```python
td_version, td_job = fv.create_train_test_split(
    description = 'Transactions fraud batch training dataset',
    data_format = 'csv', # file format
    test_size = 0.5, # 50% tr/te split
    write_options = {'wait_for_job': True}, # waits for the job to finish
    coalesce = True, # save the dataframe into a single partition before writing
)
```
`td_version` here is the version of the training data for the feature view and `td_job` can be used to follow the jo progress. A little popup to follow the job link is outputted during execution and the link to the job [can be seen here](https://c.app.hopsworks.ai/p/2289/jobs/named/credit_card_transactions_1_1_create_fv_td_05102022114302/executions) .

This train test split can then be seen on the Hopsworks website under Project Settings -> File Browser -> projectname_Training_Datasets.
Super easy!

We can then grab the x_train, y_train, x_test, y_test from the feature view with the dataset version we want:
```python
X_train, y_train, X_test, y_test = fv.get_train_test_split(td_version)
```
Note we pass `td_version` here that was generated above.

#### Aggregating data
We now will do a little feature engineering. We will start with some aggregations of data within the training dataset. One of these aggregations is to sum up all the amounts spent by the same credit card and save that in a new column called "total spent". Then we can add when this total is calculated to.

```python
# group and sum
df = df[["credit_card_number", "amount"]].groupby("credit_card_number").sum()
# rename col to total spent
df.rename(columns={"amount": "total_spent"}, inplace=True)
# set a new column storing the date up to the amounts were summed over
df["as_of_datetime"] = df[["credit_card_number", "trans_datetime"]].groupby("credit_card_number").max()
# Move credit card number to be a column since it is currently the index
df.reset_index(inplace=True)
```

Now a new feature group can be created holding this new dataset with new features!
```python
fg2 = fs.get_or_create_feature_group(
     name="credit_card_spending",
     version=1,
     description="Credit Card Spending",
     primary_key=['credit_card_number'],
     event_time='as_of_datetime' # new event time is based on the sum until date
)
```
This can then be inserted to the feature group:
```python
fg2.insert(df, write_options={"wait_for_job": False})
```

We can now add more data to the original feature group and insert it into the feature group.
```python
more_data = {

    'credit_card_number': ['9999 8888 7777 6666', '9999 8888 7777 6666','9999 8888 7777 6666', '9999 8888 7777 6666'],
    'trans_datetime': ['2022-01-02 04:11', '2022-01-03 07:24', '2022-01-05 10:33', '2022-01-05 11:50'],
    'amount': [55.67, 84, 77.95, 183],
    'location': ['San Francisco', 'San Francisco', 'Dublin', 'Dublin'],
    'fraud': [False, False, False, False]
}
# create df from dictionary
df3 = pd.DataFrame.from_dict(more_data)
# apply date time transformation
df3['trans_datetime']= pd.to_datetime(df3['trans_datetime'])
# grab the feature group from the feature store
fg = fs.get_feature_group(name="credit_card_transactions", version=1)  
# insert the new data to the feature group
fg.insert(df3, write_options={"wait_for_job": False})
```

We can now see the newly added feature group entries on [Hopsworks](https://c.app.hopsworks.ai/p/2289/fs/2234/fg/2883/data-preview)  (reminder, only I can see these, you'll need to create a Hopsworks account and follow the tutorial to see your own feature groups).

#### Time series window aggregations
We now use some windowing techniques to count how much money was spent per day (window length = '1d'). We need the `event_time` column as the index so start by reading the feature group to a df:
```python
df = fg.read() # read feature group
df = df.set_index('trans_datetime') # set index to event_time
df = df.sort_index() # sorts the index based on the times
```

Now we calculate some rolling window features, the max amount in the past day and the mean amount in the past day:
```python
df['rolling_max_1d'] = df.rolling('1d').amount.max() # max past day
df['rolling_mean_1d'] = df.rolling('1d').amount.mean() # mean past day
```
Then we reset the index (since doing these sorts of operations sets the `event_time` to be the index) and create a new feature group and insert the new data to it.
```python
fg_agg = fs.get_or_create_feature_group(
     name="credit_card_rolling_windows",
     version=1,
     description="Daily Credit Card Spending",
     primary_key=['credit_card_number'],
     event_time='trans_datetime' # remember to set the event_time back to trans_dt
)
```
We've successfully created three feature groups, each with their own sets of features in them that we can read from whenever we want! (`fg = fs.get_feature_group(name="", version=x`).

#### Feature view with features from multiple feature groups!
Ok, let's suppose we want to grab features from lots of different feature groups and use them to train a model, how do we do that? Well it's quite easy, we just use a query to select the features from one feature group that we wish to use (`fg.select_all()` for example that selects all features from `fg`) and join that to the selection from another feature group (`fg_agg.select(['rolling_max_1d'])` taking only the `rolling_max_1d` feature from the `fg_agg` feature group).

We can see an example of something like this here:
```python
# create query
query = fg.select_all().join(fg_agg.select(['rolling_max_1d', 'rolling_mean_1d']))
# read data from query
training_data = query.read()
```

We can now create a feature view containing the data from this query:
```python
fv = fs.create_feature_view(name="credit_card_fraud_rolling",
                            description="Features for a model to predict credit card fraud, including rolling windows",
                            version=1,
                            query=query)
```
Check out the feature view [here on Hopsworks](https://c.app.hopsworks.ai:443/p/2289/fs/2234/fv/credit_card_fraud_rolling/version/1) . What's great is that there are links in the feature view to take you to the exact feature group where the feature columns (or labels) have come from!
We can use this feature view to create a test train split of the data:
```python
X_train, y_train, X_test, y_test = fv.train_test_split(0.5)
```
Don't forget we can grab all the data from a feature group and read it into a dataframe shown below:
```python
# specify which feature group to read
fg = fs.get_feature_group(name="credit_card_transactions", version=1)
# read it to a df
read_df = fg.read()
```

#### Filtering
We can also grab only specific parts of the dataset by reading with a query (filtering). For example, we can read only the data with amounts greater than 100:

```python
from hsfs.feature import Feature
# only grab features from the fg with amount >100
big_amounts_df = fg.filter(Feature("amount") > 100).read()
```

## Summary
In these labs, we revised some elementary pandas techniques, fit logistic and decision tree models to simple classification problems, and investigated the use of feature stores on the Hopsworks ecosystem!