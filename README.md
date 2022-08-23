# AI200 - Human or Robot?

## My top solution for [AI200 Mar 2022: Human or Robot? Capstone Competition](https://www.kaggle.com/competitions/ai200-mar-2022-human-or-bot) 

Please view my codes at this [nbviewer link](https://nbviewer.org/github/teorongchang/ai200-capstone-project/blob/main/modelling/teorongchang_Capstone_Project_AI200.ipynb)

## **Overview of project**

* Participated in Facebook Recruitment Prediction Competition hosted on the Kaggle platform.
* Designed and developed Machine Learning models for Bot detection. 
* Generated time series and entropy features.
* Incorporated SMOTE (tackle class imbalance problem).
* Performed Hyperparameter Tuning to get optimal prediction AUC.
* Achieved Top 1 on InClass Kaggle leaderboard (0.88797 AUC) and Top 60 on Global Kaggle leaderboard (0.93162 AUC).

## Code and Resources Used 
**Python Version:** 3.9 <br>
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, plotly, imblearn, xgboost, catboost, lightgbm <br>

## **Competition Description**

The purpose of this competition is to chase down robots for an online auction site. Human bidders on the site are becoming increasingly frustrated with their inability to win auctions vs. their software-controlled counterparts. As a result, usage from the site's core customer base is plummeting.

The goal of this competition is to identify online auction bids that are placed by "robots", helping the site owners easily flag these users for removal from their site to prevent unfair auction activity.

## **Data Description**

Bidder Dataset

| Column        | Description   |
| ------------- |:-------------:|
| bidder_id | Unique identifier of a bidder. |
| payment_account | Payment account associated with a bidder. These are obfuscated to protect privacy. |
| address | Mailing address of a bidder. Thse are obfuscated to protect privacy. |
| outcome | Label of a bidder indicating whether or not it is a robot. Value 1.0 indicates a robot, where value 0.0 indicates human. |

Bid Dataset

| Column        | Description   |
| ------------- |:-------------:|
| bid_id | Unique id for this bid. |
| bidder_id | Unique identifier of a bidder. |
| auction | Unique identifier of an auction. |
| merchandise | The category of the auction site campaign, which means the bidder might come to this site by way of searching for "home goods" but ended up bidding for "sporting goods" - and that leads to this field being "home goods". |
| device | Phone model of a visitor. |
| time  | Time that the bid is made (transformed to protect privacy). |
| country | The country that the IP belongs to. |
| ip | IP address of a bidder (obfuscated to protect privacy).. |
| url  | url where the bidder was referred from (obfuscated to protect privacy). |

## **Exploratory Data Analysis (EDA)**

My group have mainly used SweetViz and Pandas Profiling to perform EDA on the datasets to investigate on Statistics and Outliers in the dataset.

It was found that the training data is largely imbalanced - there are much more humans than bots in the dataset.

![image](https://user-images.githubusercontent.com/33216106/186072493-6ad5d911-0321-40a3-9a15-7cbaed9283c4.png)

| Class | Value |
| ------------- |:-------------:|
| 0.0 | 1910 |
| 1.0 | 103 |

Oversampling using Synthetic Minority Oversampling Technique (SMOTE) will be implemented to trat this class imbalance issue.

## **Feature Engineering**

Feature Engineering was a major block in this capstone project. The current features in the dataset were used to create new variables to enhance the performance of the model. The new features generated are:

| Feature(s) | Description |
| ------------- |:-------------:|
| Adding more feature columns with `.nunique()`| i. Number of bids made per bidder_id <br> ii. Number of auctions made per bidder_id <br> iii. Number of merchandise made per bidder_id <br> iv. Number of device made per bidder_id <br> v. Number of countries originated per bidder_id <br> vi. Number of ip addresses used per bidder_id <br> vii. Number of url addresses used per bidder_id |
| Analyze Auction Data | i. Mean/Max of devices per auction for each bidder_id <br> ii. Mean/Max of IP address per auction for each bidder_id <br> iii. Mean/Max of URL per auction for each bidder_id <br> iv. Mean/Max of time per auction for each bidder_id <br> |
| Mean, std and max bids per auction by each bidder_id | Since robots are designed to win the auction, there might be a lot of bids made the robots. Therefore, we can extract information such as average, standard deviation and max number of bids made by each bidder_id. |
| Calculate maximum number of bids within devices in auction | Bots have high possibility to change devices when bidding. Therefore, it might be useful to investigate maximum number of bids within devices in auction for each bidder_id. |
| Calculate maximum number of bids within ip addresses in auction | Bots have high possibility to change IP address faster than humans. Therefore, it might be useful to investigate maximum number of bids within ip addresses in auction for each bidder_id. |
| Calculate maximum number of bids within url in auction | Bots have high possibility to change url faster than humans. Therefore, it might be useful to investigate maximum number of bids within url in auction for each bidder_id. |
| Calculate maximum number of bids within country in auction | Bidders might hire multiple bots from different countries to win the auction. Therefore, it might be useful to investigate maximum number of bids within country in auction for each bidder_id. |
| Calculate bidding time difference per user | Through calculation of time difference between every 2 bids made by each bidder, we can determine if each bidder is fast or slow making bids. However, there is no difference in the bidder's 1st bid, which will result in missing value in that row. Since these missing values is insignificant, we can drop them. |
| Maximum number of auctions a bidder has participated in one hour  | Using previous time difference data, we can investigate if any bidders have time_difference of 0 (i.e. bidding very immediately). |
| Maximum number of bids in a 30 min span | Bots have the capability to bid faster than humans within a short time span - they might have more bids than humans. Therefore, it might be useful to investigate maximum number of bids made by a bidder_id within 30 minutes time span. |
| Maximum number of bids in a 20 min span | Bots have the capability to bid faster than humans within a short time span - they might have more bids than humans. Therefore, it might be useful to investigate maximum number of bids made by a bidder_id within 20 minutes time span. |
| Maximum number of bids in a 10 min span | Bots have the capability to bid faster than humans within a short time span - they might have more bids than humans. Therefore, it might be useful to investigate maximum number of bids made by a bidder_id within 10 minutes time span. |
| Number of simultaneous bids | Bots might be employed to bid simultaneously in different auctions at the same time. Therefore, it might be useful to investigate max number of simultaneous bids made within 3 seconds. |
| The mean time of day for the bids of each bidder | It might be useful to investigate at which period of the day bidders typically bid. |
| Proportion of each bidder's bids in each day | It might be useful for the investigate the proportion of bids spent in the 3 days time period within the data. |
| Proportion of bids made during different periods of the day | It might be useful to bin the bidders' bids into different time periods of the day (i.e. Morning, Afternoon, Night). |

## **Model Training**

Using sklearn's `.Pipeline()` function, I have managed to simplfy the modelling process into a more efficient and digistable workflow. The steps within the Pipeline involves:
1. `SMOTE` function to treat imbalanced data.
2. Features are mainly numerical in nature. Therefore, `StandardScaler()` function to standardize features by removing the mean and scaling to unit variance. 
3. Adding `ExtraTreesClassifier`, `LGBMClassifier`, `XGBClassifier`, `CatBoostClassifier`, `GradientBoostingClassifier`, `HistGradientBoostingClassifier`, `RandomForestClassifier` seperately for Model Training.

## **Model Evaluation**

Submissions for this competition are judged on area under the ROC curve. Using roc_auc as the main metrics to evaluate the model after Model Training, it was found that ExtraTreesClassifier have performed the best. Below is a summary of the performance ranked by AUC:

| Model| roc_auc |
| ------------- |:-------------:|
| ExtraTreesClassifier | 0.925151 |
| RandomForestClassifier | 0.912092 |
| LGBMClassifier | 0.910364 |
| XGBClassifier | 0.910189 |
| HistGradientBoostingClassifier | 0.901195 |
| CatBoostClassifier | 0.897957 |
| GradientBoostingClassifier | 0.884073 |

## **Hyperparameter Tuning**

To improve the performance of the ExtraTreesClassifier, my group has performed Hyperparameter Tuning using the `.RandomGridSearch()` function. Through research, my group have found good parameters to tune for ExtraTrees:

{'n_estimators': [120, 300, 500, 800, 1200], <br>
 'max_depth': [5, 8, 15, 25, 30, None], <br>
 'min_samples_split' : [1, 2, 5, 10, 15, 100], <br>
 'min_samples_leaf': [1, 2, 5, 10], <br>
 'max_features': ['log2', 'sqrt', None]} <br>

## **Feature Importance**

It is important to understand which specific feature(s) have a larger effect on the model that is being used to predict a certain variable. The top 20 features of the model are:

![image](https://user-images.githubusercontent.com/33216106/186070435-15463b9a-6383-4bd0-b813-b2fc15cd66bd.png)

Time series features such as simultaneous bids within a specific time period can be seen as the main contributors to prediction of bots.
