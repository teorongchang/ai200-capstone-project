# AI200 - Human or Robot?

## My top solution for [AI200 Mar 2022: Human or Robot? Capstone Competition](https://www.kaggle.com/competitions/ai200-mar-2022-human-or-bot) 

###
Adapted from [Facebook Recruiting competition](https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/), the purpose of this competition is to chase down robots for an online auction site. Human bidders on the site are becoming increasingly frustrated with their inability to win auctions vs. their software-controlled counterparts. As a result, usage from the site's core customer base is plummeting.

In order to rebuild customer happiness, the site owners need to eliminate computer generated bidding from their auctions. Their attempt at building a model to identify these bids using behavioral data, including bid frequency over short periods of time, has proven insufficient. 

The goal of this competition is to identify online auction bids that are placed by "robots", helping the site owners easily flag these users for removal from their site to prevent unfair auction activity.

####
**Data Description**

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

####
**Exploratory Data Analysis (EDA)**

I have mainly used SweetViz and Pandas Profiling to perform EDA on the datasets to investigate on Statistics and Outliers in the dataset.

#### 
**Feature Engineering**
