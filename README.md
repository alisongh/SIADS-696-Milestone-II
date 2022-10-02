# SIADS 696 Milestone II – Project Proposal
*Choi Teng Chao, Liwen (Alison) Huang, Min Lu*

## Post-COVID Wake County, NC Single Family Home Value Prediction
### Project Overview
COVID drives the work-from-home new norm across the US. People have begun to realize that they need extra spaces in their homes for home offices in order to work effectively, especially for working parents who need to keep their kids away while working. Such a phenomenon could have driven new housing demand in the real estate market. Since people are no longer required to go to the office during the pandemic, it is rational to move from expensive apartments near business centers to cheaper suburban areas. Such rationale could also bring changes to the demand pattern throughout the market. 

This project will create a supervised learning model to predict the value of single-family homes in Wake County. We plan to train models incorporating COVID-related data, economic indicators, and Wake County’s real estate sales data. We are particularly interested in discovering whether COVID-related features interact with other features in the dataset and/or if the sensitivity to such features has changed after COVID. We also plan to leverage unsupervised learning methods to gain a deeper understanding of Wake County’s housing market segmentation.

#### Why Wake County, North Carolina?
You might ask why we picked Wake County, NC, as our target region for the project. According to the US Census Bureau’s report, 112,000 people are moving into North Carolina from 2020 to 2021. North Carolina remained the 9th largest state in the United States, with a 10.6 million population until 2021. Moreover, only 7,000 people moved into North Carolina from overseas during the pandemic period. The top 3 states most people move out of are California, Texas, and New York. Based on the Move.org survey, 45% said it was because of a lower cost of living when it came to the factors contributing to people moving. Nowadays, moving across the country is easier in the US due to the remote work policy. The housing market has become popular, and house prices keep increasing in the mover-attractive cities, Raleigh and Durham, in Wake County, NC, during the post-pandemic period. In the Triangle area, Raleigh-Durham-Chapel Hill, the house price has increased about 28% compared to a year ago. The above truth and numbers are astonishing, and we want to know what variables impact the housing market in Wake County, NC.

### Datasets
We will use the following data source to create our dataset:
| Dataset | Region & Period | Format | Source | Remark |
| :----:  | :----:          | :----: | :----: | :---- |
|House sales data| Wake, NC 2011-2012| CSV | [link](https://www.wakegov.com/departments-government/tax-administration/data-files-statistics-and-reports/residential-sales-zip-code)|Transactional housing data.|
|COVID-19 Confirmed Data|The US 2020-2022 | CSV |[link](https://covid19datahub.io/articles/data.html) | Obtain state and city-level COVID cases and other COVID-related indicators, such as stay-home restrictions, gathering restrictions, workplace closing, etc.|
|Crime data |Wake, NC 2000-2020 | CSV |[link](https://crime-data-explorer.fr.cloud.gov/pages/explorer/crime/crime-trend) | City police incident report by the FBI's Crime Data Explorer (CDE). City-level and Wake county sheriff datasets only. When a case is marked as “cleared”, it means the case is either cleared by arrest or by exceptional means. The law enforcement agencies must meet the following criteria to clear a case by exceptional means:<br><li>Identified the offender. <br><li>Gathered enough evidence to support an arrest, make a charge, and turn over the offender to the court for prosecution. <br><li>Identified the offender’s exact location so that the suspect could be taken into custody immediately. <br><li>Encountered a circumstance outside the control of law enforcement that prohibits the agency from arresting, charging, and prosecuting the offender|
|30-Year Fixed Rate Mortgage |Worldwide 1960-2022 | CSV |[link](https://fred.stlouisfed.org/series/MORTGAGE30US)|30-Year Fixed Rate Mortgage data in the U.S. |
|Iron and Steel Commodity Producer Price index|The US 1926-2022 | CSV |[link](https://fred.stlouisfed.org/series/WPU101) |According to a White House report, the COVID effect on the global supply chain significantly due to multiple national lockdowns from 2020 to 2021. It causes the prices of raw materials, which further impacts the construction cost of new houses. |
|Lumber Commodity Producer Price index|The US 1926-2022| CSV |[link](https://fred.stlouisfed.org/series/WPU081)|Same as Iron and Steel price|
|Building Material and Supplies Dealers Producer Price Index|The US 2003-2022|CSV|[link](https://fred.stlouisfed.org/series/PCU44414441)|Same as Iron and Steel price|
|Unemployment Rate|Wake, NC 2000-2022|CSV|[link-1](https://ycharts.com/indicators/wisconsin_unemployment_rate)<br>[link-2](https://jobcenterofwisconsin.com/wisconomy/query#laus_dl)|Link-1 is the state-level data<br>Link-2 is the county/city-level data|
|Interest Rates|The US 2000-2022|CSV|[link](https://databank.worldbank.org/source/world-development-indicators)|U.S. yearly interest rate data since 1960 |
|Residential House Supply and Demand Data|The US 1998-2022|EXCEL|[link](https://www.huduser.gov/portal/ushmc/hd_home_sales.html)|The total number of new homes sold and new home construction in the U.S from 1978 to 2022.|
|Residential House Supply and Demand Data|Wake, NC 2017-2022|CSV|[link-1](https://fred.stlouisfed.org/series/DESCCOUNTY37183)<br>[link-2](https://fred.stlouisfed.org/series/SUSCCOUNTY37183)|The total number of new homes sold and new home construction in Wake County from 2017 to 2022.|
|Population Growth Rate|Wake, NC 2000-2022|CSV|[link](https://www.census.gov/data/tables/2000/demo/popproj/2000-national-summary-tables.html)|Population growth rate in the U.S. for three decades.|

## Unsupervised Learning
* Our goal is to partition the data into clusters. If applicable, we would consider leveraging these label clusters as features to feed into the supervised learning model for prediction.
* **K-means clustering, Hierarchical clustering, and DBSCAN clustering** will be used. For the example of K-means, we will need to find out the value of K (aka centroid) by randomly assigning K for the initial cluster and then iterate it until there is no change in cluster centroids.
* Evaluation metrics include the Silhouette score, the elbow method, and the ratio of total within-group variance to between-group variance vs # of clusters.
* We will use the horizontal bar chart and line plot to display evaluation metrics.
## Supervised Learning 
* Our supervised learning model will assign the real estate value as outcome Y and the other features as input X’s.
* We will engineer a feature `isCOVID` to flag the dates under the COVID period as 1 and 0 for dates before COVID. We will feature-engineer other applicable features as we compile the dataset.
* We plan to apply these models: **Random Forest, Decision Tree, and Regression**, to decide which one to use as a final model for prediction based on their evaluation results. We will validate the model using cross-validation and, where applicable, use GridSearchCV from the scikit-learn library to tune the hyperparameters.
* Evaluation metrics include - F1 scores, precision-recall, ROC curve AUC, information loss, and use of dummy regressors as a baseline comparison. LIME will also be applied to explain the predicted Y for selected instances.
* Visualization - scatter plot for observing the data, pair grid to examine the data distribution, bar chart, and line chart to compare different models' accuracy.
## Project Timeline 
Sep 3 - Team formed and Initial meeting

Sep 9 - Topic chose

Sep 26 - Datasets collection

Sep 26 - Revised Proposal 

Oct 3 - Done pre-processing and EDA

Oct 4 - First stand up

Oct 8 - Second stand up

Oct 15 - Finalize modeling

Oct 22 - Finish report

Oct 25 - Report submission

## Team Member Roles and Responsibilities
We decided to assign each team member to head up specific tasks of the project. We will collectively review all the work as a team. 
* Liwen (Alison) Huang - data manipulation & feature engineering, EDA, report writing
* Min Lu - supervised modeling, visualization, report writing
* Choi Teng Chao - unsupervised modeling, visualization, report writing
## Technology Used
We will use GitHub with ZenHub add-on as the primary collaborative tool for project management, discussion, and code reviews. We will hold weekly meetings to align progress and take an agile project management approach such that we can adapt to changes based on our discoveries while maintaining our project scope.
## Limitation
With the current data, we have limited variables for house description, such as bedroom numbers, which might be one of the most critical factors influencing house prices. We will use bathroom numbers instead. Moreover, we only have data for Wake County instead of the whole county, so our model might be more overfitted to Wake County.

Some current data contain incomplete data, such as missing historical or the latest data. This might cause selection bias and error in machine learning modeling. Regarding Wake County crime rate datasets resources and collection, even though the CDE provides RESTful APIs, it does not allow annual retrieval of all crime data by city or zip code. We have to download them manually and respectfully, which is inefficient.
## References
Roberts, J. (2022, August 8). *Moving trends in 2021: Moving Industry Stats & Data.* Move.org. Retrieved September 25, 2022, from https://www.move.org/moving-stats-facts/. 

U.S. Census Bureau (2022). *Population in North Carolina.* Retrieved from https://data.census.gov/cedsci/all?q=population%20in%20North%20Carolina.
