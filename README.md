# Clustering-and-Segmentation-for-Banking
## Project Title: Exploratory Analysis and Clustering Techniques for Customer Segmentation in Banking
### PROJECT OVERVIEW

- In this project scenario, I am envisioning the role of a data scientist employed by a bank, presented with comprehensive data pertaining to the bank's customers over the preceding six months.
- This dataset encompasses information such as transaction frequency, amounts, tenure, among other relevant details.
- The objective set forth by the bank's marketing team is to harness the power of AI/ML to initiate a ```targeted advertising campaign``` tailored specifically to distinct customer groups.
- The success of this campaign hinges on effectively categorizing customers into a ```minimum of three distinct groups```, a practice commonly referred to as ```marketing segmentation```.
- This segmentation process is pivotal for ```optimizing the conversion rates of marketing campaigns```.

### MODULES OF THE PROJECT

1. Exploratory Data Analysis
2. Data Visualization
3. Feature Engineering
4. Feature Selection (Lasso CV Feature Importances)
5. Clustering (Hierarchial Clustering)
6. Principal Component Analysis (PCA)

### DATASET DESCRIPTION

The dataset, sourced from Kaggle [here](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata), provides insights into the usage behavior of approximately 9000 active credit card holders over the past six months. Organized at a customer level, the dataset encompasses 18 behavioral variables that capture diverse aspects of credit card utilization.

#### DATA DICTIONARY:
1. CUSTID: Identification of Credit Card holder
2. BALANCE: Balance amount left in customer's account to make purchases
3. BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
4. PURCHASES: Amount of purchases made from account
5. ONEOFFPURCHASES: Maximum purchase amount done in one-go
6. INSTALLMENTS_PURCHASES: Amount of purchase done in installment
7. CASH_ADVANCE: Cash in advance given by the user
8. PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
9. ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
10. PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
11. CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid
12. CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"
13. PURCHASES_TRX: Number of purchase transactions made
14. CREDIT_LIMIT: Limit of Credit Card for user
15. PAYMENTS: Amount of Payment done by user
16. MINIMUM_PAYMENTS: Minimum amount of payments made by user
17. PRC_FULL_PAYMENT: Percent of full payment paid by user
18. TENURE: Tenure of credit card service for user

### 1. DATA ANALYSIS
I'll commence by addressing the dataset's cleanliness. This involves identifying and managing null values, addressing outliers, and ensuring the consistency of the data.

#### A) Describing the Data
```creditcard_df.describe().T```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/d520b588-79b6-4be2-a196-ba475f8517f8)

*Insights*
- Mean balance is $1564
- Balance frequency is frequently updated on average ~0.9
- Purchases average is $1000
- one off purchase average is ~$600
- Average purchases frequency is around 0.5
- Average ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY, and CASH_ADVANCE_FREQUENCY are generally low
- Average credit limit ~ 4500
- Percent of full payment is 15%
- Average tenure is 11 years

#### B) Checking for Missing Values
```
# Plotting missing values
plt.figure(figsize=(10, 5))
sns.barplot(x=creditcard_df.columns, y=creditcard_df.isnull().sum(), palette='Blues')
plt.xticks(rotation=45, ha='right')
plt.title('Missing Data Visualization')
plt.show()
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/37794100-8d7b-48d1-9eb6-55b324f97efa) <br>
<br>
So, we are having Missing values in Minimum Payment Attribute. Hence, I decide to impute with KNN Imputer values where each sample’s missing values are imputed using the mean value from n_neighbors nearest neighbors found in the training set.


#### C) Checking for Outliers
Using Inter-Quartile Range (IQR), following the below approach to find outliers: <br>
Calculate the first and third quartile (Q1 and Q3).<br>
Further, evaluate the interquartile range, IQR = Q3-Q1. <br>
Estimate the lower bound, the lower bound = Q1*1.5 <br>
Estimate the upper bound, upper bound = Q3*1.5 <br>
The data points that lie outside of the lower and the upper bound are outliers. <br>
```
def outlier_percent(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    minimum = Q1 - (1.5 * IQR)
    maximum = Q3 + (1.5 * IQR)
    num_outliers =  np.sum((data < minimum) |(data > maximum))
    num_total = data.count()
    return (num_outliers/num_total)*100
```
```
non_categorical_data = creditcard_df.drop(['CUST_ID'], axis=1)
for column in non_categorical_data.columns:
    data = non_categorical_data[column]
    percent = round(outlier_percent(data), 2)
    print(f'Outliers in "{column}": {percent}%')
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/a3af3db6-790b-4d2c-8246-64259728f207)

#### D) Imputing the Missing Values and Outliers - KNN Imputer
First I set all outliers as NaN, so it will be taken care of in the next stage, where I impute the missing values.
```
# imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer()
imp_data = pd.DataFrame(imputer.fit_transform(non_categorical_data), columns=non_categorical_data.columns)
imp_data.isna().sum()
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/49cb716d-b001-4aad-ac17-3a9bb4d03efd)

### 2. DATA VISUALIZATION
#### A) Displot
```
plt.figure(figsize=(20,50))
for i in range(len(creditcard_df.columns)):
    plt.subplot(17, 1, i+1)
    displot = sns.distplot(creditcard_df[creditcard_df.columns[i]], kde_kws={"color": "b", "lw": 3, "label": "KDE"}, hist_kws={"color": "g"})
    plt.title(creditcard_df.columns[i])

displot.get_figure().savefig("Images/Distplot.png")
plt.tight_layout()
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/b5757065-e817-4f4f-9f88-f396f117cc0e)

_Insights_
- Mean of balance is 1500 dollors
- 'Balance_Frequency' for most customers is updated frequently ~1
- For 'PURCHASES_FREQUENCY', there are two distinct group of customers
- For 'ONEOFF_PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY' most users don't do one off puchases or installment purchases frequently
- Very small number of customers pay their balance in full 'PRC_FULL_PAYMENT'~0
- Credit limit average is around $4500
- Most customers are ~11 years tenure

#### B) Heatmap (Correlation Analysis)
```
correlations = creditcard_df.corr()
f, ax = plt.subplots(figsize = (20, 8))
heatmap = sns.heatmap(correlations, annot = True)
plt.show()
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/1e2d7ccc-8cbc-475c-9378-e02db9a1f985)

_Insights_
- 'PURCHASES' have high correlation between one-off purchases, 'installment purchases, purchase transactions, credit limit and payments.
- Strong Positive Correlation between 'PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY'

### 3. FEATURE ENGINEERING
```
creditcard_df["new_BALANCE_BALANCE_FREQUENCY"] = creditcard_df["BALANCE"] * creditcard_df["BALANCE_FREQUENCY"]
creditcard_df["new_ONEOFF_PURCHASES_PURCHASES"] = creditcard_df["ONEOFF_PURCHASES"] / creditcard_df["PURCHASES"]
creditcard_df["new_INSTALLMENTS_PURCHASES_PURCHASES"] = creditcard_df["INSTALLMENTS_PURCHASES"] / creditcard_df["PURCHASES"]
creditcard_df["new_CASH_ADVANCE_PURCHASES_PURCHASES"] = creditcard_df["CASH_ADVANCE"] * creditcard_df["CASH_ADVANCE_FREQUENCY"]
creditcard_df["new_PURCHASES_PURCHASES_FREQUENCY"] = creditcard_df["PURCHASES"] * creditcard_df["PURCHASES_FREQUENCY"]
creditcard_df["new_PURCHASES_ONEOFF_PURCHASES_FREQUENCY"] = creditcard_df["PURCHASES"] * creditcard_df["ONEOFF_PURCHASES_FREQUENCY"]
creditcard_df["new_PURCHASES_PURCHASES_TRX"] = creditcard_df["PURCHASES"] / creditcard_df["PURCHASES_TRX"]
creditcard_df["new_CASH_ADVANCE_CASH_ADVANCE_TRX"] = creditcard_df["CASH_ADVANCE"] / creditcard_df["CASH_ADVANCE_TRX"]
creditcard_df["new_BALANCE_CREDIT_LIMIT"] = creditcard_df["BALANCE"] / creditcard_df["CREDIT_LIMIT"]
creditcard_df["new_PAYMENTS_CREDIT_LIMIT"] = creditcard_df["PAYMENTS"] / creditcard_df["MINIMUM_PAYMENTS"]
```
```
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
for col in creditcard_df.columns:
    replace_with_thresholds(creditcard_df, col)
```
```
plt.figure(figsize=(10,5))
sns.boxplot(data=creditcard_df)
plt.xticks(rotation=90)
plt.show()
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/20dacaad-ccc9-446e-b13b-24708fa4f6f1)

### 4. FEATURE SELECTION
#### Lasso CV Feature Importances
```
X = data_scaled.drop(["BALANCE","new_BALANCE_BALANCE_FREQUENCY", "new_BALANCE_CREDIT_LIMIT", "BALANCE_FREQUENCY"],1)   #Feature Matrix
y = data_scaled["BALANCE"]          #Target Variable

reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/70be74f5-66a4-4b58-873b-5715117c0de8)
```
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  
      str(sum(coef == 0)) + " variables")
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/21890343-4c3f-4a89-b267-39149902577c)
```
imp_coef = coef.sort_values()
lasso_FE = imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/88238265-9219-4f0e-886b-2e64adaa714b)

### 5. CLUSTERING TECHNIQUES
#### Hierarchial Clustering 
Hierarchical clustering (also called hierarchical cluster analysis or HCA) is a method of cluster analysis which seeks to build a ```hierarchy of clusters```. Strategies for hierarchical clustering generally fall into two types:

- **Agglomerative:** This is a "bottom-up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
- **Divisive :** This is a "top-down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

In general, ```the merges and splits are determined in a greedy manner```. The results of hierarchical clustering are usually presented in a ```dendrogram```.
#### A) Dendogram
```
plt.figure(figsize =(6, 6)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/fe8312b4-3483-429b-96ad-b06d4f958726)

#### B) Silhouette Score
```
silhouette_scores = [] 

for n_cluster in range(2, 8):
    silhouette_scores.append( 
        silhouette_score(X_principal, 
                         AgglomerativeClustering(n_clusters = n_cluster).fit_predict(X_principal))) 
    
# Plotting a bar graph to compare the results 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show()
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/b763827b-b216-4428-b19d-836ac8468338)

_Insights_
Therefore, we see the optimal number of clusters for this particular dataset would be 3 or 4. Let us now build and visualize the clustering model for k =3.

#### C) Agglomerative Clustering
```
agg = AgglomerativeClustering(n_clusters = 3)
agg.fit(X_principal)
# Visualizing the clustering 
plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = AgglomerativeClustering(n_clusters = 3).fit_predict(X_principal), cmap =plt.cm.winter) 
plt.show()
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/038bc3ab-bcd2-4457-b4c7-955cb4dc1414)


### 6. PRINCIPAL COMPONENT ANALYSIS (PCA)
- PCA is an unsupervised ML Algorithm
- It performs dimensionality reductions while attempting at keeping the original information unchanged.
- It works by trying to find a new set of fearures called components.
- Components are composites of the uncorrelated given input features.

#### A) Obtaining the Principal Components
```
pca = PCA(n_components = 2)
principal_comp = pca.fit_transform(data_scaled)
principal_comp
```
#### B) Create a dataframe with the two components and concatenate the clusters labels to the dataframe
```
# Create a dataframe with the two components
pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])
pca_df.head()
# Concatenate the clusters labels to the dataframe
pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()
```
#### C) Visualizing the output
```
plt.figure(figsize=(20,8))
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink','yellow'])
plt.show()
```
![image](https://github.com/bsdr18/Clustering-and-Segmentation-for-Banking/assets/76464269/40424b16-2771-41ff-bb7b-3d67816aa9fb)
