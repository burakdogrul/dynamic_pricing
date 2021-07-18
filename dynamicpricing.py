#Importing required libraries

import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import statsmodels.stats.api as sms
from functools import reduce

#Pandas set option

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#Read dataframe and copy

dataframe_ = pd.read_csv("pricing.csv", sep=";")
df = dataframe_.copy()


#Function for getting basic information

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe().T)

check_df(df)

#Unique category id

df["category_id"].nunique()


#Category id value counts

df["category_id"].value_counts()

#Price: minimum, mean, median, maximum and standard deviation information for category id

df.groupby("category_id").agg({"price":["min","mean","median","max","std"]})

#Need more information price distribution

df["price"].quantile([0,0.01,0.10,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,0.95,0.96,0.97,0.98,0.99,1])

#Function for removing outliers

def remove_outliers(dataframe, variable, low=0.25, up=0.75):
    quartile1 = dataframe[variable].quantile(low)
    quartile3 = dataframe[variable].quantile(up)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    df_remove = dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)]
    print("Warning!!! There are {} deleted outliers".format(df_remove.shape[0]))
    return dataframe[~dataframe.index.isin(df_remove.index)]

#df2 --> deleted outliers dataframe

df2 = remove_outliers(df, "price")

#Checking basic information for df2

check_df(df2)

#A/B test for each double unique item

def test_AB(dataframe, group, target, alpha):

    """
    The function gives A/B test results.
    The function compares between the groups by the desired alpha value according to the target variable.

    A/B Test Assumptions:
    1-Normality
    2-Homogeneity

    Steps:

    Step-1: Prepare the data for A/B testing

    Step-2: Define hypothesis

    H0: There is no difference between group averages
    H1: There is a difference between the group averages

    if p-value < alpha HO Reject (There is no statistically significant difference between the means),
    otherwise Fail to Reject H0 (There is a statistically significant difference between the means)

    Step-3: Check Normality
    H0: Distribution is normal
    H1: Distribution is not normal

    if p-value < alpha HO Reject (The assumption of normal distribution is not provided.),
    otherwise Fail to Reject H0 (The assumption of normal distribution is provided.)

    Step-4: Check Homogeneity
    H0: Variances are equal
    H1: variances are not equal

    if p-value < alpha HO Reject (The assumption of variance homogeneity is not provided.),
    otherwise Fail to Reject H0 (The assumption of variance homogeneity is provided.)

    Step-5: Decide which test to do

    if distribution is normal and variances are equal, apply independent t test
    if distribution is normal but variances are not equal, apply welch test (hint: independet t test, equal_var=False)
    if at least one of the pairs is not normally distributed, apply mann whitney-u test.

    Step-6: Combine the results

    Parameters
    ----------
    dataframe: dataframe
            Variable names are the data to be retrieved.

    group: str
            Variable name we want to group

    target: str
            Target variable. Values that we want to measure the difference

    alpha: float
            Threshold value that we measure p-values against.
            Alpha = 1-Confidence Level

    Returns
    -------
    dataframe
    (unique item set, test type, hypothesis result, p value, first item mean, second item mean, mean difference)

    Examples:
    test_AB(df, "category_id", "price", 0.05)

    """

    #Step-1

    temp_list = []
    for x in df[group].unique():
        for y in df[group].unique():
            temp_list.append([x, y])

    fset = set(frozenset(x) for x in temp_list)
    uniq_list = [list(x) for x in fset if len(x) > 1]

    final_df = pd.DataFrame()

    #Step-2

    #H0: There is no difference between group averages
    #H1: There is a difference between the group averages

    for i in range(0, len(uniq_list)):
        group_A = dataframe[dataframe[group] == uniq_list[i][0]][target]
        group_B = dataframe[dataframe[group] == uniq_list[i][1]][target]

    #Step-3

        normal_A = shapiro(group_A)[1] < alpha
        normal_B = shapiro(group_B)[1] < alpha

    #Step-4

        if (normal_A == False) & (normal_B == False):
            levene_ = levene(group_A, group_B)[1] < alpha
    #Step-5

            if levene_ == False:
                p_value = ttest_ind(group_A, group_B, equal_var=True)[1]
            else:
                p_value = ttest_ind(group_A, group_B, equal_var=False)[1]
        else:
            p_value = mannwhitneyu(group_A, group_B)[1]

    #Step-6

        temp_df = pd.DataFrame({"Hypothesis": [p_value < alpha],"p_value": p_value,
                             "Group_A_Mean": [group_A.mean()], "Group_B_Mean": [group_B.mean()],
                             "Mean_Difference":[abs(group_A.mean()-group_B.mean())]}, index = [set(uniq_list[i])])

        temp_df["Hypothesis"] = np.where(temp_df["Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
        temp_df["Test"] = np.where((normal_A == False) & (normal_B == False), "Parametric", "Non-Parametric")
        final_df = pd.concat([final_df, temp_df[["Test", "Hypothesis", "p_value","Group_A_Mean","Group_B_Mean","Mean_Difference"]]])
    return final_df

#A/B test for include outliers dataframe

tested_with_outliers = test_AB(df, "category_id", "price", 0.05)
tested_with_outliers.sort_values(by="Hypothesis")

#Item sets that there is a difference between means (for df)

tested_with_outliers.reset_index(inplace=True)
tested_with_outliers.loc[tested_with_outliers["Hypothesis"] == "Reject H0", "index"]

#Item sets that there is no difference between means (for df)

tested_with_outliers.loc[tested_with_outliers["Hypothesis"] == "Fail to Reject H0", "index"]

#A/B test for dataframe without outliers

tested_without_outliers = test_AB(df2, "category_id", "price", 0.05)
tested_without_outliers.sort_values(by="Hypothesis")


#Item sets that there is a difference between means (for df2)

tested_without_outliers.reset_index(inplace=True)
tested_without_outliers.loc[tested_without_outliers["Hypothesis"] == "Reject H0", "index"]

#Item sets that there is no difference between means (for df2)

tested_without_outliers.loc[tested_without_outliers["Hypothesis"] == "Fail to Reject H0", "index"]

#tested_without_outliers  *** Fail to Reject H0 so there is no difference between means ***
#     {675201, 361254}
#     {874521, 675201}
#     {675201, 201436}
#     {874521, 201436}
#     {201436, 361254}
#     {874521, 361254}

# Unique category id

similar = [201436,361254,675201,874521]

#tested_without_outliers *** Reject H0 so there is a difference between means **
#     {326584, 361254}
#     {326584, 201436}
#     {201436, 489756}
#     {874521, 489756}
#     {326584, 489756}
#     {326584, 675201}
#     {675201, 489756}
#     {326584, 874521}
#     {489756, 361254}

#Unique category id except similar ids

not_similar = [326584,489756]

#All unique category id
all_list = [201436,361254,675201,874521,326584,489756]

#Item price determination function
#for constant price: if similar use mean of groups mean, if not similar use mean of groups median
#for flexible price: if similar, not similar and all use low and high value for 95% confidence, mean of groups mean, mean of groups median
#all price for just observing

def item_price(dataframe,type="all",con_price=True):
    sum = 0
    price = []
    if con_price:
        if type == "similar":
            for i in similar:
                sum += dataframe.loc[dataframe["category_id"] == i, "price"].mean()
            price = sum/len(similar)
            return price
        elif type == "not_similar":
            for i in not_similar:
                sum += dataframe.loc[dataframe["category_id"] == i, "price"].median()
            price = sum/len(not_similar)
            return price
        elif type == "all":
            for i in similar:
                sum += dataframe.loc[dataframe["category_id"] == i, "price"].mean()
            for j in not_similar:
                sum += dataframe.loc[dataframe["category_id"] == j, "price"].median()
            price = sum/len(all_list)
            return np.mean(price)
    else:
        if type == "similar":
            for i in similar:
                for j in dataframe.loc[dataframe["category_id"] == i, "price"]:
                    price.append(j)
            range = sms.DescrStatsW(price).tconfint_mean()
            mean = np.mean(price)
            median = np.median(price)
            print(f"Price mean range with 95% confidence between {'%.4f' % range[0]} and {'%.4f' % range[1]} \nMean: {'%.4f' % mean} \nMedian: {'%.4f' % median}")

        elif type == "not_similar":
            for i in not_similar:
                for j in dataframe.loc[dataframe["category_id"] == i, "price"]:
                    price.append(j)
            range = sms.DescrStatsW(price).tconfint_mean()
            mean = np.mean(price)
            median = np.median(price)
            print(f"Price mean range with 95% confidence between {'%.4f' % range[0]} and {'%.4f' % range[1]} \nMean: {'%.4f' % mean} \nMedian: {'%.4f' % median}")

        elif type == "all":
            for i in all_list:
                for j in dataframe.loc[dataframe["category_id"] == i, "price"]:
                    price.append(j)
            range = sms.DescrStatsW(price).tconfint_mean()
            mean = np.mean(price)
            median = np.median(price)
            print(f"Price mean range with 95% confidence between {'%.4f' % range[0]} and {'%.4f' % range[1]} \nMean: {'%.4f' % mean} \nMedian: {'%.4f' % median}")

        return [range, mean, median]

#constant price for all unique category id

all_con = [item_price(df2,type="all",con_price=True)]
all_con

#constant price for all unique similar category id

similar_con = [item_price(df2,type="similar",con_price=True)]
similar_con

#constant price for all unique not similar category id

not_similar_con = [item_price(df2,type="not_similar",con_price=True)]
not_similar_con


#flexible price for all unique category id

all_flex =item_price(df2,type="all",con_price=False)
all_flex

#flexible price for all unique similar category id

similar_flex = item_price(df2,type="similar",con_price=False)
similar_flex

#flexible price for all unique not similar category id

not_similar_flex = item_price(df2,type="not_similar",con_price=False)
not_similar_flex

#The function for capturing part from within categories

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

#The function for price simulation
# For constant price: to find income use constant price
# For flexible price: to find income use low confidence value, high confidence value, mean and median

def price_simulation(dataframe,data, method):
    if "con" in str(namestr(data,globals())):
        if method == "all":
            freq = len(dataframe[dataframe["price"] >= data[0]])
            income = freq * data[0]
            return income
    else:
        if method == "low_conf":
            freq = len(dataframe[dataframe["price"] >= data[0][0]])
            income = freq * data[0][0]
            return income

        if method == "high_conf":
            freq = len(dataframe[dataframe["price"] >= data[0][1]])
            income = freq * data[0][1]
            return income

        if method == "mean":
            freq = len(dataframe[dataframe["price"] >= data[1]])
            income = freq * data[1]
            return income

        if method == "median":
            freq = len(dataframe[dataframe["price"] >= data[2]])
            income = freq * data[2]
            return income

#There are 15 scenario: 3 for constant price, 12 for flexible price
#Use functions and convert to dataframe. Finally merge dataframes


all_con_price = pd.DataFrame(["all_data","constant","all",price_simulation(df2, all_con, "all")],index=["data","constant_flexible","method","Income"]).T
similar_con_price = pd.DataFrame(["similar_group","constant","all",price_simulation(df2, similar_con, "all")],index=["data","constant_flexible","method","Income"]).T
not_similar_con_price = pd.DataFrame(["not_similar_group","constant","all",price_simulation(df2, not_similar_con, "all")],index=["data","constant_flexible","method","Income"]).T

all_flex_low_conf = pd.DataFrame(["all_data","flexible","low_conf",price_simulation(df2, all_flex, "low_conf")],index=["data","constant_flexible","method","Income"]).T
all_flex_high_conf = pd.DataFrame(["all_data","flexible","high_conf",price_simulation(df2, all_flex, "high_conf")],index=["data","constant_flexible","method","Income"]).T
all_flex_mean = pd.DataFrame(["all_data","flexible","mean",price_simulation(df2, all_flex, "mean")],index=["data","constant_flexible","method","Income"]).T
all_flex_median= pd.DataFrame(["all_data","flexible","median",price_simulation(df2, all_flex, "median")],index=["data","constant_flexible","method","Income"]).T

similar_flex_low_conf = pd.DataFrame(["similar_group","flexible","low_conf",price_simulation(df2, similar_flex, "low_conf")] ,index=["data","constant_flexible","method","Income"]).T
similar_flex_high_conf = pd.DataFrame(["similar_group","flexible","high_conf",price_simulation(df2, similar_flex, "high_conf")] ,index=["data","constant_flexible","method","Income"]).T
similar_flex_mean = pd.DataFrame(["similar_group","flexible","mean",price_simulation(df2, similar_flex, "mean")] ,index=["data","constant_flexible","method","Income"]).T
similar_flex_median = pd.DataFrame(["similar_group","flexible","median",price_simulation(df2, similar_flex, "median")] ,index=["data","constant_flexible","method","Income"]).T

not_similar_flex_low_conf = pd.DataFrame(["not_similar_group","flexible","low_conf",price_simulation(df2, not_similar_flex, "low_conf")],index=["data","constant_flexible","method","Income"]).T
not_similar_flex_high_conf = pd.DataFrame(["not_similar_group","flexible","high_conf",price_simulation(df2, not_similar_flex, "high_conf")],index=["data","constant_flexible","method","Income"]).T
not_similar_flex_mean = pd.DataFrame(["not_similar_group","flexible","mean",price_simulation(df2, not_similar_flex, "mean")],index=["data","constant_flexible","method","Income"]).T
not_similar_flex_median= pd.DataFrame(["not_similar_group","flexible","median",price_simulation(df2, not_similar_flex, "median")],index=["data","constant_flexible","method","Income"]).T

merging = [all_con_price,similar_con_price,not_similar_con_price,all_flex_low_conf,all_flex_high_conf,all_flex_mean,all_flex_median,similar_flex_low_conf,similar_flex_high_conf,similar_flex_mean,similar_flex_median,not_similar_flex_low_conf,not_similar_flex_high_conf,not_similar_flex_mean,not_similar_flex_median]


df_merged = reduce(lambda  left,right: pd.merge(left,right, how='outer'), merging)
df_merged.sort_values(by="Income", ascending=False)

#Not similar group expected income table
#Fixed price should be used for non similar groups

df_merged[(df_merged["data"]=="not_similar_group") ].sort_values(by="Income", ascending=False)

#Similar group expected income table
#The median method provides the highest income for similar group but the low confidence method is more secure and provides the 2nd highest income.

df_merged[(df_merged["data"]=="similar_group")].sort_values(by="Income", ascending=False)

#Just for observing

df_merged[(df_merged["data"]=="all_data") ].sort_values(by="Income", ascending=False)

#best_scenario
df_merged[(df_merged["data"]=="not_similar_group")]["Income"].max() + df_merged[(df_merged["data"]=="similar_group")]["Income"].max()

df2["price"].sum()

expected_max_income = df_merged[(df_merged["data"]=="not_similar_group")]["Income"].max() + df_merged[(df_merged["data"]=="similar_group")]["Income"].max()
real_income = df2["price"].sum()
profit= expected_max_income-real_income
profitperc = ((expected_max_income-real_income)/real_income)
print(f"If we use constant price method for non similar group and flexieble median price method for similar group\nexpected profit will be {'%.3f' %profit} unit\nexpected profit percentage {'%.3f' %(profitperc*100)} % ")

#2nd Best Scenario

df_merged[(df_merged["data"]=="not_similar_group")]["Income"].max() + df_merged[(df_merged["data"]=="similar_group") & (df_merged["method"]=="low_conf")]["Income"].values[0]

expected_max_income = df_merged[(df_merged["data"]=="not_similar_group")]["Income"].max() + df_merged[(df_merged["data"]=="similar_group") & (df_merged["method"]=="low_conf")]["Income"].values[0]
real_income = df2["price"].sum()
profit= expected_max_income-real_income
profitperc = ((expected_max_income-real_income)/real_income)
print(f"If we use constant price method for non similar groups and flexieble median price method for similar groups\nexpected profit will be {'%.3f' %profit} unit\nexpected profit percentage {'%.3f' %(profitperc*100)} % ")