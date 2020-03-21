"""
Alyssa Allums, Bill Li
This program reads in csv files containing information about each states
political bodies, economic status, and demographics. It uses this information
to create and save multiple plots to show the relationship between these
variables, and trains a model to predict a states percent change in GDP based
on demographic information. The mean squared of the train set and test set
is printed as output.
"""

import pandas as pd
import re
from collections import Counter
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# majority in 2 of the 3 branches
def calculate_party_majority(df):
    """
    Takes in the political indicator dataframe
    Determines the political majority for each state each year in the dataframe
    by determining the party with a majority in two out of three
    branches ("congress house", "congress sen", and "electoral")
    Returns the dataframe with a column "majority" that categorizes the state
    as having a political majority of "Democrat", "Republican", "Tie", or
    "Other"
    """
    party_majority = list()
    for index, row in df.iterrows():
        house = split_parties(row, "congress house")
        senate = split_parties(row, "congress sen")
        electoral = split_parties(row, "electoral")
        branch_party = list()
        branch_party.append(process_branch(house))
        branch_party.append(process_branch(senate))
        branch_party.append(process_branch(electoral))
        c = Counter(branch_party)
        # max will be one if there is a tie
        if max(c.values()) == 1:
            party_majority.append("Tie")
        else:
            party = max(c, key=c.get)
            if is_nan(party):
                party_majority.append(party)
            elif party == "D":
                party_majority.append("Democrat")
            elif party == "R":
                party_majority.append("Republican")
            else:
                party_majority.append("Other")
    df["majority"] = party_majority
    return df


def is_nan(x):
    """
    Takes in any value
    Returns True id the value is NaN, otherwise returns false
    """
    return isinstance(x, float) and math.isnan(x)


def split_parties(row, column):
    """
    Takes in the current row of the political indicator data and the column
    which specifies the political body
    If the row has a NaN value for the column, returns NaN
    Otherwsie, it splits the column value by commas and returns the resulting
    list
    """
    if is_nan(row[column]):
        return row[column]
    else:
        return row[column].split(",")


def process_branch(branch):
    """
    Takes in the list which specifies the count of each party in the given
    branch (as specified by split_parties)
    If the branch  is NaN, returns NaN
    Otherwise, returns a letter specifying which party has a majority in the
    given branch, calculated by the party which has the highest number
    of representatives.
    Rerurns "D" for Democrat, "R" for Republican, "O" for Other parties, "T"
    for a tie between two or more parties
    """
    if is_nan(branch):
        return branch
    party_count = dict()
    for i in branch:
        party = re.findall("[a-zA-Z]+", i)[0]
        count = re.findall("\d+", i)
        if len(count) == 0:
            count = 0
        else:
            count = count[0]
        if party in party_count:
            party_count[party] += count
        else:
            party_count[party] = count
    set_of_parties = get_parties_set(party_count)
    if len(set_of_parties) == 1:
        party = set_of_parties.pop().upper()
        if party == "D" or party == "R":
            return party
        else:
            return "O"      # Third party (other)
    else:
        return "T"          # Tie between multiple parties


def get_parties_set(count_dict):
    """
    Takes in a dictionary with the key as the party abbreviation and the value
    as the count of representatives in a branch with that party.
    Determines the max count of representatives and creates a set that stores
    the party abbrevations for all parties that have the max count of
    representatives.
    Returns the set of party abbreviations.
    """
    max_count = max(count_dict.values())
    set_of_parties = set()
    for key, value in count_dict.items():
        if value == max_count:
            set_of_parties.add(key)
    return set_of_parties


def create_poli_gdp_df(political_party, gdp_total):
    """
    Takes in the political party dataframe and the GDP dataframe
    Manipulates and merges the two dataframes to make a new dataframe with
    columns "majority", "gdp", and "year"
    Returns merged dataframe
    """

    result = pd.DataFrame()
    years = list(gdp_total.loc[:, (gdp_total.columns != "GeoFips") &
                               (gdp_total.columns != "GeoName")].columns)
    for year in years:
        df1_filtered = political_party[political_party["year"] == int(year)]
        df2_filtered = gdp_total[["GeoName", year]]
        merged = df1_filtered.merge(df2_filtered, left_on="state",
                                    right_on="GeoName", how="left")
        merged = merged.groupby("majority")[year].mean()
        merged_df = pd.DataFrame({"majority": merged.index,
                                  "gdp": merged.values})
        merged_df["year"] = [year] * len(merged_df)
        result = pd.concat([result, merged_df], axis=0)
    return result


def poli_vs_urban(poli, urban):
    """
    Takes in the political party data frame and the urban percentages dataframe
    Creates a new dataframe to account for the difference in format.
    Determines which political party each state was categorized as for the most
    amount of time during the decade. This becomes the state's political party
    for the decade. Categorizes a state as urban if it is above the mean
    percenatage for the decade or rural if it is below the mean.
    After processing these values, merges and combines the dataframes into
    a new dataframe with the columns of interest "majority", "state", and type"
    Returns the new dataframe
    """
    result = pd.DataFrame()
    min_decade = poli["year"].min()
    max_decade = poli["year"].max()
    max_decade = max_decade - (max_decade % 10)
    decades = list(range(min_decade, max_decade + 1, 10))
    for decade in decades:
        factor = urbanized_factor(urban, str(decade) + "_per")
        decade_urban = urban.loc[urban[str(decade) + "_per"] >= factor,
                                      ["Area Name", str(decade) + "_per"]]
        decade_urban["type"] = "Urban"
        decade_rural = urban.loc[urban[str(decade) + "_per"] < factor,
                                      ["Area Name", str(decade) + "_per"]]
        decade_rural["type"] = "Rural"
        poli_df = poli[(poli["year"] >= decade) & (poli["year"] < decade + 10)]
        # determine whether state was mostly republican or democrat for decade
        data = poli_df.groupby("state")["majority"].value_counts().to_frame(
                name="count").reset_index()
        idx = data.groupby("state")["count"].transform(max) == data["count"]
        data = data[idx]
        data["decade"] = decade
        merged1 = data.merge(decade_urban, left_on="state",
                             right_on="Area Name", how="inner")
        merged2 = data.merge(decade_rural, left_on="state",
                             right_on="Area Name", how="inner")
        final = pd.concat([merged1, merged2], axis=0)
        result = pd.concat([result, final], axis=0, sort=True)
    return result


def plot_poli_urban(df):
    """
    Takes in the merged poltical and urban percentages dataframe
    Creates two bar plots, one for urban states and one for rural states.
    These plots show the count of states that are categorized under each
    political party for each decade in the dataframe.
    Saves the urban plot in a file called "political_urban.png"
    Saves the rural plot in a fiel called "political_rural.png"
    """
    df = df.groupby(["decade", "type", "majority"]).count().reset_index()
    rural = df[df["type"] == "Rural"]
    urban = df[df["type"] == "Urban"]
    sns.catplot(data=urban, x="decade", y="count", hue="majority",
                kind="bar")
    plt.title("Count of Urban States by Political Party")
    plt.savefig("political_urban.png", bbox_inches="tight")
    sns.catplot(data=rural, x="decade", y="count", hue="majority",
                kind="bar")
    plt.title("Count of Rural States by Political Party")
    plt.savefig("political_rural.png", bbox_inches="tight")


def poli_vs_pop(poli, pop):
    """
    Takes in the political party dataframe and the population dataframe
    Separates the data into three dataframes for low population, medium
    population, and large population.
    Calls pop_years for each of the dataframes to plot them
    """
    pop2 = pop.loc[:, 'Alaska':'medium']
    low_pop = pop2[pop.loc[:, 'Alaska':'Wyoming'] < pop.loc[:, 'below'].mean()]
    low_pop["year"] = pop["year"]
    medium_pop = pop2[(pop.loc[:, 'Alaska':'Wyoming'] > pop.loc[:,
                       'below'].mean()) & (pop.loc[:,
                                                   'Alaska':'Wyoming'] <
                                           pop.loc[:, 'medium'].mean())]
    medium_pop["year"] = pop["year"]
    large_pop = pop2[pop.loc[:, 'Alaska':'Wyoming'] > pop.loc[:,
                     'medium'].mean()]
    large_pop["year"] = pop["year"]
    pop_years("low", poli, low_pop)
    pop_years("medium", poli, medium_pop)
    pop_years("large", poli, large_pop)


def pop_years(pop_type, poli, pop):
    """
    Takes in the category of population type for the current dataframe ("low",
    "medium", or "large")
    Takes in political party dataframe and population dataframe
    Creates a cleaner dataframe with the columns "year" and "state" for data in
    the population dataframe and merges it with the political dataframe.
    Creates and saves a bar plot of the count of states in the given population
    dataframe that are categorized under each political party over a range of
    years
    Saves plot to file called "{pop_type} pop_poli.png"
    """
    year_index = {"year": list(), "state": list()}
    for year in pop["year"]:
        row = pop[pop["year"] == year]
        for column in pop.loc[:, 'Alaska':'Wyoming'].columns:
            if not is_nan(row[column].values[0]):
                year_index["year"].append(year)
                year_index["state"].append(column)
    df = pd.DataFrame(year_index)
    data = df.merge(poli, on=["year", "state"], how="inner")
    palette = {"Democrat": "C0", "Republican": "C1", "Tie": "C3",
               "Other": "C4"}
    sns.catplot(data=data, x="year", hue="majority", kind="count",
                palette=palette)
    plt.title("Count of political parties in " + pop_type +
              " population states")
    plt.xticks(rotation=90)
    plt.savefig(pop_type + "pop_poli.png", bbox_inches="tight")


def plot_political_vs_gdp(df):
    """
    Takes in the merged political and GDP dataframe
    Creates a line plot of the avergae gdp for states of each political party
    over the set of years in the dataframe
    Saves the plot to a file called "political_vs_gdp.png"
    """
    sns.relplot(data=df, x='year', y="gdp", kind="line", hue="majority")
    plt.xticks(rotation=90)
    plt.ylabel("Average gdp")
    plt.title("Average GDP by Political Party")
    plt.savefig("political_vs_gdp.png", bbox_inches="tight")


def plot_political_vs_wage(poli, wage):
    """
    Takes in the political party dataframe and the wage dataframe
    Merges the dataframes based on state and year
    Creates a line plot of the average minimum wage for states categorized
    under each political party.
    Saves plot to file called "political_vs_wage.png"
    """
    data = pd.merge(poli, wage, how="inner", left_on=["state", "year"],
                    right_on=["State", "Year"])
    data = (data.groupby(["majority", "year"])["Low.Value"].mean()).to_frame(
            name="avg_wage").reset_index()
    sns.relplot(data=data, x="year", y="avg_wage", kind="line", hue="majority")
    plt.xticks(rotation=90)
    plt.ylabel("Average Minimum Wage (dollars)")
    plt.title("Average Minimum Wage by Political Party")
    plt.savefig("political_vs_wage.png", bbox_inches="tight")


def plot_political_vs_unemployment(poli, employ):
    """
    Takes in the political party dataframe and unemployment dataframe
    Merges the dataframes based on state and year
    Creates a line plot of the average unemployment rates for states
    categorized under each political party over a range of years.
    Saves plot to file called "political_vs_unemployment.png"
    """
    data = employ.groupby(["State", "Year"])["Rate"].mean().to_frame(
            name="avg_unemployment").reset_index()
    merged = pd.merge(poli, data, how="inner",
                      left_on=["state", "year"], right_on=["State", "Year"])
    merged = merged.groupby(
            ["majority", "year"])["avg_unemployment"].mean().to_frame(
                    name="avg_unemployment").reset_index()
    sns.relplot(data=merged, x="year", y="avg_unemployment",
                kind="line", hue="majority")
    plt.xticks(rotation=90)
    plt.ylabel("Average Unemplopyment Rate (percent)")
    plt.title("Average Unemploymemnt Rate by Political Party")
    plt.savefig("political_vs_unemployment.png", bbox_inches="tight")


def demographics_economy_GDP(GDP, urban):
    """
        This function will first merge GDP
        and urban dataset based on the states' name.

        Then it split the merged dataset into two parts
        based on whether a states consider to be a urban/rural
        state.

        It will plot a bar plot that will indicate year
        on the x-axis and average GDP on the y-axis in the end.
    """
    result = pd.DataFrame()
    year = list()
    for j in range(2):
        for i in range(1997, 2019):
            year.append(i)
    result['year'] = year

    combined = GDP.loc[1:, :].merge(urban.loc[1:,
                                              ['Area Name', '2000_per',
                                               '2010_per']],
                                    left_on='GeoName', right_on='Area Name',
                                    how='left')
    combined = combined.dropna()
    # split the dataset
    first_decade_urban = combined[combined['2000_per']
                                  >= urbanized_factor(combined, '2000_per')]
    first_decade_rural = combined[combined['2000_per']
                                  < urbanized_factor(combined, '2000_per')]
    second_decade_urban = combined[combined['2010_per']
                                   >= urbanized_factor(combined, '2010_per')]
    second_decade_rural = combined[combined['2010_per']
                                   < urbanized_factor(combined, '2010_per')]
    # Create a new dataframe for plotting
    cate = list()
    for i in range(1997, 2019):
        cate.append('urban')
    for i in range(1997, 2019):
        cate.append('rural')
    result['category'] = cate
    result['GDP'] = make_columns(first_decade_urban, second_decade_urban) \
        + make_columns(first_decade_rural, second_decade_rural)
    sns.catplot(x='year', y='GDP', data=result, hue='category', kind='bar')
    plt.xticks(rotation=45)
    plt.ylabel('Avearge GDP each year in rural/urban states(USD)')
    plt.title('Demographics vs GDP')
    plt.savefig('demographics_vs_economy_GDP.png', bbox_inches='tight')


def demographics_economy_unemployment(unemploy, urban):
    """
        This function will first merge unemployment
        and urban dataset based on the states' name.

        Then it split the merged dataset into two parts
        based on whether a states consider to be a urban/rural
        state.

        It will plot a bar plot that will indicate year
        on the x-axis and average unemployment rate on the y-axis in the end.
    """
    result = pd.DataFrame()
    # merge the dataset
    combined = unemploy.merge(urban.loc[1:, ['Area Name', '1990_per',
                                             '2000_per', '2010_per']],
                              left_on='State', right_on='Area Name',
                              how='left')
    combined = combined.dropna()
    # Create a dataframe for plotting
    rate_col = list()
    category = list()
    year = list()
    cate = ['rural', 'urban']
    for i in range(2):
        for j in range(1990, 2017):
            year.append(j)
            category.append(cate[i])
    # Split dataset based on urban/rural
    decade = find_decade(combined, 1990, 2020)
    for i in range(6):
        res = decade[i].groupby(['Year', 'State'])['Rate'].mean().to_frame(
                name='Rate')
        res = res.reset_index().groupby('Year')['Rate'].mean().tolist()
        rate_col += res
    result['Year'] = year
    result['Category'] = category
    result['Rate'] = rate_col
    sns.catplot(x='Year', y='Rate', data=result, hue='Category', kind='bar')
    plt.xticks(rotation=45)
    plt.ylabel('Average unemployment rate each year in rural/urban states')
    plt.title('Demographics vs Unemployment rate')
    plt.savefig('demographics_vs_economy_Unemployment.png',
                bbox_inches='tight')


def find_decade(df, year1, year2):
    """
        This is a private method for demographics_economy_unemployment
        and demographics_economy_min_wage which split the dataset based
        on rural/urban in decades.
    """
    res = list()
    for i in range(year1, year2, 10):
        res.append(df[(df[str(i)+'_per'] < urbanized_factor(df, str(i)+'_per'))
                      & ((df['Year'] >= i) & (df['Year'] < i+10))])
    for i in range(year1, year2, 10):
        res.append(df[(df[str(i)+'_per'] >= urbanized_factor(df,
                                                             str(i)+'_per')) &
                      ((df['Year'] >= i) & (df['Year'] < i+10))])
    return res


def demographics_economy_min_wage(wage, urban):
    """
        This function will first merge minimum wage
        and urban dataset based on the states' name.

        Then it split the merged dataset into two parts
        based on whether a states consider to be a urban/rural
        state.

        It will plot a bar plot that will indicate year
        on the x-axis and average minimum wage on the y-axis in the end.
    """
    result = pd.DataFrame()
    # merged dataset
    combined = wage.merge(urban.loc[1:,
                                    ['Area Name', '1960_per', '1970_per',
                                     '1980_per', '1990_per', '2000_per',
                                     '2010_per']],
                          left_on='State', right_on='Area Name', how='left')
    combined['Low.Value'] = combined['Low.Value'].fillna(0)
    # create new dataframe for plotting
    wage = list()
    category = list()
    year = list()
    cate = ['rural', 'urban']
    for i in range(2):
        for j in range(1968, 2018):
            year.append(j)
            category.append(cate[i])
    # Split data based on rural/urban
    decade = find_decade(combined, 1960, 2020)
    for i in range(12):
        res = decade[i].groupby(['Year'])['Low.Value'].mean()
        res = res.to_frame(name='Low.Value')
        res = res.reset_index()['Low.Value'].tolist()
        wage += res
    result['Year'] = year
    result['Category'] = category
    result['Low.Value'] = wage
    # plot
    sns.catplot(x='Year', y='Low.Value', data=result, hue='Category',
                kind='bar')
    plt.xticks(rotation=90)
    plt.ylabel('Average minimum wage each year in rural/urban states(USD)')
    plt.title('Demographics vs Min Wage')
    plt.savefig('demographics_vs_economy_min_wage.png', bbox_inches='tight')


def make_columns(df1, df2):
    """
        Find the mean of sum of columns 1997 to 2019
    """
    res = list()
    for i in range(1997, 2019):
        res.append(df1[str(i)].mean() + df2[str(i)].mean())
    return res


def urbanized_factor(df, year):
    """
        Find the mean of urbanized percentage
        of that year
    """
    return df.loc[:, year].mean()


def question2(pop, GDP_Total):
    """
        This function will first split the population
        dataset into three sets based on the its population
        comparing to the mean of total population of that year.

        Then it will match each population dataset with the corresponding
        GDP data.

        Finally, it will plot a scatter plot with populatio on x axis and
        GDP average on the y axis.
    """
    # Split the whole population into three based on the average pop of year
    pop2 = pop.loc[:, 'Alaska':'medium']
    low_pop = pop2[pop.loc[:, 'Alaska':'Wyoming'] < pop.loc[:, 'below'].mean()]
    medium_pop = pop2[(pop.loc[:, 'Alaska':'Wyoming'] > pop.loc[:,
                       'below'].mean())
                      & (pop.loc[:, 'Alaska':'Wyoming'] < pop.loc[:,
                         'medium'].mean())]
    large_pop = pop2[pop.loc[:, 'Alaska':'Wyoming'] > pop.loc[:,
                     'medium'].mean()]
    # Match GDP with population
    GDP = GDP_Total.loc[:, 'Alaska':'Wyoming']
    low_pop_GDP = GDP[pop.loc[:, 'Alaska':'Wyoming'] <
                      pop.loc[:, 'below'].mean()]
    low_pop_GDP['Sum'] = low_pop_GDP.mean(axis=1)
    medium_pop_GDP = GDP[(pop.loc[:, 'Alaska':'Wyoming'] > pop.loc[:,
                          'below'].mean())
                         & (pop.loc[:, 'Alaska':'Wyoming'] < pop.loc[:,
                            'medium'].mean())]
    medium_pop_GDP['Sum'] = medium_pop_GDP.mean(axis=1)
    large_pop_GDP = GDP[pop.loc[:, 'Alaska':'Wyoming'] >
                        pop.loc[:, 'medium'].mean()]
    large_pop_GDP['Sum'] = large_pop_GDP.mean(axis=1)
    low_pop['Sum'] = low_pop.sum(axis=1)
    medium_pop['Sum'] = medium_pop.sum(axis=1)
    large_pop['Sum'] = large_pop.sum(axis=1)
    low_pop['GDP'] = low_pop_GDP['Sum']
    medium_pop['GDP'] = medium_pop_GDP['Sum']
    large_pop['GDP'] = large_pop_GDP['Sum']
    low_pop['year'] = pop['year']
    medium_pop['year'] = pop['year']
    large_pop['year'] = pop['year']
    sns.relplot(x='Sum', y='GDP', data=low_pop, kind='scatter', size='year')
    plt.xticks(rotation=45)
    plt.xlabel('Total population in thousands from 1997 to 2018')
    plt.ylabel('Average GDP of all low population states in' +
               'millions of dollar')
    plt.title('Correlation between population in low population state and GDP')
    plt.savefig('low_pop_vs_GDP.png', bbox_inches='tight')
    sns.relplot(x='Sum', y='GDP', data=medium_pop, kind='scatter', size='year')
    plt.xticks(rotation=45)
    plt.xlabel('Total population in thousands from 1997 to 2018')
    plt.ylabel('Average GDP of all medium population states' +
               'in millions of dollar')
    plt.title('Correlation between population in medium population' +
              'state and GDP')
    plt.savefig('medium_pop_vs_GDP.png', bbox_inches='tight')
    sns.relplot(x='Sum', y='GDP', data=large_pop, kind='scatter', size='year')
    plt.xticks(rotation=45)
    plt.xlabel('Total population in thousands from 1997 to 2018')
    plt.ylabel('Average GDP of all large population' +
               'states in millions of dollar')
    plt.title('Correlation between population in large' +
              'population state and GDP')
    plt.savefig('large_pop_vs_GDP.png', bbox_inches='tight')


def question3(pop, percent_gdp, urban):
    """
    Takes in the population population dataframe, percent change in gdp
    dataframe, and urban percentages dataframe.
    Trains a model on part of the data, and uses the model to predict on both
    the train and test set
    Prints the mean squared error for both the train and test set
    """
    result = pd.DataFrame()
    # Combining pop, percent gdp and urban csv into one dataframe on the state
    pop_2 = pop.loc[1:, '1/1/1997':'1/1/2017']
    pop_2['State'] = pop.loc[1:, 'State']
    combined = pop_2.merge(urban.loc[1:, ['Area Name', '1990_per',
                                          '2000_per', '2010_per']],
                           left_on='State', right_on='Area Name', how='left')
    combined = combined.merge(percent_gdp.loc[1:, 'GeoName':'2017-2018'],
                              left_on='State', right_on='GeoName', how='left')
    first_decade_urban = combined[combined['1990_per']
                                  >= urbanized_factor(combined, '1990_per')]
    first_decade_rural = combined[combined['1990_per']
                                  < urbanized_factor(combined, '1990_per')]
    second_decade_urban = combined[combined['2000_per']
                                   >= urbanized_factor(combined, '2000_per')]
    second_decade_rural = combined[combined['2000_per']
                                   < urbanized_factor(combined, '2000_per')]
    third_decade_urban = combined[combined['2010_per']
                                  >= urbanized_factor(combined, '2010_per')]
    third_decade_rural = combined[combined['2010_per']
                                  < urbanized_factor(combined, '2010_per')]
    # Creating a new dataframe for the regression model
    pop = list()        # population of the corresponding state/year/category
    year = list()       # year from 1997 to 2017
    category = list()   # urban or rural
    states = list()     # all state
    gdp_per = list()    # gdp per of the corresponding state, year and category
    all_decade = [first_decade_rural, first_decade_urban, second_decade_rural,
                  second_decade_urban, third_decade_rural, third_decade_urban]
    decades = [1990, 1990, 2000, 2000, 2010, 2010]
    cate = ['rural', 'urban', 'rural', 'urban', 'rural', 'urban']
    for i in range(6):
        for j in range(1997, 2018):
            if (j >= decades[i]) and (j < decades[i]+10):
                res = all_decade[i]['1/1/' + str(j)].tolist()
                gdp_per += all_decade[i][str(j) + '-' + str(j+1)].tolist()
                pop += res
                states += all_decade[i]['State'].tolist()
                year += [j] * len(res)
                category += [cate[i]] * len(res)
    result['Pop'] = pop
    result['Year'] = year
    result['Category'] = category
    result['States'] = states
    result['GDP_per'] = gdp_per
    result = result.dropna()
    # build the regression model
    X = result.loc[:, result.columns != 'GDP_per']
    X = pd.get_dummies(X)
    Y = result['GDP_per']
    model = DecisionTreeRegressor()
    # Split into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model.fit(X_train, Y_train)
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    # The MSE is smaller for trained set than for the predicted sets.
    print("Mean squared error of train set:")
    print(mean_squared_error(Y_train, y_train_pred))
    print("Mean squared error of test set:")
    print(mean_squared_error(Y_test, y_test_pred))


def main():
    """
    Reads in all csv files and calls all functions to create plots and models
    """
    gdp_percent = pd.read_csv("Data/GDP_percent_change.csv")
    unemployment = pd.read_csv("Data/output.csv")
    wage = pd.read_csv('Data/Minimum Wage Data.csv', encoding="ISO-8859-1",
                       na_values=["", 0])
    poli = pd.read_csv('Data/states_party_strength_cleaned.csv')
    poli["state"] = poli["state"].apply(lambda x: x.title())
    urban_percent = pd.read_csv("Data/urban_percentages.csv")
    GDP_original = pd.read_csv('Data/GDP_total.csv')
    GDP_total_cleaned = pd.read_csv('Data/GDP_total_cleaned.csv')
    pop = pd.read_csv('Data/state_population.csv')
    pop_trans = pd.read_csv('Data/state_populations_thousands_transposed.csv')
    min_wage = pd.read_csv('Data/Minimum Wage Data.csv',
                           encoding='ISO-8859-1',
                           na_values=['', 0])
    poli["state"] = poli["state"].apply(lambda x: x.title())
    poli = calculate_party_majority(poli)
    poli_urban = poli_vs_urban(poli, urban_percent)

    plot_political_vs_gdp(create_poli_gdp_df(poli, GDP_original))
    plot_political_vs_wage(poli, wage)
    question2(pop, GDP_total_cleaned)
    plot_political_vs_unemployment(poli, unemployment)
    plot_poli_urban(poli_urban)
    poli_vs_pop(poli, pop)
    demographics_economy_GDP(GDP_original, urban_percent)
    demographics_economy_unemployment(unemployment, urban_percent)
    demographics_economy_min_wage(min_wage, urban_percent)
    question3(pop_trans, gdp_percent, urban_percent)


if __name__ == '__main__':
    main()
