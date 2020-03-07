import pandas as pd
import re
from collections import Counter
import math
import seaborn as sns
import matplotlib.pyplot as plt


#majority in 2 of the 3 branches
def calculate_party_majority(df):
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
        #max will be one if there is a tie
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
    return isinstance(x, float) and math.isnan(x)

def split_parties(row, column):
    if is_nan(row[column]):
        return row[column]
    else:
        return row[column].split(",")

def process_branch(branch):
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
            return "O"      #Third party (other) 
    else:
        return "T"          #Tie between multiple parties

def get_parties_set(count_dict):
    max_count = max(count_dict.values())
    set_of_parties = set()
    for key, value in count_dict.items():
        if value == max_count:
            set_of_parties.add(key)
    return set_of_parties


def question1(poli, GDP_Total):
    demo = dict()
    rep = dict()
    


def question2(pop, GDP_Total):
    # Split the whole population into three based on the average pop of each year
    pop2 = pop.loc[:, 'Alaska':'medium']
    low_pop = pop2[pop.loc[:, 'Alaska':'Wyoming'] < pop.loc[:, 'below'].mean()]
    medium_pop = pop2[(pop.loc[:, 'Alaska':'Wyoming'] > pop.loc[:, 'below'].mean()) & (pop.loc[:, 'Alaska':'Wyoming'] < pop.loc[:, 'medium'].mean())]
    large_pop = pop2[pop.loc[:, 'Alaska':'Wyoming'] > pop.loc[:, 'medium'].mean()]
    # Match GDP with population
    GDP = GDP_Total.loc[:, 'Alaska':'Wyoming']
    low_pop_GDP = GDP[pop.loc[:, 'Alaska':'Wyoming'] < pop.loc[:, 'below'].mean()]
    low_pop_GDP['Sum'] = low_pop_GDP.mean(axis=1)
    medium_pop_GDP = GDP[(pop.loc[:, 'Alaska':'Wyoming'] > pop.loc[:, 'below'].mean()) & (pop.loc[:, 'Alaska':'Wyoming'] < pop.loc[:, 'medium'].mean())]
    medium_pop_GDP['Sum'] = medium_pop_GDP.mean(axis=1)
    large_pop_GDP = GDP[pop.loc[:, 'Alaska':'Wyoming'] > pop.loc[:, 'medium'].mean()]
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
    plt.xticks(rotation = 45)
    plt.xlabel('Total population in thousands from 1997 to 2018')
    plt.ylabel('Average GDP of all low population states in dollars')
    plt.title('Correlation between population in low population state and GDP')
    plt.savefig("low_pop_vs_GDP.png")
    sns.relplot(x='Sum', y='GDP', data=medium_pop, kind='scatter', size='year')
    plt.xticks(rotation = 45)
    plt.xlabel('Total population in thousands from 1997 to 2018')
    plt.ylabel('Average GDP of all medium population states in dollars')
    plt.title('Correlation between population in medium population state and GDP')
    plt.savefig("medium_pop_vs_GDP.png")
    sns.relplot(x='Sum', y='GDP', data=large_pop, kind='scatter', size='year')
    plt.xticks(rotation = 45)
    plt.xlabel('Total population in thousands from 1997 to 2018')
    plt.ylabel('Average GDP of all large population states in dollars')
    plt.title('Correlation between population in large population state and GDP')
    plt.savefig("large_pop_vs_GDP.png")

    
def question3():
    return None

def create_poli_gdp_df(political_party, gdp_total):
    #filter into states that are democrat:
    #find average of each column
    #make new df
    result = pd.DataFrame()
    years = list(gdp_total.loc[:, (gdp_total.columns != "GeoFips") & (gdp_total.columns != "GeoName")].columns)
    for year in years:
        df1_filtered = political_party[political_party["year"] == int(year)]
        df2_filtered = gdp_total[["GeoName", year]]
        merged = df1_filtered.merge(df2_filtered, left_on = "state",  right_on = "GeoName", how = "left")
        merged = merged.groupby("majority")[year].mean()
        merged_df = pd.DataFrame({"majority":merged.index, "gdp":merged.values})
        merged_df["year"] = [year] * len(merged_df)
        result = pd.concat([result, merged_df], axis=0)
    return result
    
def plot_political_vs_gdp(df):
    sns.relplot(data=df, x='year', y="gdp", kind="line", hue="majority")
    plt.xticks(rotation = 90)
    plt.ylabel("Average gdp")
    plt.title("Average gdp by Political Party")
    plt.savefig("political_vs_gdp.png")


def main():
    poli = pd.read_csv('Data/states_party_strength_cleaned.csv')
    gdp_percent = pd.read_csv("Data/GDP_percent_change.csv")
    GDP_Total = pd.read_csv('Data/GDP_total_cleaned.csv')
    unemployment = pd.read_csv("Data/output.csv")
    poli["state"] = poli["state"].apply(lambda x: x.title())
    urban_percent =pd.read_csv('Data/urban_percentages.csv')
    pop = pd.read_csv('Data/state_population.csv')
    poli = calculate_party_majority(poli)
    plot_political_vs_gdp(create_poli_gdp_df(poli, GDP_Total))
    question2(pop, GDP_Total)
    question3()

if __name__ == '__main__':
    main()