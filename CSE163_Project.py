import pandas as pd
import re

def question1(poli, GDP_Total):
    demo = dict()
    rep = dict()
    for i in range(len(poli)):
        row = poli.loc[i, :]
        congress = str(row['congress house'])
        senate = str(row['congress sen'])
        print(congress)
        print(senate)
        num_demo = int(congress[congress.index('D')-1 if 'D' in congress else 0]) + int(senate[senate.index('D')-1 if 'D' in senate else 0])
        print(num_demo)
    

def question2(pop, GDP_Total):
    low_pop = pop.loc[52, '1/1/1990':'1/1/2019']
    #pop[pop[:, '1/1/1990':'1/1/2019'] < pop.loc[52, '1/1/1990':'1/1/2019']]
    print(low_pop)


def question3():
    return None


def main():
    poli = pd.read_csv('Data/states_party_strength_cleaned.csv', encoding = "ISO-8859-1")
    GDP_Total = pd.read_csv('Data/GDP_total.csv')
    pop = pd.read_csv('Data/state_populations_thousands_transposed.csv')
    #question1(poli, GDP_Total)
    question2(pop, GDP_Total)
    question3()


if __name__ == '__main__':
    main()