import pandas as pd


def main():

    # read dataset
    titanic_df = pd.read_csv('../datasets/titanic.csv')

    # modification dataset
    titanic_df = titanic_df[['Pclass', 'Sex', 'Age']]
    titanic_df['Pclass'][0:10] = 2

    # save dataset
    titanic_df.to_csv('../datasets/titanic.csv', index=False)


if __name__ == '__main__':
    main()
