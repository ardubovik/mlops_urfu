import pandas as pd


def main():

    # read dataset
    titanic_df = pd.read_csv('../datasets/titanic.csv')

    # replace null to mean
    mean_age = titanic_df['Age'].mean()
    titanic_df['Age'] = titanic_df['Age'].fillna(mean_age)

    # save dataset
    titanic_df.to_csv('../datasets/titanic.csv', index=False)


if __name__ == '__main__':
    main()
