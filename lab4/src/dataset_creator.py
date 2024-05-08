from catboost.datasets import titanic

def main():

    # load dataset
    titanic_df, _ = titanic()

    # save dataset
    titanic_df.to_csv('../datasets/titanic.csv', index=False)


if __name__ == '__main__':
    main()
