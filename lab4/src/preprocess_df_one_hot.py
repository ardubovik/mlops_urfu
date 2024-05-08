import pandas as pd
from sklearn.preprocessing import OneHotEncoder 


def main():

    # read dataset
    titanic_df = pd.read_csv('../datasets/titanic.csv')

    # onehot encoding
    encoder = OneHotEncoder(sparse_output=False) 
    train_encoded = encoder.fit_transform(titanic_df[['Sex']]) 

    encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoder.get_feature_names_out(['Sex']),
        dtype=int,
    )

    titanic_df = pd.concat([titanic_df.drop(columns=['Sex']), encoded_df], axis=1)

    # save dataset
    titanic_df.to_csv('../datasets/titanic.csv', index=False)

if __name__ == '__main__':
    main()
