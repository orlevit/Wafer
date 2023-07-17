import warnings
import pandas as pd
from additional_funcs import select_train_data, calc_df, print_stat, hyperparameters_search
from config import TRAIN_DATA_LOC, TEST_DATA_LOC, TRAIN_IND, TEST_IND
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")


def main():
    # Train - search for Hyper-parameters
    train_data_df = pd.read_csv(TRAIN_DATA_LOC)
    train_selected_data_df = select_train_data(train_data_df, TRAIN_IND)
    print_stat(train_data_df, train_selected_data_df, TRAIN_IND)
    best_poly_order, best_strength = hyperparameters_search(train_selected_data_df)

    # Predictions
    test_data_df = pd.read_csv(TEST_DATA_LOC)
    test_selected_data_df = select_train_data(test_data_df, TEST_IND)
    test_selected_data_df['IsScratchDie'] = False
    print_stat(test_data_df, test_selected_data_df, TEST_IND)
    _ = calc_df(test_selected_data_df, best_poly_order, best_strength, TEST_IND)
    test_data_df['IsScratchDie'] = False
    WaferName_scrach_pred_index = test_selected_data_df[test_selected_data_df['IsScratchDie'] == True].index.to_list()
    test_data_df.loc[WaferName_scrach_pred_index, 'IsScratchDie'] = True
    test_data_df.to_csv(TEST_DATA_LOC)


if __name__ == '__main__':
    main()
