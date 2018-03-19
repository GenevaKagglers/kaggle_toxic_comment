import os
import pandas as pd
import defines
import matplotlib.pyplot as plt
import seaborn as sns

in_dir = '/Users/scasasso/GenevaKagglers/kaggle_toxic_comment/submissions/best'

subm_dict = {
    'linear_regression_l2': 0.9772,
    # 'lightgbm_test': 0.9761,
    'fmftrl_test': 0.9805,
    'nbsvm_test': 0.9744,
    'spaCy_biLSTM': 0.9787,
    'biGRU_blend': 0.9836,
    # 'biGRU_fastText': 0.9832,
    # 'biGRU_glove': 0.9832,
    'biGRU_conv1d_blend': 0.9812,
    # 'biGRU_conv1d_glove': 0.9810,
    # 'biGRU_conv1d_word2vec': 0.9776,
    # 'biGRU_conv1d_fastText': 0.9768,
    'lvl0_lgbm_clean_sub': 0.9794
}

df_dict = {k.replace('.csv', ''): pd.read_csv(os.path.join(in_dir, k + '.csv'), index_col=0) for k, v in subm_dict.items()}
ids = df_dict['linear_regression_l2'].index

for class_name in defines.CLASS_COLS:
    print(class_name)

    df = pd.DataFrame(index=ids)

    for sub, df_sub in df_dict.items():
        df[sub] = df_sub[class_name].copy()

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax,
                # vmin=0.7,
                vmax=0.95
                )
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns, rotation='horizontal')
    plt.tight_layout()
    plt.savefig(class_name + '.png')

