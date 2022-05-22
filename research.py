import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

lang_l = ['ru', 'en', 'de', 'zh', 'zhw']
# lang_engine_l = ['ru_core_news_sm', 'en_core_web_sm', 'de_core_news_sm', 'zh_core_web_sm', 'zh_core_web_sm']

for lang in lang_l:
    data = pd.read_csv(f'{lang}/{lang}_final_score.csv', index_col=0)
    # data['score_tanh'] = np.tanh(data['score'])
    # data.to_csv(f'{lang}/{lang}_final_score.csv')
    data['score_tanh'].hist(bins=20)
    plt.show()