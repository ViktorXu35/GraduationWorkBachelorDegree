import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# def preprocess():
#     lang_l = ['en', 'de', 'zh', 'zhw']
#
#     for lang in lang_l:
#         data = pd.read_csv(f"outputs/original/{lang}.csv", index_col=0).dropna()
#         meta = pd.read_excel(f'outputs/meta/{lang}.xlsx', index_col=0)
#
#         data.columns = [eval(x)[0] for x in data.columns[:len(data.columns)-1]] + ['score_tanh']
#         meta['DATE'] = pd.to_datetime(meta['DATE_NEW'], format='%Y年%m月%d日')
#         # meta['DATE'] = meta['DATE_NEW'].map(lambda x: pd.to_datetime('1899-12-30') + pd.Timedelta(str(x)+'D'))
#
#         section = meta.loc[:,'SOURCE':'DATE']
#         section['AGENCY'] = section['SOURCE'].str.split('_').map(lambda x: x[0])
#         section['FILE'] = section['SOURCE'].str.split('_').map(lambda x: int(x[1]))
#
#         new = pd.merge(data, section, on=['AGENCY', 'FILE'], how='inner')
#
#         new.to_csv(f'outputs/sentences/{lang}.csv')


def condition(items):
    con = ''
    for i in items:
        c = f"(data['SENTENCES'].str.contains('{i}'))"
        con += c + '|'
    return con.strip('|')


def overall_in_between(condi, score='score_sigmoid'):
    # 按条件筛选出的
    i = data.loc[eval(condi)]
    i.reset_index()

    c = i.groupby(['AGENCY', pd.to_datetime(i['DATE']).dt.strftime('%Y%m')])[score].mean().reset_index()
    p = i.groupby(['AGENCY', pd.to_datetime(i['DATE']).dt.strftime('%Y%m')])['SENTENCES'].size().reset_index()
    t = pd.DataFrame({'DATE': time_series})
    t2 = pd.DataFrame({'DATE': time_series})

    for name, group in c.groupby('AGENCY'):
        t = pd.merge(t, group.loc[:, 'DATE': score].reset_index(drop=True).rename(columns={score: name}), on='DATE',
                     how='outer')
    for name, group in p.groupby('AGENCY'):
        t2 = pd.merge(t2, group.loc[:, 'DATE': 'SENTENCES'].reset_index(drop=True).
                      rename(columns={'SENTENCES': name}), on='DATE', how='outer')

    for num, i in enumerate(list_agency):
        if i not in t.columns:
            if score == 'score_tanh':
                t.insert(num, i, 0)
            elif score == 'score_sigmoid':
                t.insert(num, i, 0.5)
        if i not in t2.columns:
            t2.insert(num, i, 0)

    if score == 'score_tanh':
        t = t.set_index('DATE').fillna(0)
    elif score == 'score_sigmoid':
        t = t.set_index('DATE').fillna(0.5)
    t2 = t2.set_index('DATE').fillna(0)

    # 算出每一家媒体的权重（句数量），与情感值作积，最后算平均值，得到总趋势
    result = (t2.div(t2.sum(axis=1), axis=0) * t).sum(axis=1).replace(0, 0.5).rename('总')
    return t2, result


def detailed(condi, score='score_sigmoid'):
    # 按条件筛选出的
    i = data.loc[eval(condi)]
    i.reset_index()
    time_series = ['202001', '202002', '202003', '202004', '202005', '202006', '202007', '202008', '202009',
                   '202010', '202011', '202012', '202101', '202102', '202103', '202104', '202105', '202106',
                   '202107', '202108', '202109']

    c = i.groupby(['AGENCY', pd.to_datetime(i['DATE']).dt.strftime('%Y%m')])[score].mean().reset_index()
    t = pd.DataFrame({'DATE': time_series})

    for name, group in c.groupby('AGENCY'):
        t = pd.merge(t, group.loc[:, 'DATE': score].reset_index(drop=True).rename(columns={score: name}),
                     on='DATE', how='outer')

    for num, i in enumerate(list_agency):
        if i not in t.columns:
            if score == 'score_tanh':
                t.insert(num, i, 0)
            elif score == 'score_sigmoid':
                t.insert(num, i, 0.5)

    if score == 'score_tanh':
        t = t.set_index('DATE').fillna(0)
    elif score == 'score_sigmoid':
        t = t.set_index('DATE').fillna(0.5)
    return t


if __name__ == '__main__':
    lang_l = ['ru', 'en', 'de', 'zh', 'zhw']

    vaccines = {
        '辉瑞': ['Pfizer', 'Пфайзер', '辉瑞', '輝瑞'],
        '国药': ['Sinopharm', '国药', 'BIBP', '國藥'],
        '卫星V': ['Спутник V', 'Sputnik V', '卫星-V', '卫星V', '卫星五号', '史普尼克5號', '史波尼克5號'],
        '阿斯利康': ['AstraZeneca', '阿斯利康', 'vaxzevria疫苗', 'AZ疫苗']
    }

    places = {
        '中国': ["Кита", "китайская", "китайской", 'China', 'Chinese', 'chinesisch', '中國', '中', '中国'],
        '俄罗斯': ['Росси', "российская", 'российской', 'Russia', 'Russland', 'russisch', '俄羅斯', '俄', '俄罗斯'],
        '美国': ['США', 'Америк', "американская", "американской", 'USA', 'US', 'amerikanisch', '美國', '美', '美国'],
        '德国': ["Германи", 'немецкая', 'немецкой', 'Germany', 'Germany', 'Deutschland', 'deutsch', '德國', '德', '德国'],
        '台湾': ['Тайван', "Тайваньская", "Тайваньской", 'Taiwan', 'Taiwanese', 'taiwanisch', '台湾', '台', '台灣'],
        '欧盟': ['ЕС', 'EU', '歐盟', '欧盟'],
    }

    time_series = ['202001', '202002', '202003', '202004', '202005', '202006', '202007', '202008', '202009',
                   '202010', '202011', '202012', '202101', '202102', '202103', '202104', '202105', '202106',
                   '202107', '202108', '202109']
    
    for kind, dic in dict(疫苗=vaccines, 地区=places).items():
        for k, v in tqdm(dic.items()):
            all_ = pd.DataFrame(index=time_series)
            for lang in lang_l:
                data = pd.read_csv(f'outputs/sentences/{lang}.csv', index_col=0)
                data.columns = [x.replace('\'', '').replace('(', '').replace(')', "").replace(',', '') for x in data.columns]
                list_agency = list(set(data['AGENCY'].to_list()))
                # 疫苗，地区
                t = detailed(condition(v))
                t2, m = overall_in_between(condition(v))

                t.plot(subplots=True, layout=(6, -1), ylim=(0, 1))
                fig = plt.gcf()
                fig.set_size_inches((8.5, 11), forward=False)
                fig.savefig(f'outputs/插图/{kind}/{lang}_{k}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                sns.heatmap(t.corr(method='spearman'), vmin=-1, vmax=1, annot=True, cmap='RdBu_r')
                fig = plt.gcf()
                fig.set_size_inches((8, 7.5), forward=False)
                fig.savefig(f'outputs/插图/热力图_{kind}/{lang}_{k}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)

                t2.plot.area()
                plt.savefig(f'outputs/插图/总描述_{kind}_区域图/{lang}_{k}.png', bbox_inches='tight', dpi=300)
                plt.close()

                m.plot()
                plt.savefig(f'outputs/插图/总_{kind}_折线图/{lang}_{k}.png', bbox_inches='tight', dpi=300)
                plt.close()

                all_ = pd.merge(all_, m, left_index=True, right_index=True)
            all_.columns = lang_l

    # 1.
            all_.plot(ylim=(-0.1, 1.1))
            # fig = plt.gcf()
            # fig.set_size_inches((8.5, 4), forward=False)
            plt.savefig(f'outputs/插图/不同媒体_{kind}_折线图/{k}.png', bbox_inches='tight', dpi=300)
            plt.close()
    # 2,
            # 考虑到序列为二分类序列，需要使用Kendall相关性系数
            sns.heatmap(all_.corr(method='spearman'), vmin=-1, vmax=1, annot=True, cmap='RdBu_r')
            # manager = plt.get_current_fig_manager()
            # manager.window.showMaximized()
            fig = plt.gcf()
            fig.set_size_inches((8, 7.5), forward=False)
            fig.savefig(f'outputs/插图/不同媒体_{kind}/{k}.png', bbox_inches='tight', dpi=300)
            plt.close(fig)
