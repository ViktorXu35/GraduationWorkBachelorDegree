import pandas as pd
import spacy
from tqdm import tqdm
import networkx as nx
import numpy as np
from zhconv import convert


# 公用区域
def parse(lang, lang_engine):
    sent_df = pd.read_csv(f'{lang}/{lang}_sentences_base.csv', index_col=0).dropna()

    # 加载停用词和专有名词短语
    sw = open(f'{lang}/config/stopwords.txt', 'r', encoding='utf-8', errors='ignore')
    cl = open(f'{lang}/config/col.txt', 'r', encoding='utf-8', errors='ignore')
    stop_word = sw.readlines()
    collocation = cl.readlines()
    nlp = spacy.load(lang_engine)

    if lang == 'zhw':
        data = []
        for i in tqdm(sent_df.index):
            sentence = convert(sent_df['SENTENCES'][i], 'zh-cn')
            doc = nlp(sentence)
            sentence_parse = [(token, token.head, token.dep_, token.lemma_, token.pos_,
                               sent_df['AGENCY'][i], sent_df['FILE'][i], sent_df['LABEL'][i], token.i, token.head.i) for
                              token in doc]
            data.extend(sentence_parse)

        parse = pd.DataFrame(data)
        parse.columns = ['word', 'dependence', 'mark', 'original_form', 'POS',
                         'agency', 'file', 'sentence', 'word_number', 'dependence_number']
        parse.to_csv(f'{lang}/{lang}_word_base.csv')

    else:
        data = []
        for i in tqdm(sent_df.index):
            sentence = sent_df['SENTENCES'][i]
            doc = nlp(sentence)
            sentence_parse = [(token, token.head, token.dep_, token.lemma_, token.pos_,
                               sent_df['AGENCY'][i], sent_df['FILE'][i], sent_df['LABEL'][i], token.i, token.head.i) for
                              token in doc]
            data.extend(sentence_parse)

        parse = pd.DataFrame(data)
        parse.columns = ['word', 'dependence', 'mark', 'original_form', 'POS',
                         'agency', 'file', 'sentence', 'word_number', 'dependence_number']
        parse.to_csv(f'{lang}/{lang}_word_base.csv')
    return parse


def analysis(lang, lang_engine):
    senti_df = pd.read_csv('senti_word.csv', index_col=0)

    nlp = spacy.load(lang_engine)

    # 情感词典
    senti_ = senti_df[senti_df['LANG'] == lang].reset_index(drop=True)
    senti = pd.concat([senti_, pd.read_csv(f'{lang}/config/add.csv', index_col=0)], ignore_index=True)

    POS = []
    for i in tqdm(senti['WORDS']):
        doc = nlp(str(i))
        POS.append([token.pos_ for token in doc][0])

    senti['POS'] = POS

    # 数值化
    senti['POLARITY'] = senti['POLARITY'].map(lambda x: -1 if x == 'neg' else 1)
    # 名词、副词1分，形容词、动词2分
    senti['POS_score'] = senti['POS'].map(lambda x: 1.5 if (x == 'NOUN' or x == 'ADV' or x == 'PART')
    else (2 if (x == 'ADJ' or x == 'VERB') else 0))

    senti['SCORE'] = senti['POLARITY'] * senti['POS_score']

    # parsed_file
    parsed_file = pd.read_csv(f'{lang}/{lang}_word_base.csv', index_col=0)

    groups = parsed_file.groupby(['agency', 'file', 'sentence'])

    patterns = [('ADJ', 'NOUN'), ('VERB', 'NOUN'), ('NOUN', 'NOUN'), ('NUM', 'NOUN'),
                ('ADJ', 'VERB'), ('ADV', 'VERB'), ('VERB', 'VERB'), ('NOUN', 'VERB'), ('PART', 'VERB'),
                ('ADJ', 'ADJ'), ('ADV', 'ADJ'),
                ('ADJ', 'PRON'), ('VERB', 'PRON'), ('NOUN', 'PRON'), ('NUM', 'PRON')]

    results = []
    for name, group in tqdm(groups):
        # 匹配词的情感值
        trial = group.reset_index(drop=True)

        if lang == 'zh' or lang == 'zhw':
            trial['senti'] = trial['word'].map(
                lambda x: 1 if len(senti.loc[senti['WORDS'] == x]['SCORE'].values) == 0 else
                senti.loc[senti['WORDS'] == x]['SCORE'].values[0])

        else:
            trial['senti'] = trial['original_form'].map(
                lambda x: 1 if len(senti.loc[senti['WORDS'] == x]['SCORE'].values) == 0 else
                senti.loc[senti['WORDS'] == x]['SCORE'].values[0])

        weighted_edges = zip(trial['word_number'], trial['dependence_number'])

        g = nx.DiGraph()

        g.add_edges_from(weighted_edges)

        # 出度为1，入度为0的点为leaf（子叶）
        leaf = [x for x in g.nodes() if g.out_degree(x) == 1 and g.in_degree(x) == 0]

        root = list(filter(lambda x: x[0] == x[1], g.edges()))

        sign = lambda x: -1 if x < 0 else (1 if x > 0 else 0)

        scores = 0
        if len(root) == 1:
            paths = [list(nx.all_simple_paths(g, i, root[0][0])) for i in leaf]
            # 检查路径中是否存在pattern
            for i in paths:
                # 把位置映射为词性
                mapped_paths = list(map(lambda x: (x, trial['POS'][x]), i[0]))
                score = 0
                for j in range(len(mapped_paths) - 1):
                    if (mapped_paths[j][1], mapped_paths[j + 1][1]) in patterns:
                        s = abs(trial['senti'][mapped_paths[j][0]]) * abs(trial['senti'][mapped_paths[j + 1][0]]) * sign(
                            trial['senti'][mapped_paths[j][0]])
                        if s > 0:
                            s -= 1
                            score += s
                        else:
                            score += s
                    else:
                        pass
                    scores += score
        else:
            pass
        results.append(scores)

    final_score = pd.DataFrame(
        {
            'agency': [x[0] for x in groups.groups.keys()],
            'file': [x[1] for x in groups.groups.keys()],
            'sentence': [x[2] for x in groups.groups.keys()],
            'score': results
        }
    )

    final_score['score_mmstd'] = (final_score.score - final_score.score.min()) / (final_score.score.max() - final_score.score.min())
    final_score['score_max_abs_score'] = final_score.score / final_score.score.abs().max()
    final_score['score_sigmoid'] = 1 / (1 + np.exp(-final_score.score))
    final_score.to_csv(f'{lang}/{lang}_final_score.csv')
    return final_score


if __name__ == '__main__':
    lang_l = ['ru', 'en', 'de', 'zh', 'zhw']
    lang_engine_l = ['ru_core_news_sm', 'en_core_web_sm', 'de_core_news_sm', 'zh_core_web_sm', 'zh_core_web_sm']

    # for lang, lang_engine in zip(lang_l, lang_engine_l):
    lang = 'zhw'
    lang_engine = 'zh_core_web_sm'
    # parse(lang, lang_engine)
    # data = analysis(lang, lang_engine)
    # 合并
    for lang in lang_l:
        data = pd.read_csv(f'{lang}/{lang}_sentences_base.csv', index_col=0)
        score = pd.read_csv(f'{lang}/{lang}_final_score.csv', index_col=0)
        score.columns = ['AGENCY', 'FILE', 'LABEL', *zip(score.columns[3:])]
        merged = data.merge(score, on=['AGENCY', 'FILE', 'LABEL'])
        merged.columns = [*zip(merged.columns[:len(merged.columns) - 1]), 'score_tanh']
        merged.to_csv(f"outputs/{lang}.csv")
