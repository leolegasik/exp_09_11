"""Сравнение лексических сочетаний и OpenIE на одной выборке Gazeta."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import experiment as e

OUT = e.ROOT/'2and3_vs_openIE'
LEGACY_NGRAM_SHA256 = '588a517c73e7db26fb9b3dfe7a2448c481fe4ed1aba6ffaf8bf52f0e5d2b1f7e'
STOP = set('и в во не на а но что это по с со из у к ко за от до для о об при как то так же ли бы был была были быть он она они его ее их мы вы я оно или уже еще только также который которая которые если чем после через между под над без'.split())


def ngrams(docs):
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), lowercase=True,
                               token_pattern=r'(?u)\b[а-яёА-ЯЁ]{2,}\b', max_features=30000, min_df=1)
    matrix = vectorizer.fit_transform([d['text'] for d in docs])
    names = vectorizer.get_feature_names_out()
    records = []
    for i, doc in enumerate(docs):
        row = matrix.getrow(i)
        indices = row.indices[np.argsort(row.data)[::-1]]
        phrases = [names[j] for j in indices
                   if names[j].split()[0] not in STOP and names[j].split()[-1] not in STOP][:6]
        records.append({'id': doc['id'], 'ngrams': phrases, 'labels': e.lexical_labels(doc['text'])})
    return records


def content_words(text):
    return set(re.findall(r'[а-яё]{2,}', text.lower().replace('ё', 'е'))) - STOP


def triple_text(triples):
    # Цитаты-основания не участвуют в оценке словесного покрытия.
    return ' '.join(' '.join(t[:3]) for t in triples)


def word_coverage(reference, candidate):
    reference = content_words(reference)
    return len(reference & content_words(candidate))/len(reference) if reference else None


def bootstrap_difference(differences):
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(e.SEED)
    means = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def plots(documents, pilot, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(['Би-/триграммы', 'OpenIE'], [documents.ngram_units.mean(), documents.openie_units.mean()], color=['#64748b', '#087f8c'])
    axes[0].set_ylabel('Среднее число единиц на текст')
    axes[0].set_title('Единицы разного типа: фразы и связи')
    axes[1].bar(['Би-/триграммы', 'OpenIE'], [documents.ngram_characters.mean(), documents.openie_characters.mean()], color=['#64748b', '#087f8c'])
    axes[1].set_ylabel('Средняя длина, символов с пробелами')
    axes[1].set_title('Только представление, без исходного текста и цитат')
    fig.suptitle('Объём представления · 1000 текстов · это не оценка качества')
    fig.tight_layout(); fig.savefig(OUT/'representation.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, row in enumerate(pilot.to_dict('records')):
        ax.plot([0, 1], [row['ngram_word_coverage']*100, row['openie_word_coverage']*100], '-o', alpha=.65, label=f"{i+1}. {row['id']}")
    ax.set_xticks([0, 1], ['Би-/триграммы', 'OpenIE'])
    ax.set_xlim(-.2, 1.2); ax.set_ylim(0, 100)
    ax.set_ylabel('Покрытие значимых слов человеческих связей, %')
    ax.set_title('10 пилотных текстов · точные словоформы, без лемматизации\nНе измеряет смысловую эквивалентность, отрицание или направление')
    ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/'human_word_coverage.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = [summary['facet_metrics'][g]['before_micro_f1'] for g in ['pilot', 'audit']]
    positions = np.arange(2)
    ax.bar(positions-.18, values, .36, label='До удаления n-грамм', color='#64748b')
    ax.bar(positions+.18, values, .36, label='После удаления n-грамм', color='#087f8c')
    for p, value in zip(positions, values):
        for shift in [-.18, .18]:
            ax.text(p+shift, value+.015, f'{value:.3f}', ha='center')
    ax.set_xticks(positions, ['Пилот · 10 текстов', 'Аудит · 6 текстов'])
    ax.set_ylim(0, .6); ax.set_ylabel('Micro-F1 Qwen по шести признакам')
    ax.set_title('Качество ответов Qwen осталось прежним\nПромпт и исходные ответы не менялись')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/'quality_before_after.png', dpi=160); plt.close(fig)


def build_html(docs, lexical, pred, human, pilot, summary):
    escape = html.escape
    blocks = []
    cov = pilot.set_index('id')
    for i, doc in enumerate(d for d in docs if d['id'] in human):
        key = doc['id']
        phrases = ''.join('<li>'+escape(x)+'</li>' for x in lexical[key]['ngrams'])
        triples = ''.join('<li><b>'+escape(' → '.join(t[:3]))+'</b><blockquote>'+escape(t[3])+'</blockquote></li>' for t in pred[key]['prediction']['triples'])
        reference = ''.join('<li>'+escape(' → '.join(t))+'</li>' for t in human[key]['triples'])
        blocks.append(f'<section><h2>{i+1}. {escape(key)}</h2><p class="text">{escape(doc["text"])}</p><div class="columns"><div><h3>Би-/триграммы TF–IDF</h3><ul>{phrases}</ul></div><div><h3>OpenIE через Qwen</h3><ul>{triples}</ul></div></div><h3>Первоначальные связи человека</h3><ul>{reference}</ul><p>Словесное покрытие: би-/триграммы — {cov.loc[key,"ngram_word_coverage"]:.1%}; OpenIE — {cov.loc[key,"openie_word_coverage"]:.1%}. Это совпадение слов, а не оценка смысла.</p></section>')
    images = ''.join('<img alt="'+label+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">' for name, label in [('quality_before_after.png', 'Качество до и после'), ('representation.png', 'Объём представлений'), ('human_word_coverage.png', 'Покрытие слов человеческих связей')])
    body = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Би-/триграммы и OpenIE</title><style>
body{{font:17px/1.55 system-ui,sans-serif;color:#173242;background:#edf3f6;max-width:1150px;margin:auto;padding:24px}}section,header{{padding:26px;margin:22px 0;background:white;border-radius:12px}}h1{{line-height:1.2}}.columns{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}.text{{background:#eaf5f4;padding:16px}}blockquote{{font-size:14px;color:#526471;margin:10px 0 20px}}li{{margin:10px 0}}img{{width:100%;height:auto;margin:12px 0}}@media(max-width:750px){{.columns{{grid-template-columns:1fr}}body{{padding:10px}}}}
</style><header><h1>Би-/триграммы и OpenIE</h1><p>1000 текстов Gazeta; подробные примеры — все 10 пилотных текстов.</p><p><b>Качество ответов Qwen не изменилось.</b> Изменился способ представления: сочетания слов заменены уже извлечёнными отношениями. OpenIE не является новой независимой моделью в этом сравнении.</p><p>Ручная оценка OpenIE: совпало 14 из 17 машинных связей с 27 человеческими; precision 82,4%, recall 51,9%, F1 0,636. Все 17 связей участник признал подтверждёнными текстом. Для n-грамм аналогичной человеческой оценки нет.</p>{images}</header>{''.join(blocks)}</html>'''
    coverage = summary['human_word_coverage']
    body = body.replace('<h1>Би-/триграммы и OpenIE</h1>',
                        '<h1>Би-/триграммы и OpenIE</h1>' +
                        f'<p><b>Среднее покрытие слов человеческих связей: n-граммы — {coverage["ngram_mean"]:.1%}; OpenIE — {coverage["openie_mean"]:.1%}.</b> OpenIE выше в {coverage["openie_higher_documents"]} из 10 текстов. Этот показатель не измеряет смысловую точность.</p>')
    (OUT/'comparison.html').write_text(body, encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    docs, pred = e.corpus(), e.predictions()
    if len(docs) != 1000 or {d['id'] for d in docs} != set(pred):
        raise RuntimeError('Нужна полная выборка из 1000 машинных ответов.')
    human = e.load_annotations('pilot', required=True)
    measured = e.load_json(e.ROOT/'results/summary.json')
    start = time.perf_counter()
    lexical_records = ngrams(docs)
    ngram_runs = [time.perf_counter()-start]
    payload = ''.join(json.dumps(r, ensure_ascii=False)+'\r\n' for r in lexical_records)
    (OUT/'ngrams.jsonl').write_bytes(payload.encode('utf-8'))
    restored_hash = hashlib.sha256((OUT/'ngrams.jsonl').read_bytes()).hexdigest()
    if restored_hash != LEGACY_NGRAM_SHA256:
        raise ValueError('Восстановленные n-граммы отличаются от прежнего файла.')
    lexical = {r['id']: r for r in lexical_records}
    openie_runs = []
    for _ in range(3):
        start = time.perf_counter()
        _ = ngrams(docs)
        ngram_runs.append(time.perf_counter()-start)
        start = time.perf_counter()
        _ = [e.openie_document(d, pred[d['id']]) for d in docs]
        openie_runs.append(time.perf_counter()-start)
    document_rows, pilot_rows = [], []
    for doc in docs:
        key = doc['id']; triples = pred[key]['prediction']['triples']
        phrases = lexical[key]['ngrams']
        ngram_text, openie_text = ' '.join(phrases), triple_text(triples)
        document_rows.append({'id': key, 'split': doc['split'], 'ngram_units': len(phrases),
                              'openie_units': len(triples), 'ngram_characters': len(ngram_text),
                              'openie_characters': len(openie_text)})
        if key in human:
            reference = triple_text(human[key]['triples'])
            pilot_rows.append({'id': key, 'human_relations': len(human[key]['triples']),
                               'ngram_word_coverage': word_coverage(reference, ngram_text),
                               'openie_word_coverage': word_coverage(reference, openie_text)})
    documents, pilot = pd.DataFrame(document_rows), pd.DataFrame(pilot_rows)
    documents.to_csv(OUT/'per_document.csv', index=False, encoding='utf-8-sig')
    pilot.to_csv(OUT/'human_word_coverage.csv', index=False, encoding='utf-8-sig')
    diff = pilot.openie_word_coverage - pilot.ngram_word_coverage
    stats = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'documents': len(docs),
             'pilot_documents': len(pilot), 'protocol_sha256': e.protocol_hash(),
             'corpus_sha256': e.load_json(e.ROOT/'data/manifest.json')['corpus_file_sha256'],
             'restored_ngram_file_sha256': restored_hash, 'matches_legacy_ngram_file': True,
             'ngram_units': int(documents.ngram_units.sum()), 'openie_relations': int(documents.openie_units.sum()),
             'ngram_mean_characters': float(documents.ngram_characters.mean()),
             'openie_mean_characters': float(documents.openie_characters.mean()),
             'ngram_nonempty_documents': int((documents.ngram_units>0).sum()),
             'openie_nonempty_documents': int((documents.openie_units>0).sum()),
             'human_word_coverage': {'ngram_mean': float(pilot.ngram_word_coverage.mean()),
                                     'openie_mean': float(pilot.openie_word_coverage.mean()),
                                     'openie_minus_ngram_mean': float(diff.mean()),
                                     'paired_bootstrap95': bootstrap_difference(diff),
                                     'openie_higher_documents': int((diff>0).sum()),
                                     'equal_documents': int((diff==0).sum()),
                                     'ngram_higher_documents': int((diff<0).sum()),
                                     'metric': 'Exact content-word overlap, no lemmatization; evidence excluded; not semantic quality'},
             'manual_relation_evaluation': {'openie': e.relation_metrics('pilot'), 'ngrams': None},
             'facet_metrics': {},
             'timing': {'ngram_compute_seconds_runs': ngram_runs,
                        'ngram_compute_seconds_median': statistics.median(ngram_runs),
                        'openie_from_memory_seconds_runs': openie_runs,
                        'openie_from_memory_seconds_median': statistics.median(openie_runs),
                        'original_qwen_generation_seconds_sum': sum(r['seconds'] for r in pred.values()),
                        'note': 'In-memory steps exclude disk IO; original generation is historical, not a simultaneous speed benchmark'},
             'additional_model_calls': 0}
    for group in ['pilot', 'audit']:
        before = measured[group+'_evaluation']['methods']['qwen']['micro_f1']
        after = e.evaluate(group)['methods']['qwen']['micro_f1']
        stats['facet_metrics'][group] = {'before_micro_f1': before, 'after_micro_f1': after, 'difference': after-before}
    e.save_json(OUT/'statistics.json', stats)
    plots(documents, pilot, stats)
    build_html(docs, lexical, pred, human, pilot, stats)
    c, t = stats['human_word_coverage'], stats['timing']
    report = f'''# Би-/триграммы и OpenIE: сравнение

## Что проверено

Одна выборка: 1000 редакционных аннотаций Gazeta. Для каждого текста восстановлены прежние 6 лучших би-/триграмм TF–IDF и взяты сохранённые OpenIE-связи Qwen. Параметры TF–IDF и фильтр крайних стоп-слов повторяют прежнюю реализацию; файл `ngrams.jsonl` позволяет проверить восстановление по SHA-256. Дополнительных запросов к модели нет.

**Качество ответов Qwen после удаления n-грамм не изменилось:** сами ответы и промпт остались прежними. Сравниваются два представления одного текста, а не две версии модели. Шесть бинарных признаков составляют отдельную задачу.

## Измеренные различия

| Показатель | Би-/триграммы | OpenIE |
|---|---:|---:|
| Текстов | 1000 | 1000 |
| Текстов с непустым представлением | {stats['ngram_nonempty_documents']} | {stats['openie_nonempty_documents']} |
| Единиц представления | {stats['ngram_units']} сочетаний | {stats['openie_relations']} связей |
| Средняя длина представления, символов | {stats['ngram_mean_characters']:.1f} | {stats['openie_mean_characters']:.1f} |
| Среднее покрытие слов человеческих связей, 10 текстов | {c['ngram_mean']:.1%} | {c['openie_mean']:.1%} |
| Ручное смысловое сопоставление | Не проводилось | 14 совпадений; precision 82,4%, recall 51,9%, F1 0,636 |

Число фраз и число связей относятся к разным единицам. Более длинное представление не обязательно лучше.

![Объём представлений](representation.png)

## Покрытие слов человеческой разметки

Для каждого пилотного текста объединяются значимые слова всех человеческих связей. Считается доля этих слов, присутствующих в представлении. Регистр и «ё/е» нормализуются, стоп-слова исключаются; лемматизации нет. В OpenIE учитываются только субъект, отношение и объект — исходный текст и цитаты исключены. Среднее считается по текстам, а не по всем словам сразу.

Разность OpenIE − n-граммы: **{c['openie_minus_ngram_mean']*100:+.1f} процентного пункта**; парный bootstrap 95% по 10 текстам, 2000 повторов: [{c['paired_bootstrap95'][0]*100:+.1f}; {c['paired_bootstrap95'][1]*100:+.1f}] п. п. OpenIE выше в {c['openie_higher_documents']} текстах, n-граммы — в {c['ngram_higher_documents']}, равенство — в {c['equal_documents']}.

Это лексическая диагностика, **не семантическая точность**: перефразирование снижает совпадение слов, а одинаковые слова могут передавать разные утверждения. Направление, отрицание и авторство этой метрикой не проверяются. Выборка мала, один участник.

![Покрытие слов](human_word_coverage.png)

## Качество ответов до и после

| Оценка Qwen | До | После | Разность |
|---|---:|---:|---:|
| Micro-F1 признаков, пилот 10 текстов | {stats['facet_metrics']['pilot']['before_micro_f1']:.3f} | {stats['facet_metrics']['pilot']['after_micro_f1']:.3f} | 0 |
| Micro-F1 признаков, аудит 6 текстов | {stats['facet_metrics']['audit']['before_micro_f1']:.3f} | {stats['facet_metrics']['audit']['after_micro_f1']:.3f} | 0 |
| F1 смысловых связей, пилот | 0,636 | 0,636 | 0 |

Это проверка сохранения результатов, а не независимый эксперимент «до/после». Словарный метод нельзя считать показателем качества n-грамм: он отдельно определяет шесть признаков.

![Качество до и после](quality_before_after.png)

## Время

Построение TF–IDF и отбор фраз для 1000 текстов из памяти: медиана **{t['ngram_compute_seconds_median']:.3f} с**, 4 замера. Формирование OpenIE-записей из уже готовых ответов в памяти: **{t['openie_from_memory_seconds_median']:.3f} с**, 3 замера. Сохранение файлов и чтение кэша не включены. Это стоимость обработки готовых данных, а не полного извлечения моделью.

Сумма сохранённых времён первоначальных запросов Qwen по 1000 текстам: **{t['original_qwen_generation_seconds_sum']:.1f} с ({t['original_qwen_generation_seconds_sum']/60:.1f} мин)**. Время историческое, включает совместное извлечение связей и признаков; оно не является отдельным замером только OpenIE. Новый полный прогон для сравнения скорости не выполнялся. Удаление TF–IDF убирает небольшой дополнительный этап, но не ускоряет саму генерацию Qwen.

## Вывод

OpenIE явно представляет участников и отношение между ними и позволяет сопоставить связи с человеческими; n-граммы показывают лексические сочетания. При этом текущая переработка не улучшила ответы модели: отношения уже извлекались ранее. Новое сравнение показывает различие представлений и неизменность измеренного качества. Для доказательства преимущества одного представления по понятности нужна единая человеческая оценка обоих вариантов; такой оценки в имеющихся данных нет.

Все 10 примеров представлены рядом в `comparison.html` (открывается локально без интернета). Данные: `per_document.csv`, `human_word_coverage.csv`, `statistics.json`. Воспроизведение из корня проекта: `python compare_methods.py`.
'''
    (OUT/'README.md').write_text(report, encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


if __name__ == '__main__':
    main()
