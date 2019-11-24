import re
import warnings
warnings.filterwarnings("ignore")

import requests
from pymystem3 import Mystem
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

m = Mystem()

url = 'https://raw.githubusercontent.com/akutuzov/universal-pos-tags/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map'

obscene_words=['блядство', 'блять', 'хуесос', 'еблан', 'говноед', 'блядина', 'ебёт', 'ебать', 'нехуй', 'нихуя', 'пиздуй', 'пиздабол']

mapping = {}
r = requests.get(url, stream=True)
for pair in r.text.split('\n'):
    pair = re.sub('\s+', ' ', pair, flags=re.U).split(' ')
    if len(pair) > 1:
        mapping[pair[0]] = pair[1]


def tag_mystem(text):
    processed = m.analyze(text)
    tagged = []
    for w in processed:
        try:
            lemma = w["analysis"][0]["lex"].lower().strip()
            pos = w["analysis"][0]["gr"].split(',')[0]
            pos = pos.split('=')[0].strip()
            obscene = False
            if 'обсц' in w["analysis"][0]['gr'] or w['text'].lower() in obscene_words:
                obscene = True
            if pos in mapping:
                tagged.append([lemma + '_' + mapping[pos], w['text'], obscene, morph.parse(w['text'])[0].tag])
            else:
                tagged.append([lemma + '_X', w['text'], obscene, None])
        except KeyError:
            tagged.append([None, w['text'], False, None])
            continue
    return tagged


MODEL = 'tayga_upos_skipgram_300_2_2019'
FORMAT = 'csv'


def api_neighbor(w):
    neighbors = {}
    url = '/'.join(['http://rusvectores.org', MODEL, w, 'api', FORMAT]) + '/'
    r = requests.get(url=url, stream=True)
    for line in r.text.split('\n'):
        try:  # РїРµСЂРІС‹Рµ РґРІРµ СЃС‚СЂРѕРєРё РІ С„Р°Р№Р»Рµ -- СЃР»СѓР¶РµР±РЅС‹Рµ, РёС… РјС‹ РїСЂРѕРїСѓСЃС‚РёРј
            word, sim = re.split('\s+', line)  # СЂР°Р·Р±РёРІР°РµРј СЃС‚СЂРѕРєСѓ РїРѕ РѕРґРЅРѕРјСѓ РёР»Рё Р±РѕР»РµРµ РїСЂРѕР±РµР»Р°Рј
            neighbors[word] = sim
        except:
            continue
    return neighbors

# РҐСѓР№РЅСЏ, С…СѓР№ Рё РјС‹Р»Рѕ. Р“СЂСЏРґС‘С‚ РїРёР·РґРµС†... Р“РѕС‚РѕРІСЊСЃСЏ Рє РїРёР·РґРµС†Сѓ! РџРёР·РґСѓР№ РѕС‚СЃСЋРґР°, РјСѓРґРёР»Р°. Р‘Р»СЏС‚СЊ, РµР±Р°Р» С‚РµР±СЏ РІ СЂРѕС‚!!!
processed_mystem = tag_mystem(text="привет всем")
output_text = ''
indices = []
for index, triple in enumerate(processed_mystem):
    if triple[2]:
        indices.append(index)
        most_similar_variants = list(api_neighbor(triple[0]).keys())
        variants = ' '.join(most_similar_variants)
        processed = m.analyze(variants)
        was_correct = False
        while not was_correct:
            for w in processed:
                if 'analysis' not in w or not w['analysis'] or 'gr' not in w['analysis'][0] or \
                        len(w['text']) < 3 or '*' in w['text']:
                    continue
                if 'обсц' not in w['analysis'][0]["gr"] and w['text'] not in obscene_words:
                    was_correct = True
                    parsed_text = morph.parse(w['text'])[0].inflect(set(str(triple[3]).split()[-1].split(',')))
                    if hasattr(parsed_text, 'word'):
                        w['text'] = parsed_text.word
                    if re.match('[А-ЯЁ]+$', triple[1]):
                        output_text += w['text'].upper()
                    elif re.match('[А-ЯЁ][а-яё]+$', triple[1]):
                        output_text += w['text'].capitalize()
                    else:
                        output_text += w['text'].lower()
                    break
            if not was_correct:
                if not variants:
                    output_text += '*' * len(triple[1])
                    break
                most_similar_variants = list(api_neighbor(variants.split()[0]).keys())
                variants = variants[1:]
                processed = m.analyze(' '.join(most_similar_variants))
    else:
        output_text += triple[1]

print(output_text, [index // 2 for index in indices])
