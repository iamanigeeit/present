import re

O_TO_UO, O_TO_UO_SUB = re.compile(r'^([bpmf])o(?!u)'), r'\1uo'
U_TO_V, U_TO_V_SUB = re.compile(r'^([jqxy])u'), r'\1v'
I_TO_IH, I_TO_IH_SUB = re.compile(r'^([zcs]h?|r)i'), r'\1ɨ'
AN_TO_EN, AN_TO_EN_SUB = re.compile(r'([iv])an(?!g)'), r'\1en'


NULL_INITIAL = 'ʔ'
NULL_FINAL = 'ɨ'

INITIALS = [
    NULL_INITIAL,
    'b', 'p', 'm', 'f',
    'd', 't', 'n', 'l',
    'g', 'k', 'h',
    'j', 'q', 'x',
    'zh', 'ch', 'sh', 'r',
    'z', 'c', 's',
    'y', 'w'
]

RIMES = [
    NULL_FINAL,
    'a', 'an', 'ang', 'ai', 'ao',
    'e', 'en', 'eng', 'ei', 'er',
    'i', 'in', 'ing', 'ia', 'iang', 'iao', 'ie', 'ien', 'iong', 'iu',
    'o', 'ong', 'ou',
    'u', 'un', 'ua', 'uan', 'uang', 'uai', 'ui', 'uo',
    'v', 'vn', 've', 'ven',
]

VALID_PINYIN_REGEX = re.compile(
    rf"({'|'.join(INITIALS)})({'|'.join(RIMES)})"
)
VALID_PINYINS = [
    'zhɨ', 'chɨ', 'shɨ', 'rɨ', 'zɨ', 'cɨ', 'sɨ',

    'ʔa', 'ba', 'pa', 'ma', 'fa', 'da', 'ta', 'na', 'la', 'ga', 'ka', 'ha', 'zha', 'cha', 'sha', 'za', 'ca', 'sa', 'ya', 'wa',
    'ʔan', 'ban', 'pan', 'man', 'fan', 'dan', 'tan', 'nan', 'lan', 'gan', 'kan', 'han', 'zhan', 'chan', 'shan', 'ran', 'zan', 'can', 'san', 'wan',
    'ʔang', 'bang', 'pang', 'mang', 'fang', 'dang', 'tang', 'nang', 'lang', 'gang', 'kang', 'hang', 'zhang', 'chang', 'shang', 'rang', 'zang', 'cang', 'sang', 'yang', 'wang',
    'ʔai', 'bai', 'pai', 'mai', 'dai', 'tai', 'nai', 'lai', 'gai', 'kai', 'hai', 'zhai', 'chai', 'shai', 'zai', 'cai', 'sai', 'yai', 'wai',
    'ʔao', 'bao', 'pao', 'mao', 'dao', 'tao', 'nao', 'lao', 'gao', 'kao', 'hao', 'zhao', 'chao', 'shao', 'rao', 'zao', 'cao', 'sao',

    'ʔe', 'me', 'de', 'te', 'ne', 'le', 'ge', 'ke', 'he', 'zhe', 'che', 'she', 're', 'ze', 'ce', 'se',
    'ʔen', 'ben', 'pen', 'men', 'fen', 'den', 'nen', 'gen', 'ken', 'hen', 'zhen', 'chen', 'shen', 'ren', 'zen', 'cen', 'sen', 'wen',
    'ʔeng', 'beng', 'peng', 'meng', 'feng', 'deng', 'teng', 'neng', 'leng', 'geng', 'keng', 'heng', 'zheng', 'cheng', 'sheng', 'reng', 'zeng', 'ceng', 'seng', 'weng',
    'ʔei', 'bei', 'pei', 'mei', 'fei', 'dei', 'tei', 'nei', 'lei', 'gei', 'kei', 'hei', 'zhei', 'shei', 'zei', 'cei', 'sei', 'wei',
    'ʔer',

    'bi', 'pi', 'mi', 'di', 'ti', 'ni', 'li', 'ji', 'qi', 'xi', 'yi',
    'pia', 'dia', 'nia', 'lia', 'jia', 'qia', 'xia',
    'bien', 'pien', 'mien', 'dien', 'tien', 'nien', 'lien', 'jien', 'qien', 'xien', 'yen',
    'bie', 'pie', 'mie', 'die', 'tie', 'nie', 'lie', 'jie', 'qie', 'xie', 'ye',
    'biao', 'piao', 'miao', 'fiao', 'diao', 'tiao', 'niao', 'liao', 'jiao', 'qiao', 'xiao', 'yao',
    'miu', 'diu', 'niu', 'liu', 'kiu', 'jiu', 'qiu', 'xiu',

    'bin', 'pin', 'min', 'nin', 'lin', 'jin', 'qin', 'xin', 'yin',
    'biang', 'diang', 'niang', 'liang', 'jiang', 'qiang', 'xiang',
    'bing', 'ping', 'ming', 'ding', 'ting', 'ning', 'ling', 'jing', 'qing', 'xing', 'ying',
    'jiong', 'qiong', 'xiong',

    'ʔo', 'wo',
    'ʔong', 'dong', 'tong', 'nong', 'long', 'gong', 'kong', 'hong', 'zhong', 'chong', 'rong', 'zong', 'cong', 'song', 'yong',
    'ʔou', 'pou', 'mou', 'fou', 'dou', 'tou', 'nou', 'lou', 'gou', 'kou', 'hou', 'zhou', 'chou', 'shou', 'rou', 'zou', 'cou', 'sou', 'you',

    'bu', 'pu', 'mu', 'fu', 'du', 'tu', 'nu', 'lu', 'gu', 'ku', 'hu', 'zhu', 'chu', 'shu', 'ru', 'zu', 'cu', 'su', 'wu',
    'gua', 'kua', 'hua', 'zhua', 'chua', 'shua', 'rua',
    'buo', 'puo', 'muo', 'fuo', 'duo', 'tuo', 'nuo', 'luo', 'guo', 'kuo', 'huo', 'zhuo', 'chuo', 'shuo', 'ruo', 'zuo', 'cuo', 'suo',
    'guai', 'kuai', 'huai', 'zhuai', 'chuai', 'shuai',

    'dui', 'tui', 'gui', 'kui', 'hui', 'zhui', 'chui', 'shui', 'rui', 'zui', 'cui', 'sui',
    'duan', 'tuan', 'nuan', 'luan', 'guan', 'kuan', 'huan', 'zhuan', 'chuan', 'shuan', 'ruan', 'zuan', 'cuan', 'suan',
    'dun', 'tun', 'nun', 'lun', 'gun', 'kun', 'hun', 'zhun', 'chun', 'shun', 'run', 'zun', 'cun', 'sun',
    'duang', 'guang', 'kuang', 'huang', 'zhuang', 'chuang', 'shuang',

    'nv', 'lv', 'jv', 'qv', 'xv', 'yv',
    'nve', 'lve', 'jve', 'qve', 'xve', 'yve',
    'lven', 'jven', 'qven', 'xven', 'yven',
    'lvn', 'jvn', 'qvn', 'xvn', 'yvn',
]


INITIAL_TO_PINYINS = {
    'ʔ': ['ʔa', 'ʔan', 'ʔang', 'ʔai', 'ʔao', 'ʔe', 'ʔen', 'ʔeng', 'ʔei', 'ʔer', 'ʔo', 'ʔong', 'ʔou'],
    'b': ['ba', 'ban', 'bang', 'bai', 'bao', 'ben', 'beng', 'bei', 'bi', 'bien', 'bie', 'biao', 'bin', 'biang', 'bing', 'bu', 'buo'],
    'p': ['pa', 'pan', 'pang', 'pai', 'pao', 'pen', 'peng', 'pei', 'pi', 'pia', 'pien', 'pie', 'piao', 'pin', 'ping', 'pou', 'pu', 'puo'],
    'm': ['ma', 'man', 'mang', 'mai', 'mao', 'me', 'men', 'meng', 'mei', 'mi', 'mien', 'mie', 'miao', 'miu', 'min', 'ming', 'mou', 'mu', 'muo'],
    'f': ['fa', 'fan', 'fang', 'fen', 'feng', 'fei', 'fiao', 'fou', 'fu', 'fuo'],
    'd': ['da', 'dan', 'dang', 'dai', 'dao', 'de', 'den', 'deng', 'dei', 'di', 'dia', 'dien', 'die', 'diao', 'diu', 'diang', 'ding', 'dong', 'dou', 'du', 'duo', 'dui', 'duan', 'dun', 'duang'],
    't': ['ta', 'tan', 'tang', 'tai', 'tao', 'te', 'teng', 'tei', 'ti', 'tien', 'tie', 'tiao', 'ting', 'tong', 'tou', 'tu', 'tuo', 'tui', 'tuan', 'tun'],
    'n': ['na', 'nan', 'nang', 'nai', 'nao', 'ne', 'nen', 'neng', 'nei', 'ni', 'nia', 'nien', 'nie', 'niao', 'niu', 'nin', 'niang', 'ning', 'nong', 'nou', 'nu', 'nuo', 'nuan', 'nun', 'nv', 'nve'],
    'l': ['la', 'lan', 'lang', 'lai', 'lao', 'le', 'leng', 'lei', 'li', 'lia', 'lien', 'lie', 'liao', 'liu', 'lin', 'liang', 'ling', 'long', 'lou', 'lu', 'luo', 'luan', 'lun', 'lv', 'lve', 'lven', 'lvn'],
    'g': ['ga', 'gan', 'gang', 'gai', 'gao', 'ge', 'gen', 'geng', 'gei', 'gong', 'gou', 'gu', 'gua', 'guo', 'guai', 'gui', 'guan', 'gun', 'guang'],
    'k': ['ka', 'kan', 'kang', 'kai', 'kao', 'ke', 'ken', 'keng', 'kei', 'kiu', 'kong', 'kou', 'ku', 'kua', 'kuo', 'kuai', 'kui', 'kuan', 'kun', 'kuang'],
    'h': ['ha', 'han', 'hang', 'hai', 'hao', 'he', 'hen', 'heng', 'hei', 'hong', 'hou', 'hu', 'hua', 'huo', 'huai', 'hui', 'huan', 'hun', 'huang'],
    'j': ['ji', 'jia', 'jien', 'jie', 'jiao', 'jiu', 'jin', 'jiang', 'jing', 'jiong', 'jv', 'jve', 'jven', 'jvn'],
    'q': ['qi', 'qia', 'qien', 'qie', 'qiao', 'qiu', 'qin', 'qiang', 'qing', 'qiong', 'qv', 'qve', 'qven', 'qvn'],
    'x': ['xi', 'xia', 'xien', 'xie', 'xiao', 'xiu', 'xin', 'xiang', 'xing', 'xiong', 'xv', 'xve', 'xven', 'xvn'],
    'zh': ['zhɨ', 'zha', 'zhan', 'zhang', 'zhai', 'zhao', 'zhe', 'zhen', 'zheng', 'zhei', 'zhong', 'zhou', 'zhu', 'zhua', 'zhuo', 'zhuai', 'zhui', 'zhuan', 'zhun', 'zhuang'],
    'ch': ['chɨ', 'cha', 'chan', 'chang', 'chai', 'chao', 'che', 'chen', 'cheng', 'chong', 'chou', 'chu', 'chua', 'chuo', 'chuai', 'chui', 'chuan', 'chun', 'chuang'],
    'sh': ['shɨ', 'sha', 'shan', 'shang', 'shai', 'shao', 'she', 'shen', 'sheng', 'shei', 'shou', 'shu', 'shua', 'shuo', 'shuai', 'shui', 'shuan', 'shun', 'shuang'],
    'r': ['rɨ', 'ran', 'rang', 'rao', 're', 'ren', 'reng', 'rong', 'rou', 'ru', 'rua', 'ruo', 'rui', 'ruan', 'run'],
    'z': ['zɨ', 'za', 'zan', 'zang', 'zai', 'zao', 'ze', 'zen', 'zeng', 'zei', 'zong', 'zou', 'zu', 'zuo', 'zui', 'zuan', 'zun'],
    'c': ['cɨ', 'ca', 'can', 'cang', 'cai', 'cao', 'ce', 'cen', 'ceng', 'cei', 'cong', 'cou', 'cu', 'cuo', 'cui', 'cuan', 'cun'],
    's': ['sɨ', 'sa', 'san', 'sang', 'sai', 'sao', 'se', 'sen', 'seng', 'sei', 'song', 'sou', 'su', 'suo', 'sui', 'suan', 'sun'],
    'y': ['ya', 'yang', 'yai', 'yi', 'yen', 'ye', 'yao', 'yin', 'ying', 'yong', 'you', 'yv', 'yve', 'yven', 'yvn'],
    'w': ['wa', 'wan', 'wang', 'wai', 'wen', 'weng', 'wei', 'wo', 'wu'],
}

RIME_TO_PINYINS = {
    'ɨ': ['zhɨ', 'chɨ', 'shɨ', 'rɨ', 'zɨ', 'cɨ', 'sɨ'],
    'a': ['ʔa', 'ba', 'pa', 'ma', 'fa', 'da', 'ta', 'na', 'la', 'ga', 'ka', 'ha', 'zha', 'cha', 'sha', 'za', 'ca', 'sa', 'ya', 'wa'],
    'an': ['ʔan', 'ban', 'pan', 'man', 'fan', 'dan', 'tan', 'nan', 'lan', 'gan', 'kan', 'han', 'zhan', 'chan', 'shan', 'ran', 'zan', 'can', 'san', 'wan'],
    'ang': ['ʔang', 'bang', 'pang', 'mang', 'fang', 'dang', 'tang', 'nang', 'lang', 'gang', 'kang', 'hang', 'zhang', 'chang', 'shang', 'rang', 'zang', 'cang', 'sang', 'yang', 'wang'],
    'ai': ['ʔai', 'bai', 'pai', 'mai', 'dai', 'tai', 'nai', 'lai', 'gai', 'kai', 'hai', 'zhai', 'chai', 'shai', 'zai', 'cai', 'sai', 'yai', 'wai'],
    'ao': ['ʔao', 'bao', 'pao', 'mao', 'dao', 'tao', 'nao', 'lao', 'gao', 'kao', 'hao', 'zhao', 'chao', 'shao', 'rao', 'zao', 'cao', 'sao', 'yao'],
    'e': ['ʔe', 'me', 'de', 'te', 'ne', 'le', 'ge', 'ke', 'he', 'zhe', 'che', 'she', 're', 'ze', 'ce', 'se', 'ye'],
    'en': ['ʔen', 'ben', 'pen', 'men', 'fen', 'den', 'nen', 'gen', 'ken', 'hen', 'zhen', 'chen', 'shen', 'ren', 'zen', 'cen', 'sen', 'wen', 'yen'],
    'eng': ['ʔeng', 'beng', 'peng', 'meng', 'feng', 'deng', 'teng', 'neng', 'leng', 'geng', 'keng', 'heng', 'zheng', 'cheng', 'sheng', 'reng', 'zeng', 'ceng', 'seng', 'weng'],
    'ei': ['ʔei', 'bei', 'pei', 'mei', 'fei', 'dei', 'tei', 'nei', 'lei', 'gei', 'kei', 'hei', 'zhei', 'shei', 'zei', 'cei', 'sei', 'wei'],
    'er': ['ʔer'],
    'i': ['bi', 'pi', 'mi', 'di', 'ti', 'ni', 'li', 'ji', 'qi', 'xi', 'yi'],
    'ia': ['pia', 'dia', 'nia', 'lia', 'jia', 'qia', 'xia'],
    'ien': ['bien', 'pien', 'mien', 'dien', 'tien', 'nien', 'lien', 'jien', 'qien', 'xien'],
    'ie': ['bie', 'pie', 'mie', 'die', 'tie', 'nie', 'lie', 'jie', 'qie', 'xie'],
    'iao': ['biao', 'piao', 'miao', 'fiao', 'diao', 'tiao', 'niao', 'liao', 'jiao', 'qiao', 'xiao'],
    'iu': ['miu', 'diu', 'niu', 'liu', 'kiu', 'jiu', 'qiu', 'xiu'],
    'in': ['bin', 'pin', 'min', 'nin', 'lin', 'jin', 'qin', 'xin', 'yin'],
    'iang': ['biang', 'diang', 'niang', 'liang', 'jiang', 'qiang', 'xiang'],
    'ing': ['bing', 'ping', 'ming', 'ding', 'ting', 'ning', 'ling', 'jing', 'qing', 'xing', 'ying'],
    'iong': ['jiong', 'qiong', 'xiong'],
    'o': ['ʔo', 'wo'],
    'ong': ['ʔong', 'dong', 'tong', 'nong', 'long', 'gong', 'kong', 'hong', 'zhong', 'chong', 'rong', 'zong', 'cong', 'song', 'yong'],
    'ou': ['ʔou', 'pou', 'mou', 'fou', 'dou', 'tou', 'nou', 'lou', 'gou', 'kou', 'hou', 'zhou', 'chou', 'shou', 'rou', 'zou', 'cou', 'sou', 'you'],
    'u': ['bu', 'pu', 'mu', 'fu', 'du', 'tu', 'nu', 'lu', 'gu', 'ku', 'hu', 'zhu', 'chu', 'shu', 'ru', 'zu', 'cu', 'su', 'wu'],
    'ua': ['gua', 'kua', 'hua', 'zhua', 'chua', 'shua', 'rua'],
    'uo': ['buo', 'puo', 'muo', 'fuo', 'duo', 'tuo', 'nuo', 'luo', 'guo', 'kuo', 'huo', 'zhuo', 'chuo', 'shuo', 'ruo', 'zuo', 'cuo', 'suo'],
    'uai': ['guai', 'kuai', 'huai', 'zhuai', 'chuai', 'shuai'],
    'ui': ['dui', 'tui', 'gui', 'kui', 'hui', 'zhui', 'chui', 'shui', 'rui', 'zui', 'cui', 'sui'],
    'uan': ['duan', 'tuan', 'nuan', 'luan', 'guan', 'kuan', 'huan', 'zhuan', 'chuan', 'shuan', 'ruan', 'zuan', 'cuan', 'suan'],
    'un': ['dun', 'tun', 'nun', 'lun', 'gun', 'kun', 'hun', 'zhun', 'chun', 'shun', 'run', 'zun', 'cun', 'sun'],
    'uang': ['duang', 'guang', 'kuang', 'huang', 'zhuang', 'chuang', 'shuang'],
    'v': ['nv', 'lv', 'jv', 'qv', 'xv', 'yv'],
    've': ['nve', 'lve', 'jve', 'qve', 'xve', 'yve'],
    'ven': ['lven', 'jven', 'qven', 'xven', 'yven'],
    'vn': ['lvn', 'jvn', 'qvn', 'xvn', 'yvn'],
}


def regularize_pinyin(pinyin):
    pinyin = O_TO_UO.sub(O_TO_UO_SUB, pinyin)
    pinyin = U_TO_V.sub(U_TO_V_SUB, pinyin)
    pinyin = I_TO_IH.sub(I_TO_IH_SUB, pinyin)
    pinyin = AN_TO_EN.sub(AN_TO_EN_SUB, pinyin)
    if pinyin[0] in 'aoe':
        pinyin = NULL_INITIAL + pinyin
    return pinyin
