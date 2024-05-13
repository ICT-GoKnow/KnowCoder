import os
#import spacy

from utils import read_jsonl_file, dump_jsonl_file

nlp = spacy.load("/home/bingxing2/home/scx6592/corpus/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0/en_core_web_sm/en_core_web_sm-2.3.0")


def find_head(arg_start, arg_end, doc):
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <= arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head
            break
        else:
            cur_i = doc[cur_i].head.i
    return (cur_i, cur_i)


def find_arg_span(arg_words, context_words, trigger_start=-1, trigger_end=-1, doc=None):
    match = None
    arg_len = len(arg_words)
    min_dis = len(context_words)  # minimum distance to trigger
    for i, w in enumerate(context_words):
        if context_words[i:i + arg_len] == arg_words:
            if trigger_start == -1:
                match = (i, i + arg_len - 1)
                break
            else:
                if i < trigger_start:
                    dis = abs(trigger_start - i - arg_len)
                else:
                    dis = abs(i - trigger_end)
                if dis < min_dis:
                    match = (i, i + arg_len - 1)
                    min_dis = dis
    if match:
        assert doc is not None
        match = find_head(match[0], match[1], doc)
    return match


def find_arg_span_normalized(entity_text, context_words, trigger_start=1, trigger_end=-1):
    lowercased_context_words = [w.lower() for w in context_words]
    lowercased_doc = nlp(' '.join(lowercased_context_words))
    normalized_entity_text = []
    for word in entity_text:
        word = word.lower()
        # process hyphenated words
        if "-" in word and len(word) > 1:
            normalized_entity_text.extend(word.replace("-", " - ").split())
        else:
            normalized_entity_text.append(word)
        # TODO: If we really want higher performance on ACE05,
        # we could fix U.S. -> U.S, british -> british, etc.
    match = find_arg_span(normalized_entity_text,
                          lowercased_context_words,
                          trigger_start,
                          trigger_end,
                          doc=lowercased_doc)
    return match


def get_tokens_from_text(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        tokens.append(token.text)
    return tokens


def clean_span(tokens, span):
    if tokens[span[0]].lower() in {'the', 'an', 'a'}:
        if span[0] != span[1]:
            return (span[0] + 1, span[1])
    return span


def get_head(sentence, text, trigger_start=-1, trigger_end=-1, span=None):
    context_words = get_tokens_from_text(sentence)
    doc = nlp(sentence)

    #if span:
    #    span = (span[0], span[1] - 1)
    #    span = clean_span(context_words, span)
    #    if span[0] != span[1]:
    #        match = find_head(span[0], span[1], doc=doc)
    #    else:
    #        match = (span[0], span[1])
    #    return match
#
    text_words = get_tokens_from_text(text)
    match = find_arg_span(text_words, context_words, trigger_start, trigger_end, doc=doc)
    if not match:
        match = find_arg_span_normalized(text_words, context_words, trigger_start, trigger_end)
    return match


def get_tokens_idxs_from_text(text):
    doc = nlp(text)
    tokens = []
    idxs = []
    for token in doc:
        tokens.append(token.text)
        idxs.append(token.idx)
    return tokens, idxs


def gen_label(preds_tuple, golds_tuple):
    pred_labels = {}
    gold_labels = {}
    for sentence_id in golds_tuple:
        pred_labels[sentence_id] = []
        gold_labels[sentence_id] = []
        pred_sentence_mentions = preds_tuple[sentence_id]
        gold_sentence_mentions = golds_tuple[sentence_id]
        for mention in pred_sentence_mentions:
            if mention in gold_sentence_mentions:
                pred_labels[sentence_id].append("TP")
            else:
                pred_labels[sentence_id].append("FP")
        for mention in gold_sentence_mentions:
            if mention in pred_sentence_mentions:
                gold_labels[sentence_id].append("--")
            else:
                gold_labels[sentence_id].append("FN")
    return pred_labels, gold_labels


def output_label(pred_labels, gold_labels, path):
    pred_records = read_jsonl_file(os.path.join(path, "prediction.json"))
    gold_records = read_jsonl_file(os.path.join(path, "label.json"))
    for line in pred_records:
        idx = line['id']
        labels = pred_labels[idx]
        for idx, label in enumerate(labels):
            line['relations'][idx]["label"] = label
    for line in gold_records:
        idx = line['id']
        labels = gold_labels[idx]
        for idx, label in enumerate(labels):
            line['relations'][idx]["label"] = label
    dump_jsonl_file(pred_records, os.path.join(path, "prediction.json"))
    dump_jsonl_file(gold_records, os.path.join(path, "label.json"))


def judge_match(pred_mention, gold_mentions):
    for gold_mention in gold_mentions:
        if pred_mention[1:] == gold_mention[1:]:
            for type in pred_mention[0]:
                if type == gold_mention[0]:
                    return True
    return False


def deduplication(data):
    de_data = []
    for d in data:
        if d not in de_data:
            de_data.append(d)
    return de_data


def score(preds_tuple, golds_tuple):
    '''
    Modified from https://github.com/xinyadu/eeqa
    '''
    gold_mention_n, pred_mention_n, true_positive_n, sentence_n = 0, 0, 0, 0
    for sentence_id in golds_tuple:
        sentence_n += 1
        gold_sentence_mentions = deduplication(golds_tuple[sentence_id])
        pred_sentence_mentions = deduplication(preds_tuple[sentence_id])
        gold_sentence_mentions = golds_tuple[sentence_id]
        pred_sentence_mentions = preds_tuple[sentence_id]
        for mention in gold_sentence_mentions:
            gold_mention_n += 1
        for mention in pred_sentence_mentions:
            pred_mention_n += 1
            flag = judge_match(mention, gold_sentence_mentions) if isinstance(mention[0], list) else mention in gold_sentence_mentions
            if flag:
                true_positive_n += 1
    prec_c = true_positive_n / pred_mention_n if pred_mention_n != 0 else 0
    recall_c = true_positive_n / gold_mention_n if gold_mention_n != 0 else 0
    f1_c = 2 * prec_c * recall_c / (prec_c + recall_c) if prec_c or recall_c else 0
    return prec_c, recall_c, f1_c


def score_by_type(preds_tuple, golds_tuple):
    '''
    Modified from https://github.com/xinyadu/eeqa
    '''
    gold_mention_n, pred_mention_n, true_positive_n, result, sentence_n = {}, {}, {}, {}, 0
    for sentence_id in golds_tuple:
        sentence_n += 1
        gold_sentence_mentions = deduplication(golds_tuple[sentence_id])
        pred_sentence_mentions = deduplication(preds_tuple[sentence_id])
        for mention in gold_sentence_mentions:
            mention_type = mention[0]
            if mention_type not in gold_mention_n:
                gold_mention_n[mention_type] = 0
                true_positive_n[mention_type] = 0
            gold_mention_n[mention_type] += 1
        for mention in pred_sentence_mentions:
            mention_type = mention[0][-1] if isinstance(mention[0], list) else mention[0]
            if mention_type not in gold_mention_n:
                continue          
            if mention_type not in pred_mention_n:
                pred_mention_n[mention_type] = 0
                #true_positive_n[mention_type] = 0
            pred_mention_n[mention_type] += 1
            flag = judge_match(mention, gold_sentence_mentions) if isinstance(mention[0], list) else mention in gold_sentence_mentions
            if flag:
                true_positive_n[mention_type] += 1
    for tp_type, tp_n in true_positive_n.items():
        pred_n = pred_mention_n.get(tp_type)
        gold_n = gold_mention_n.get(tp_type)
        prec_c = tp_n / pred_n if pred_n else 0
        recall_c = tp_n / gold_n if gold_n else 0
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c) if prec_c or recall_c else 0
        result[tp_type] = (prec_c, recall_c, f1_c)
    return result
def score_by_type_span(preds_tuple, golds_tuple):
    '''
    Modified from https://github.com/xinyadu/eeqa
    '''
    gold_mention_n, pred_mention_n, true_positive_n, result, sentence_n = {}, {}, {}, {}, 0
    for sentence_id in golds_tuple:
        sentence_n += 1
        gold_sentence_mentions = deduplication(golds_tuple[sentence_id])
        pred_sentence_mentions = deduplication(preds_tuple[sentence_id])
        for mention in gold_sentence_mentions:
            mention_type = mention[0]
            if mention_type not in gold_mention_n:
                gold_mention_n[mention_type] = 0
                true_positive_n[mention_type] = 0
            gold_mention_n[mention_type] += 1
        for mention in pred_sentence_mentions:
            mention_type = mention[0][-1] if isinstance(mention[0], list) else mention[0]
            #if mention_type not in gold_mention_n:
            #    continue          
            if mention_type not in pred_mention_n:
                pred_mention_n[mention_type] = 0
                #true_positive_n[mention_type] = 0
            
            gold_span=[]
            for item in gold_sentence_mentions:
                gold_span.append(item[1])
            pred_mention_n[mention_type] += 1
            flag = mention[1] in gold_span
            if flag:
                if mention_type not in true_positive_n:
                    true_positive_n[mention_type]=0
                    true_positive_n[mention_type] += 1
                else:
                    true_positive_n[mention_type] += 1
    for tp_type, tp_n in true_positive_n.items():
        pred_n = pred_mention_n.get(tp_type)
        gold_n = gold_mention_n.get(tp_type)
        prec_c = tp_n / pred_n if pred_n else 0
        recall_c = tp_n / gold_n if gold_n else 0
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c) if prec_c or recall_c else 0
        result[tp_type] = (prec_c, recall_c, f1_c)
    return result

def score_by_source(preds_tuple, golds_tuple, source_dict):
    '''
    Modified from https://github.com/xinyadu/eeqa
    '''
    gold_mention_n, pred_mention_n, true_positive_n, result, sentence_n = {}, {}, {}, {}, 0
    for sentence_id in golds_tuple:
        sentence_n += 1
        source = source_dict[sentence_id]
        if source not in gold_mention_n:
            gold_mention_n[source] = 0
            pred_mention_n[source] = 0
            true_positive_n[source] = 0
        
        gold_sentence_mentions = golds_tuple[sentence_id]
        pred_sentence_mentions = preds_tuple[sentence_id]
        #if gold_sentence_mentions==pred_sentence_mentions==[]:
        #    gold_mention_n[source] += 1
        #    pred_mention_n[source] += 1
        #    true_positive_n[source] += 1
        for mention in gold_sentence_mentions:
            gold_mention_n[source] += 1
        for mention in pred_sentence_mentions:
            pred_mention_n[source] += 1
            flag = judge_match(mention, gold_sentence_mentions) if isinstance(mention[0], list) else mention in gold_sentence_mentions
            if flag:
                true_positive_n[source] += 1

    for src, tp_n in true_positive_n.items():
        pred_n = pred_mention_n[src]
        gold_n = gold_mention_n[src]
        prec_c = tp_n / pred_n if pred_n else 0
        recall_c = tp_n / gold_n if gold_n else 0
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c) if prec_c or recall_c else 0
        result[src] = (prec_c, recall_c, f1_c)
    return result
def score_by_source_type(preds_tuple, golds_tuple, source_dict):
    '''
    Modified from https://github.com/xinyadu/eeqa
    '''
    gold_mention_n, pred_mention_n, true_positive_n, result, sentence_n = {}, {}, {}, {}, 0
    for sentence_id in golds_tuple:
        sentence_n += 1
        source = source_dict[sentence_id]
        #if source not in gold_mention_n:
        #    gold_mention_n[source] = {}
        #    pred_mention_n[source] = {}
        #    true_positive_n[source] = {}
        #print(golds_tuple[sentence_id])
        
        gold_sentence_mentions = deduplication(golds_tuple[sentence_id])
        pred_sentence_mentions = deduplication(preds_tuple[sentence_id])
        #print(gold_sentence_mentions)
        for mention in gold_sentence_mentions:
            mention_type = mention[0]
            
            if mention_type not in gold_mention_n.keys():
                gold_mention_n[mention_type] = {}
                true_positive_n[mention_type] = {}
                if source not in gold_mention_n[mention_type].keys():
                    gold_mention_n[mention_type][source] = 0
                    true_positive_n[mention_type][source] = 0
            else:
                if source not in gold_mention_n[mention_type].keys():
                    gold_mention_n[mention_type][source] = 0
                    true_positive_n[mention_type][source] = 0
            gold_mention_n[mention_type][source] += 1
        for mention in pred_sentence_mentions:
            mention_type = mention[0][-1] if isinstance(mention[0], list) else mention[0]
            if mention_type not in pred_mention_n.keys():
                pred_mention_n[mention_type] = {}
                true_positive_n[mention_type] = {}
                if source not in pred_mention_n[mention_type].keys():
                    #gold_mention_n[mention_type][source] = 0
                    pred_mention_n[mention_type][source] = 0
                    true_positive_n[mention_type][source] = 0
            else:
                if source not in pred_mention_n[mention_type].keys():
                    #gold_mention_n[mention_type][source] = 0
                    pred_mention_n[mention_type][source] = 0
                    true_positive_n[mention_type][source] = 0
            pred_mention_n[mention_type][source] += 1
            
            flag = judge_match(mention, gold_sentence_mentions) if isinstance(mention[0], list) else mention in gold_sentence_mentions
            if flag:
                true_positive_n[mention_type][source] += 1
    for tp_type, tp_n in true_positive_n.items():
        
        #print(tp_n)
        result[tp_type]={}
        for tp_source,tp_m in tp_n.items():
            if tp_type in pred_mention_n.keys():
                pred_n = pred_mention_n[tp_type]
                if tp_source in pred_n.keys():
                    pred_n=pred_n[tp_source]
                else:
                    pred_n=None
            else:
                pred_n=None
            #pred_n = pred_mention_n.get(tp_type)
            #pred_n = pred_n.get(tp_source)
            if tp_type in gold_mention_n.keys():
                gold_n = gold_mention_n[tp_type]
                if tp_source in gold_n.keys():
                    gold_n=gold_n[tp_source]
                else:
                    gold_n=None
            else:
                gold_n=None
            #print(gold_mention_n)
            #print(true_positive_n)
            #print(tp_source)
            ##print(gold_n)
            #gold_n = gold_n.get(tp_source)
            prec_c = tp_m / pred_n if pred_n else 0
            recall_c = tp_m / gold_n if gold_n else 0
            f1_c = 2 * prec_c * recall_c / (prec_c + recall_c) if prec_c or recall_c else 0
            #print(result[tp_type].keys())
            
            result[tp_type][tp_source] = (prec_c, recall_c, f1_c)
            
    return result

def gen_tuples_type(record):
    result = {}
    if record:
        for idx, tuple_list in record.items():
            result[idx] = []
            for tuple in tuple_list:
                result[idx].append(tuple[0])
    else:
        return result


# NER tuples generation
def gen_tuples_ei_ec(record, match_type):
    if record:
        ei, ec = [], []
        sentence = record[0]
        for entity in record[1]:
            typ = entity['type']
            if match_type == 'HM':
                match = get_head(sentence, entity['word'])
            else:
                match = entity['word']
            if not match:
                continue
            ei_one = (match)
            ec_one = (typ, match)
            ei.append(ei_one)
            ec.append(ec_one)
        return ei, ec
    else:
        return [], []


# RE tuples generation
def gen_tuples_ri_rc(record, match_type):
    if record:
        ri, rc = [], []
        sentence = record[0]
        for relation in record[1]:
            typ = relation['type']
            if match_type == 'HM':
                match1 = get_head(sentence, relation['subject']['word'])
                match2 = get_head(sentence, relation['object']['word'])
            else:
                match1 = relation['subject']['word']
                match2 = relation['object']['word']
            if not match1 or not match2:
                continue
            ri_one = (match1, match2)
            rc_one = (typ, match1, match2)
            ri.append(ri_one)
            rc.append(rc_one)
        return ri, rc
    else:
        return [], []


# EE tuples generation
def gen_tuples_ti_tc_ai_ac(record, match_type):
    if record:
        ti, tc, ai, ac = [], [], [], []
        sentence = record[0]
        for event in record[1]:
            typ = event['type']
            if match_type == 'HM':
                match1 = get_head(sentence, event['trigger']['word'])
            else:
                match1 = event['trigger']['word']
            if not match1:
                continue
            ti_one = (match1)
            tc_one = (typ, match1)
            ti.append(ti_one)
            tc.append(tc_one)

            for arg_role in event['args']:
                for arg_role_one in event['args'][arg_role]:
                    if match_type == 'HM':
                        match2 = get_head(sentence, arg_role_one['word'])
                    else:
                        match2 = arg_role_one['word']
                    if not match2:
                        continue
                    ai_one = (typ, match2)
                    ac_one = (typ, match2, arg_role)
                    ai.append(ai_one)
                    ac.append(ac_one)
        return ti, tc, ai, ac
    else:
        return [], [], [], []


# EAE tuples generation
def gen_tuples_ai_ac(record, match_type):
    if record:
        ai, ac = [], []
        sentence = record[0]
        for event in record[1]:
            typ = event['type']
            for arg_role in event['args']:
                for arg_role_one in event['args'][arg_role]:
                    if match_type == 'HM':
                        #trigger_start = event['trigger']['word'][0]
                        #trigger_end = event['trigger']['word'][1]
                        match = get_head(sentence,
                                         arg_role_one['word'],
                                         #trigger_start,
                                         #trigger_end,
                                         span=arg_role_one.get('word')
                                         )
                    else:
                        match = arg_role_one['word']
                    if not match:
                        continue
                    ai_one = (typ, match)
                    ac_one = (typ, match, arg_role)
                    ai.append(ai_one)
                    ac.append(ac_one)
        return ai, ac
    else:
        return [], []


# EE scores calculation
def cal_scores_ti_tc_ai_ac(preds, golds, match_type, granularity, source_dict=None):
    '''
    :param preds: {id: [{type:'', 'trigger':{'span':[], 'word':[]}, 'args':[role1:[], role2:[], ...}, ...]}
    :param golds:
    :return:
    '''
    tuples_pred = [{}, {}, {}, {}]  # ti, tc, ai, ac
    tuples_gold = [{}, {}, {}, {}]  # ti, tc, ai, ac
    for idx in golds:
        pred = preds[idx] if idx in preds else None
        gold = golds[idx]
        tuples_pred[0][idx], tuples_pred[1][idx], tuples_pred[2][idx], tuples_pred[3][idx] = gen_tuples_ti_tc_ai_ac(pred, match_type)
        tuples_gold[0][idx], tuples_gold[1][idx], tuples_gold[2][idx], tuples_gold[3][idx] = gen_tuples_ti_tc_ai_ac(gold, match_type)
    if granularity == 'overall':
        prf_s = []
        for i in range(4):
            prf = score(tuples_pred[i], tuples_gold[i])
            prf_s.append(prf)
        return prf_s
    elif granularity == 'type':
        prf_s = {}
        prf1 = score_by_type(tuples_pred[1], tuples_gold[1])
        prf2 = score_by_type(tuples_pred[2], tuples_gold[2])
        prf3 = score_by_type(tuples_pred[3], tuples_gold[3])
        for type, prf in prf1.items():
            prf_s[type] = [prf1[type]]
        for type, prf in prf2.items():
            if type not in prf_s:
                prf_s[type] = [(0, 0, 0)]
            prf_s[type].append(prf2[type])
        for type, prf in prf3.items():
            if type not in prf_s:
                prf_s[type] = [(0, 0, 0), (0, 0, 0)]
            prf_s[type].append(prf3[type])
        return prf_s
    elif granularity == 'source':
        prf_s = {}
        prf1 = score_by_source(tuples_pred[0], tuples_gold[0], source_dict)
        prf2 = score_by_source(tuples_pred[1], tuples_gold[1], source_dict)
        prf3 = score_by_source(tuples_pred[2], tuples_gold[2], source_dict)
        prf4 = score_by_source(tuples_pred[3], tuples_gold[3], source_dict)
        for source, prf in prf1.items():
            prf_s[source] = [prf1[source], prf2[source], prf3[source], prf4[source]]
        return prf_s
    elif granularity == 'source_type':
        prf_s = {}
        #prf1 = score_by_source_type(tuples_pred[0], tuples_gold[0], source_dict)
        prf2 = score_by_source_type(tuples_pred[1], tuples_gold[1], source_dict)
        for type,prfs in prf2.items():
            prf_s[type]={}
            for source, prf in prfs.items():

                prf_s[type][source] = [prf2[type][source], prf2[type][source]]
        return prf_s


# EAE scores calculation
def cal_scores_ai_ac(preds, golds, match_type, granularity, source_dict=None):
    '''
    :param preds: {id: [{type:'', 'trigger':{'span':[], 'word':[]}, 'args':[role1:[], role2:[], ...}, ...]}
    :param golds:
    :return:
    '''
    tuples_pred = [{}, {}]  # ai, ac
    tuples_gold = [{}, {}]  # ai, ac
    for idx in golds:
        pred = preds[idx] if idx in preds else None
        gold = golds[idx]
        tuples_pred[0][idx], tuples_pred[1][idx] = gen_tuples_ai_ac(pred, match_type)
        tuples_gold[0][idx], tuples_gold[1][idx] = gen_tuples_ai_ac(gold, match_type)

    if granularity == 'overall':
        prf_s = []
        for i in range(2):
            prf = score(tuples_pred[i], tuples_gold[i])
            prf_s.append(prf)
        return prf_s
    elif granularity == 'type':
        prf_s = {}
        prf1 = score_by_type(tuples_pred[1], tuples_gold[1])
        for type, prf in prf1.items():
            prf_s[type] = [prf1[type]]
        return prf_s
    elif granularity == 'source':
        prf_s = {}
        prf1 = score_by_source(tuples_pred[0], tuples_gold[0], source_dict)
        prf2 = score_by_source(tuples_pred[1], tuples_gold[1], source_dict)
        for source, prf in prf1.items():
            prf_s[source] = [prf1[source], prf2[source]]
        return prf_s
    elif granularity == 'source_type':
        prf_s = {}
        #prf1 = score_by_source_type(tuples_pred[0], tuples_gold[0], source_dict)
        prf2 = score_by_source_type(tuples_pred[1], tuples_gold[1], source_dict)
        for type,prfs in prf2.items():
            prf_s[type]={}
            for source, prf in prfs.items():

                prf_s[type][source] = [prf2[type][source], prf2[type][source]]
        return prf_s


# NER scores calculation
def cal_scores_ei_ec(preds, golds, match_type, granularity, source_dict=None):
    '''
    :param preds: {id: [{type:'', 'word':''}, ...]}
    :param golds:
    :return:
    '''
    tuples_pred = [{}, {}]  # ei, ec
    tuples_gold = [{}, {}]  # ei, ec
    for idx in golds:
        pred = preds[idx] if idx in preds else None
        gold = golds[idx]
        tuples_pred[0][idx], tuples_pred[1][idx] = gen_tuples_ei_ec(pred, match_type)
        tuples_gold[0][idx], tuples_gold[1][idx] = gen_tuples_ei_ec(gold, match_type)

    if granularity == 'overall':
        prf_s = []
        for i in range(2):
            prf = score(tuples_pred[i], tuples_gold[i])
            prf_s.append(prf)
        return prf_s
    elif granularity == 'type':
        prf_s = {}
        prf0 = score_by_type_span(tuples_pred[1], tuples_gold[1])
        prf1 = score_by_type(tuples_pred[1], tuples_gold[1])
        
        for type, prf in prf1.items():
            prf_s[type] = [prf0[type],prf1[type]]
        return prf_s
    elif granularity == 'source':
        prf_s = {}
        prf1 = score_by_source(tuples_pred[0], tuples_gold[0], source_dict)
        prf2 = score_by_source(tuples_pred[1], tuples_gold[1], source_dict)
        for source, prf in prf1.items():
            prf_s[source] = [prf1[source], prf2[source]]
        return prf_s
    elif granularity == 'source_type':
        prf_s = {}
        #prf1 = score_by_source_type(tuples_pred[0], tuples_gold[0], source_dict)
        prf2 = score_by_source_type(tuples_pred[1], tuples_gold[1], source_dict)
        for type,prfs in prf2.items():
            prf_s[type]={}
            for source, prf in prfs.items():

                prf_s[type][source] = [prf2[type][source], prf2[type][source]]
        return prf_s


# RE scores calculation
def cal_scores_ri_rc(preds, golds, match_type, granularity, source_dict=None,type_dict=None):
    '''
    :param preds: {id: [{type:'', 'subject':'', 'object':''}, ...]}
    :param golds:
    :return:
    '''
    tuples_pred = [{}, {}]  # ri, rc
    tuples_gold = [{}, {}]  # ri, rc
    for idx in golds:
        pred = preds[idx] if idx in preds else None
        gold = golds[idx]
        tuples_pred[0][idx], tuples_pred[1][idx] = gen_tuples_ri_rc(pred, match_type)
        tuples_gold[0][idx], tuples_gold[1][idx] = gen_tuples_ri_rc(gold, match_type)
    if granularity == 'overall':
        prf_s = []
        for i in range(2):
            prf = score(tuples_pred[i], tuples_gold[i])
            prf_s.append(prf)
        return prf_s
    elif granularity == 'type':
        prf_s = {}
        prf1 = score_by_type(tuples_pred[1], tuples_gold[1])
        for type, prf in prf1.items():
            prf_s[type] = [prf1[type]]
        return prf_s
    elif granularity == 'source':
        prf_s = {}
        prf1 = score_by_source(tuples_pred[0], tuples_gold[0], source_dict)
        prf2 = score_by_source(tuples_pred[1], tuples_gold[1], source_dict)
        for source, prf in prf1.items():
            prf_s[source] = [prf1[source], prf2[source]]
        return prf_s
    elif granularity == 'source_type':
        prf_s = {}
        prf2 = score_by_source_type(tuples_pred[1], tuples_gold[1], source_dict)
        for type,prfs in prf2.items():
            prf_s[type]={}
            for source, prf in prfs.items():
                prf_s[type][source] = [prf2[type][source], prf2[type][source]]
        return prf_s
    


def gen_idx_events_dict(path='', events_list=[]):
    if events_list:
        records = events_list
    else:
        records = read_jsonl_file(path)
    data_dict = {}
    for line in records:
        idx = line['id']
        events = line['events']
        content = line['content'].replace("**", "")
        data_dict[idx] = (content, events)
    return data_dict


def gen_idx_entities_dict(path='', entities_list=[]):
    if entities_list:
        records = entities_list
    else:
        records = read_jsonl_file(path)
    data_dict = {}
    for line in records:
        idx = line['id']
        entities = line['entities']
        content = line['content']
        data_dict[idx] = (content, entities)
    return data_dict


def gen_idx_relations_dict(path='', relations_list=[]):
    if relations_list:
        records = relations_list
    else:
        records = read_jsonl_file(path)
    data_dict = {}
    for line in records:
        idx = line['id']
        relations = line['relations']
        content = line['content']
        data_dict[idx] = (content, relations)
    return data_dict
