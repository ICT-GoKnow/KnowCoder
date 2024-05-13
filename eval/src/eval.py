import argparse
import time
import json

from metric import (cal_scores_ti_tc_ai_ac, cal_scores_ai_ac, cal_scores_ei_ec, cal_scores_ri_rc,
                    gen_idx_events_dict, gen_idx_entities_dict, gen_idx_relations_dict)

import convert.convert_ee as cvt_ee
import convert.convert_ner as cvt_ner
import convert.convert_re as cvt_re

from utils import gen_idx_sources_dict,gen_idx_sources_dict1


def eval_res(args):
    time_s = time.time()
    if args.task_type == "EAE":
        if args.eval_type == 'obj':
            pred_list, gold_list = cvt_eae.do_convert(args)
            pred_dict = gen_idx_events_dict(events_list=pred_list)
            gold_dict = gen_idx_events_dict(events_list=gold_list)
        else:
            pred_dict = gen_idx_events_dict(path=args.pred_file)
            gold_dict = gen_idx_events_dict(path=args.gold_file)
        source_dict = gen_idx_sources_dict(path=args.input_file) if args.granularity == "source" else {}
        prf_s = cal_scores_ai_ac(pred_dict, gold_dict, args.match_type, args.granularity, source_dict)
        # {'AI': 'Argument Identification' , 'AC': 'Argument Classification'}
        metric_names = ['AI', 'AC']
    elif args.task_type == "EE":
        if args.eval_type == 'obj':
            pred_list, gold_list = cvt_ee.do_convert(args)
            pred_dict = gen_idx_events_dict(events_list=pred_list)
            gold_dict = gen_idx_events_dict(events_list=gold_list)
        else:
            pred_dict = gen_idx_events_dict(path=args.pred_file)
            gold_dict = gen_idx_events_dict(path=args.gold_file)
        source_dict = gen_idx_sources_dict(path=args.input_file) if args.granularity == "source" else {}
        prf_s = cal_scores_ti_tc_ai_ac(pred_dict, gold_dict, args.match_type, args.granularity, source_dict)
        # {'TI': 'Trigger Identification ', 'TC':'Trigger Classification ',
        #  'AI': 'Argument Identification ', 'AC': 'Argument Classification '}
        metric_names = ['TI', 'TC']
    elif args.task_type == "NER":
        if args.eval_type == 'obj':
            pred_list, gold_list = cvt_ner.do_convert(args)
            pred_dict = gen_idx_entities_dict(entities_list=pred_list)
            gold_dict = gen_idx_entities_dict(entities_list=gold_list)
        else:
            pred_dict = gen_idx_entities_dict(path=args.pred_file)
            gold_dict = gen_idx_entities_dict(path=args.gold_file)
        source_dict = gen_idx_sources_dict(path=args.input_file) if args.granularity == "source" else {}
        prf_s = cal_scores_ei_ec(pred_dict, gold_dict, args.match_type, args.granularity, source_dict)
        # {'EI': 'Entity Identification ', 'EC': 'Entity Classification '}
        metric_names = ['EI', 'EC']
    elif args.task_type == "RE":
        if args.eval_type == "obj":
            pred_list, gold_list = cvt_re.do_convert(args)
            pred_dict = gen_idx_relations_dict(relations_list=pred_list)
            gold_dict = gen_idx_relations_dict(relations_list=gold_list)
        else:
            pred_dict = gen_idx_relations_dict(path=args.pred_file)
            gold_dict = gen_idx_relations_dict(path=args.gold_file)
        source_dict = gen_idx_sources_dict(path=args.input_file) if args.granularity == "source" else {}
        prf_s = cal_scores_ri_rc(pred_dict, gold_dict, args.match_type, args.granularity, source_dict)
        # {'RI': 'Relation Identification ', 'RC': 'Relation Classification '}
        metric_names = ['RI', 'RC']

    result = {}
    if args.granularity == 'overall':
        for i, prf in enumerate(prf_s):
            try:
                result[f'{args.match_type}-{metric_names[i]}-P'] = f"{prf[0]*100:.1f}"
                result[f'{args.match_type}-{metric_names[i]}-R'] = f"{prf[1]*100:.1f}"
                result[f'{args.match_type}-{metric_names[i]}-F1'] = f"{prf[2]*100:.1f}"
            except:
                pass
    elif args.granularity == 'type':
        for type, prfs in prf_s.items():
            result[type] = {}
            for i, prf in enumerate(prfs):
                try:
                    result[type][f'{args.match_type}-{metric_names[i]}-P'] = f"{prf[0]*100:.1f}"
                    result[type][f'{args.match_type}-{metric_names[i]}-R'] = f"{prf[1]*100:.1f}"
                    result[type][f'{args.match_type}-{metric_names[i]}-F1'] = f"{prf[2]*100:.1f}"
                except:
                    pass
    elif args.granularity == 'source':
        for source, prfs in prf_s.items():
            result[source] = {}
            for i, prf in enumerate(prfs):
                try:
                    result[source][f'{args.match_type}-{metric_names[i]}-P'] = f"{prf[0]*100:.1f}"
                    result[source][f'{args.match_type}-{metric_names[i]}-R'] = f"{prf[1]*100:.1f}"
                    result[source][f'{args.match_type}-{metric_names[i]}-F1'] = f"{prf[2]*100:.1f}"
                except:
                    pass

    time_e = time.time()
    with open(f"{args.output_dir}/{args.result_file}", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    #print(f"The result has been dumped in {args.output_dir}/{args.result_file}")
    #print(f'Time Cost: {time_e - time_s:.2f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default="prediction.json")
    parser.add_argument("--gold_file", type=str, default="label.json")
    parser.add_argument("--input_file", type=str, default="../eval_res/0919_test/intermediate_EAE.json")
    parser.add_argument("--prediction_name", type=str, default="200_prediction.json")
    parser.add_argument("--eval_type", type=str, default="file")
    parser.add_argument("--task_type", type=str, default="EAE", choices=['EAE', 'EE', 'NER', 'RE'])   
    parser.add_argument("--match_type", type=str, default="HM", choices=['EM', 'HM'])   # {'EM': 'Exact Match', 'HM': 'Head word Match'}
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--result_file", type=str, default="result.json")
    parser.add_argument("--filter_outlier", type=int, default=0, help="Filtering items that don't belong to defined type architecture and are not in sentence.")
    parser.add_argument("--granularity", type=str, default="overall", choices=['overall', 'type', 'source'])
    parser.add_argument("--wikidata_upper", type=int, default=0)
    parser.add_argument("--ontology_dir", type=str, default="The directory path of defined ontology architecture(could be empty).")
    parser.add_argument("--schema_type", type=str, default="aligned", choices=['aligned', 'unaligned'])
    args = parser.parse_args()
    eval_res(args)
