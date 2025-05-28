import argparse
import os
import json
import copy
import re
import sys
from owlready2 import get_ontology
from tqdm import tqdm
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0, parentdir)  

from utils import get_schema_dict, read_json_file, norm_name
import ast


def get_ancestors_qid(cur_qid):
    ancestors_qid = []
    cur_onto = onto[cur_qid]
    if cur_onto:
        ancestors = cur_onto.ancestors()
    else:
        return ancestors_qid
    for item in ancestors:
        item = str(item)
        if item.startswith('wikidata_ontology.'):
            ancestors_qid.append(item.split('.')[1])
    return ancestors_qid


def convert_type(entity_type):
    entity_types = []
    qids = name2qid[entity_type] if entity_type in name2qid else []
    for qid in qids:
        ancestors_qid = get_ancestors_qid(qid)
        for ancestor_qid in ancestors_qid:
            entity_types.extend(qid2name[ancestor_qid] if ancestor_qid in qid2name else [])
    return entity_types


def extract_from_string(input_string):
    entities_dict = {}
    if input_string == "" or input_string == "None":
        return entities_dict
    #pattern = r'(\w+)\s*=\s*(\w+)\(.*?(\w+)\s*=\s*\"(.*?)\"'
    #pattern = r'\s*(\w+)\(.*?(\w+)\s*=\s*\"(.*?)\"'
    # pattern = r'\s*(\w+)\(\s*\"(.*?)\"'
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, input_string, re.DOTALL)
    # print(matches)
    if matches and matches[0]:
        entities_list = re.findall(r"'(.*?)'", matches[0])
        for idx, ent in enumerate(entities_list):
            entities_dict[idx] = (ent, "Entity")
    return entities_dict


def convert(dataset, prediction_name=None, filter_outlier=0, schema=None, wikidata_upper=0):
    predictions = []
    labels = []
    
    for idx, data in enumerate(tqdm(dataset)):

        new_data = {"id": idx, "content": data["sentence"], "entities": []}
        label = copy.deepcopy(new_data)
        for entity in data["entities"]:
            if norm_name(entity["type"])=="other":
                continue
            e = {"type": norm_name(entity["type"]), "word": entity["name"]}
            #if entity["name"]!="##<pad>##":
            #    label["entities"].append(e)
            if e not in label["entities"]:
                label["entities"].append(e)
        labels.append(label)

        prediction = copy.deepcopy(new_data)
        entities_dict = extract_from_string(data[prediction_name])
        for _, (entity_word, entity_type) in entities_dict.items():
            
            entity_type = norm_name(entity_type)
            if entity_type=="other":
                continue
            #source=data['source']
            if filter_outlier:
                #source="CoNLL 2003"
                #print(source.split("/"))
                source=data['source']
                #source=source.split("/")[-2]
                #source="CrossNER_AI"
                if not entity_word or entity_word not in data["sentence"]:
                    continue
                try:
                    if entity_type not in schema[source]:
                        continue
                except:
                    pass
            if wikidata_upper:
                origin_type = entity_type
                entity_type = convert_type(entity_type)
                entity_type.append(origin_type)    
            e = {"type": entity_type, "word": entity_word}
            if e not in prediction["entities"]:
                prediction["entities"].append(e)
        predictions.append(prediction)
    return predictions, labels


def do_convert(args):
    if not os.path.exists(args.input_file):
        raise ValueError("Please input the correct path of dataset file.")

    if args.wikidata_upper:
        global onto
        global name2qid
        global qid2name
        name2qid = read_json_file(f'{args.ontology_dir}/name2qid.json')
        qid2name = read_json_file(f'{args.ontology_dir}/qid2name.json')
        onto = get_ontology(f'{args.ontology_dir}/wikidata_ontology.owl').load()

    with open(args.input_file, "r", encoding="utf-8") as infile:
        dataset = json.load(infile)
        if args.filter_outlier:
            schema = get_schema_dict(f'{args.ontology_dir}/Entity_{args.schema_type}.json')
            prediction, label = convert(dataset, args.prediction_name, args.filter_outlier, schema, args.wikidata_upper)
        else:
            prediction, label = convert(dataset, args.prediction_name, wikidata_upper=args.wikidata_upper)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.eval_type == 'obj':
        return prediction, label

    with open(os.path.join(args.output_dir, "prediction.json"), "w", encoding="utf-8") as outfile:
        for item in prediction:
            outline = json.dumps(item, ensure_ascii=False)
            outfile.write(outline + "\n")

    with open(os.path.join(args.output_dir, "label.json"), "w", encoding="utf-8") as outfile:
        for item in label:
            outline = json.dumps(item, ensure_ascii=False)
            outfile.write(outline + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="The file path of input dataset, only support the JSON format.")
    parser.add_argument("--output_dir", type=str, help="Saving path in specified format.")
    parser.add_argument("--prediction_name", type=str, default=None, help="The name of prediction field.")
    parser.add_argument("--eval_type", type=str, default="file")
    parser.add_argument("--filter_outlier", type=int, default=0, help="Filtering items that don't belong to defined type architecture and are not in sentence.")
    parser.add_argument("--wikidata_upper", type=int, default=0)
    parser.add_argument("--ontology_dir", type=str, help="The directory path of defined ontology architecture(could be empty).")
    parser.add_argument("--schema_type", type=str, default="aligned", choices=['aligned', 'unaligned'])
    args = parser.parse_args()
    do_convert(args)