import argparse
import os
import json
import copy
import re
import sys
from tqdm import tqdm
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0, parentdir)  

from utils import get_schema_dict, norm_name


def extract_from_string(input_string):
    relations_dict = {}
    if input_string == "" or input_string == "None":
        return relations_dict
    pattern = r"(\w+)\(\s*(\w+)\(\"(.+?)\"\),\s*(\w+)\(\"(.+?)\"\)"
    matches = re.findall(pattern, input_string)
    relations_output=[]
    for match in matches:
        key = ((match[2], match[1]), (match[4], match[3]))
        value = match[0]
        relations_dict[key] = value
        relations_output.append((key,value))
    return relations_output


def convert(dataset, prediction_name=None, filter_outlier=False, schema=None):
    predictions = []
    labels = []
    for idx, data in enumerate(tqdm(dataset)):
        new_data = {"id": idx, "content": data["sentence"], "relations": []}
        label = copy.deepcopy(new_data)
        for relation in data["relations"]:
            subject = {"word": relation["head"]["name"]}
            object = {"word": relation["tail"]["name"]}
            r = {"type": norm_name(relation["type"]), "subject": subject, "object": object}
            label["relations"].append(r)
        labels.append(label)

        prediction = copy.deepcopy(new_data)
        relations_dict = extract_from_string(data[prediction_name])
        for relation_words, relation_type in relations_dict:
            relation_type = norm_name(relation_type)
            if filter_outlier:
                if not relation_words[0][0] or not relation_words[1][0] or relation_words[0][0] == relation_words[1][0]:
                    continue
                if relation_words[0][0] not in data["sentence"] or relation_words[1][0] not in data["sentence"]:
                    continue
                #print(data['source'])
                source1="CoNLL 2004"
                #source1=data['source']
                #source1=data['source'].split("/")[-2]
                if relation_type not in schema[source1]:
                    continue
            r = {"type": relation_type, "subject": {"word": relation_words[0][0], "entity_type": norm_name(relation_words[0][1])}, "object": {"word": relation_words[1][0], "entity_type": norm_name(relation_words[1][1])}}
            prediction["relations"].append(r)
        predictions.append(prediction)
    return predictions, labels


def do_convert(args):
    if not os.path.exists(args.input_file):
        raise ValueError("Please input the correct path of dataset file.")

    with open(args.input_file, "r", encoding="utf-8") as infile:
        dataset = json.load(infile)
        if args.filter_outlier:
            schema = get_schema_dict(f'{args.ontology_dir}/Relation_{args.schema_type}.json')
            prediction, label = convert(dataset, args.prediction_name, args.filter_outlier, schema)
        else:
            prediction, label = convert(dataset, args.prediction_name)

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
    parser.add_argument("--ontology_dir", type=str, help="The directory path of defined ontology architecture(could be empty).")
    parser.add_argument("--schema_type", type=str, default="aligned", choices=['aligned', 'unaligned'])
    args = parser.parse_args()
    do_convert(args)