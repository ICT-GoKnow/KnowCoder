import argparse
import os
import json
import copy
import re
import sys
from tqdm import tqdm
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0, parentdir)  

from utils import get_schema_dict, norm_name, timeout
event_dict={
    "acquit": "justice",
    "appeal": "justice",
    "arrest jail detain": "justice",
    "attack": "conflict",
    "be born": "life",
    "charge indict": "justice",
    "convict": "justice",
    "declare bankruptcy": "business",
    "demonstrate": "conflict",
    "die": "life",
    "divorce": "life",
    "elect": "personnel",
    "end organization": "business",
    "end position": "personnel",
    "execute": "justice",
    "extradite": "justice",
    "fine": "justice",
    "injure": "life",
    "marry": "life",
    "meet": "contact",
    "merge organization": "business",
    "nominate": "personnel",
    "pardon": "justice",
    "phone write": "contact",
    "release parole": "justice",
    "sentence": "justice",
    "start organization": "business",
    "start position": "personnel",
    "sue": "justice",
    "transfer money": "transaction",
    "transfer ownership": "transaction",
    "transportation": "movement",
    "trial hearing": "justice"
}

@timeout(0.3)
def extract_from_string(input_string):
    events_dict = {}
    if input_string == "" or input_string == "None":
        return events_dict
    pattern = r'\s*(\w+)\(((?:[^()]*|\([^()]*\)|\((.*?)\((.*?)\)(.*?)\))*)\)'
    try:
        events = re.findall(pattern, input_string, re.DOTALL)
    except:
        return events_dict
    events_list_out=[]
    for event in events:
        pattern = r'(\w+)\s*=\s*\[(.*?)\]'
        args = re.findall(pattern, event[2], re.DOTALL)
        args_dict = {arg[0]: re.findall(r'\("(.*?)"\)', arg[1]) for arg in args}
        t_a = {"args": args_dict}
        pattern = r'\s*"([^"]*)"'
        t_a["trigger"] = re.findall(pattern, event[1])
        events_list_out.append((event[0],t_a))
        events_dict[event[0]] = t_a
    return events_list_out


def convert(dataset, prediction_name=None, filter_outlier=0, schema=None):
    predictions = []
    labels = []
    for idx, data in enumerate(tqdm(dataset)):
        new_data = {"id": idx, "content": data["sentence"], "events": []}
        label = copy.deepcopy(new_data)
        for event in data["events"]:
            label_args = {}

            #for arg in event["arguments"]:
            #    role = norm_name(arg["role"])
            #    word = arg["name"]
            #    if role not in label_args:
            #        label_args[role] = []
            #    label_args[role].append({"word": word})
            if norm_name(event["type"]) in event_dict:
                e = {"type": event_dict[norm_name(event["type"])], "trigger": {"word": event["trigger"].lower()}, "args": label_args}
                #e = {"type": event_dict[norm_name(event["type"])], "trigger": {"word":" "}, "args": label_args}

            else:
                e = {"type": norm_name(event["type"]), "trigger": {"word": event["trigger"].lower()}, "args": label_args}
                #e = {"type": norm_name(event["type"]), "trigger": {"word": " "}, "args": label_args}

            label["events"].append(e)
        labels.append(label)

        prediction = copy.deepcopy(new_data)
        events_dict = extract_from_string(data[prediction_name])
        for event_type, event_t_a in events_dict:
            if norm_name(event_type) in event_dict:
                event_type = event_dict[norm_name(event_type)]
            else:
                event_type = norm_name(event_type)
            
            try:
                trigger = event_t_a["trigger"][0]
            except:
                continue
           #if filter_outlier:
           #    #source="ACE 2005"
           #    #if event_type not in schema[source]:
           #    #    continue
           #    if not trigger or trigger not in data['sentence']:
           #        continue
            pred_args = {}
            for arg_role, arg_values in event_t_a["args"].items():
                arg_role = norm_name(arg_role)
                #if filter_outlier and arg_role not in schema[data['source']][event_type]:
                #    continue
                pred_args[arg_role] = []
                for arg_value in arg_values:
                    if filter_outlier and (not arg_value or arg_value not in data['sentence']):
                        continue
                    pred_args[arg_role].append({"word": arg_value})
            e = {"type": event_type, "trigger": {"word": trigger.lower()}, "args": {}}
            #e = {"type": event_type, "trigger": {"word": " "}, "args": {}}
            prediction["events"].append(e)
        predictions.append(prediction)
    return predictions, labels


def do_convert(args):
    if not os.path.exists(args.input_file):
        raise ValueError("Please input the correct path of dataset file.")

    with open(args.input_file, "r", encoding="utf-8") as infile:
        dataset = json.load(infile)
        if args.filter_outlier:
            schema = get_schema_dict(f'{args.ontology_dir}/Event_{args.schema_type}.json')
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
