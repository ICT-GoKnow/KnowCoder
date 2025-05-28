import argparse
import os
import json
import copy
import re
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0, parentdir)  


from utils import get_schema_dict, read_json_file, norm_name


def extract_from_string(input_string):
    result = {}
    if input_string == "" or input_string == "None":
        return result
    pattern = r'(\w+)\s*=\s*\[(.*?)\]'
    matches = re.findall(pattern, input_string, re.DOTALL)
    for match in matches:
        result[match[0]] = re.findall(r'\("(.*?)"\)', match[1])
    return result


def convert(dataset, prediction_name=None, filter_outlier=False, schema=None):
    predictions = []
    labels = []
    for idx, data in enumerate(dataset):
        new_data = {"id": idx, "content": data["sentence"], "events": []}
        trigger = data["trigger"]
        event = {"type": norm_name(data["event_type"]), "trigger": {"span": [trigger["start"], trigger["end"]], "word": trigger["text"]}, "args": {}}
        new_data["events"].append(event)

        label_args = {}
        for arg in data["arguments"]:
            role = norm_name(arg["role"])
            word = arg["text"]
            if role not in label_args:
                label_args[role] = []
            if 'start' in arg:
                arg_role_dict = {"word": word, "span": [arg["start"], arg["end"]]}
            else:
                arg_role_dict = {"word": word}
            label_args[role].append(arg_role_dict)
        label = copy.deepcopy(new_data)
        label["events"][0]["args"] = label_args
        labels.append(label)

        prediction = copy.deepcopy(new_data)
        pred_args = {}
        args_dict = extract_from_string(data[prediction_name])
        for arg_role, arg_values in args_dict.items():
            arg_role = norm_name(arg_role)
            pred_args[arg_role] = []
            for arg_value in arg_values:
                if filter_outlier:
                    if not arg_value or arg_value not in data["sentence"]:
                        continue
                pred_args[arg_role].append({"word": arg_value})
        prediction["events"][0]["args"] = pred_args
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
