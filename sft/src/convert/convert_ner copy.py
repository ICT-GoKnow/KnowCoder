import argparse
import os
import json
import copy
import re


def extract_from_string(input_string):
    pattern = r'(\w+)\s*=\s*(\w+)\(.*?(\w+)\s*=\s*\"(.*?)\"'
    matches = re.findall(pattern, input_string, re.DOTALL)
    entities_dict = {}

    for match in matches:
        key = match[3]
        value = match[1]
        entities_dict[key] = value

    return entities_dict


def convert(dataset, prediction_name=None):
    predictions = []
    labels = []

    for idx, data in enumerate(dataset):
        new_data = {"id": idx, "content": data["input"], "entities": []}
        label = copy.deepcopy(new_data)
        for entity in data["entities"]:
            e = {"type": entity["type"], "word": entity["name"]}
            label["entities"].append(e)
        labels.append(label)

        if data[prediction_name] == "":
            continue
        prediction = copy.deepcopy(new_data)
        entities_dict = extract_from_string(data[prediction_name])
        for entity_word, entity_type in entities_dict.items():
            e = {"type": entity_type, "word": entity_word}
            prediction["entities"].append(e)
        predictions.append(prediction)

    return predictions, labels


def do_convert(args):
    if not os.path.exists(args.input_file):
        raise ValueError("Please input the correct path of dataset file.")

    with open(args.input_file, "r", encoding="utf-8") as infile:
        dataset = json.load(infile)
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

    parser.add_argument("--input_file",
                        type=str,
                        default="data/ner/NER-test-prediction.json",
                        help="The file path of input dataset, only support the JSON format.")
    parser.add_argument("--output_dir", type=str, default="data/ner/", help="Saving path in specified format.")
    parser.add_argument("--prediction_name", type=str, default=None, help="The name of prediction field.")
    parser.add_argument("--eval_type", type=str, default="file")

    args = parser.parse_args()

    do_convert(args)
