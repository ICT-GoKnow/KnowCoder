import argparse
import time

from utils import read_jsonl_file, dump_json_file


def do_combine(args):
    for task in args.run_tasks.split(","):
        print(f"start to combine {task} checkpoints prediction!")
        time_s = time.time()
        combined_dict = {}
        combined_list = []
        for ckpt in args.ckpts.split(","):
            ckpt_data = read_jsonl_file(f"{args.input_dir}/{ckpt}_{task}/prediction.json")
            for data in ckpt_data:
                id = data["id"]
                if id not in combined_dict:
                    combined_dict[id] = {"id": id, "content": data["content"], f"{ckpt}_prediction": data["entities"]}
                else:
                    combined_dict[id][f"{ckpt}_prediction"] = data["entities"]
        for _, value in combined_dict.items():
            combined_list.append(value)
        time_e = time.time()
        dump_json_file(combined_list, f"{args.output_dir}/{task}_combined_prediction.json")
        print(f"The result has been dumped in {args.output_dir}/{task}_combined_prediction.json")
        print(f'Time Cost: {time_e - time_s:.2f}s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, help="The directory path of input data, only support the JSON format.")
    parser.add_argument("--output_dir", type=str, help="Saving path in specified format.")
    parser.add_argument("--run_tasks", type=str, default="tasks")
    parser.add_argument("--ckpts", type=str, help="checkpoints")

    args = parser.parse_args()
    do_combine(args)