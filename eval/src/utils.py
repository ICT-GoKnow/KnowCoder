import os
import re
import json
import time
import signal
import traceback
from functools import wraps


def extract_generated_code(resp, tokenizer=None):
    if not isinstance(resp, str):
        import vllm  

    if isinstance(resp, vllm.outputs.RequestOutput):
        # print(resp.prompt[-200:])
        # print('#'*100)
        resp = resp.prompt + tokenizer.decode(resp.outputs[0].token_ids)
        resp = resp.strip()
    if not resp.endswith('"""'):
        resp += '"""'
    ans = resp.split('"""')
    return ans[-2].strip().replace('</s>', '')


def read_json_file(file):
    with open(file, 'r', encoding='UTF-8') as file:
        data = json.load(file)
    return data


def read_jsonl_file(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def dump_json_file(obj, file):
    with open(file, 'w', encoding='UTF-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def dump_jsonl_file(records, path):
    with open(path, 'w', encoding='utf-8') as outfile:
        for record in records:
            outline = json.dumps(record, ensure_ascii=False)
            outfile.write(outline + "\n")




def insert_data_to_visual_html(data, ori_html='static/visual.html', new_html='static/visual_new.html'):
    '''
    Args:
        data: 
        ori_html: 
        new_html: 
    '''
    ori_html_lines = []
    with open(ori_html, 'r', encoding='UTF-8') as f:
        for line in f:
            ori_html_lines.append(line)

    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)

    pattern = r'\s+const metric_data = .*;\n'
    for idx in range(len(ori_html_lines)):
        if re.match(pattern, ori_html_lines[idx]):
            prefix_idx = ori_html_lines[idx].find('const metric_data = ')
            new_line = ori_html_lines[idx][:prefix_idx] + "const metric_data = '" + str(data) + "';\n"
            ori_html_lines[idx] = new_line
            break

    with open(new_html, 'w', encoding='UTF-8') as f:
        for line in ori_html_lines:
            f.write(line)


def get_schema_list(path):
    if not os.path.exists(path):
        return []
    else:
        return read_json_file(path)


def get_schema_dict(path):
    if not os.path.exists(path):
        return {}
    else:
        return read_json_file(path)


def gen_idx_sources_dict(path):
    source_dict = {}
    if not os.path.exists(path):
        raise ValueError("Please input the correct path of dataset file.")
    dataset = read_json_file(path)
    for idx, data in enumerate(dataset):
        source_dict[idx] = data['source']
    return source_dict
def gen_idx_sources_dict1(path):
    source_dict = {}
    if not os.path.exists(path):
        raise ValueError("Please input the correct path of dataset file.")
    dataset = read_jsonl_file(path)
    for idx, data in enumerate(dataset):
        source_dict[idx] = data['source']
    return source_dict

def norm_name(name):
    if '-' in name:
        return name.replace('-', ' ').lower()
    elif '_' in name:
        return name.replace('_', ' ').lower()
    else:
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        return name.lower()


class MyTimeOutError(AssertionError):
    def __init__(self, value="Time Out"):
        self.value = value

    def __str__(self):
        return repr(self.value)


def _raise_exception(exception, exception_message):
    if exception_message is None:
        raise exception()
    else:
        raise exception(exception_message)


def timeout(seconds, timeout_exception=MyTimeOutError, exception_message=None):

    def decorate(function):

        def handler(signum, frame):
            _raise_exception(timeout_exception, exception_message)

        @wraps(function)
        def new_function(*args, **kwargs):
            if not seconds:
                return function(*args, **kwargs)

            old = signal.signal(signal.SIGALRM, handler)
            old_left_time = signal.getitimer(signal.ITIMER_REAL)

            
            true_seconds = seconds
            if old_left_time[0]:
                true_seconds = min(old_left_time[0], seconds)

            
            signal.setitimer(signal.ITIMER_REAL, true_seconds)

            start_time = time.time()
            try:
                result = function(*args, **kwargs)
            finally:
                end_time = time.time()
                old_left_time = max(0, old_left_time[0] - (end_time - start_time))

                signal.setitimer(signal.ITIMER_REAL, old_left_time)
                signal.signal(signal.SIGALRM, old)
            return result

        return new_function

    return decorate
