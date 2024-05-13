

import argparse
import os

from tqdm import tqdm

from utils import dump_json_file, read_json_file, single_plot, insert_data_to_visual_html


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_tasks', type=str, default='EAE,NER,RE', help='involved tasks')
    parser.add_argument('--run_match_types', type=str, default='HM,EM', help='match types')
    parser.add_argument('--ckpts', type=str, default='200,400,600,800,1000,1200,1400', help='checkpoints')
    parser.add_argument('--intermediate_data_dir', type=str, default='intermediate_data', help='intermediate data dir')
    parser.add_argument('--save_split_results', action='store_true', help='save split results')
    parser.add_argument('--output_dir', type=str, default='../eval/0919_test', help='intermediate data dir')
    parser.add_argument('--granularity', type=str, default='overall', help='intermediate data dir')

    args = parser.parse_args()
    if args.granularity=="overall":
        all_metric_res = {}
        for task in args.run_tasks.split(','):
            if task not in all_metric_res:
                all_metric_res[task] = {}
            for ckpt in args.ckpts.split(','):
                if ckpt not in all_metric_res[task]:
                    all_metric_res[task][ckpt] = {}
                for match_type in args.run_match_types.split(','):
                    res_file = os.path.join(args.intermediate_data_dir, f'{task}/{ckpt}/result_overall_{match_type}.json')
                    res = read_json_file(res_file)
                    for metric_name, metric_data in res.items():
                        all_metric_res[task][ckpt][metric_name] = metric_data

        dump_json_file(all_metric_res, os.path.join(args.output_dir, 'all_metric_res.json'))
        old_html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static/visual.html")
        new_html_file = os.path.join(args.output_dir, os.path.basename(os.path.abspath(args.output_dir)) + '_visual.html')
        insert_data_to_visual_html(all_metric_res, ori_html=old_html_file, new_html=new_html_file)

        
        if args.save_split_results:
            visual_dir = os.path.join(args.output_dir, 'visual')
            png_dir = os.path.join(visual_dir, 'png')
            html_dir = os.path.join(visual_dir, 'html')
            os.makedirs(visual_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(html_dir, exist_ok=True)

        x_ckpts = sorted([int(ckpt) for ckpt in args.ckpts.split(',')])
        for task in args.run_tasks.split(','):
            metric_list = all_metric_res[task][str(x_ckpts[0])].keys()
            for metric in tqdm(metric_list):
                y_data = [float(all_metric_res[task][str(ckpt)][metric]) for ckpt in x_ckpts]
                if args.save_split_results:
                    fig_name = os.path.join(png_dir, f'{task}-{metric}.png')
                    html_name = os.path.join(html_dir, f'{task}-{metric}.html')
                else:
                    fig_name, html_name = None, None
                single_plot(x_data=x_ckpts, y_data=y_data, show_img=False,
                            x_title='Checkpoint', y_title='Value', title=f'{task}-{metric}',
                            fig_name=fig_name, html_name=html_name)
    if args.granularity=="marco_source":
        args.granularity="macromean"
        all_metric_res = {}
        for task in args.run_tasks.split(','):
            if task not in all_metric_res:
                all_metric_res[task] = {}
            for ckpt in args.ckpts.split(','):
                if ckpt not in all_metric_res[task]:
                    all_metric_res[task][ckpt] = {}
                for match_type in args.run_match_types.split(','):
                    res_file = os.path.join(args.intermediate_data_dir, f'{task}/{ckpt}/result_marco_source_{match_type}.json')
                    res = read_json_file(res_file)
                    for metric_name, metric_data in res.items():
                        all_metric_res[task][ckpt][metric_name] = metric_data

        dump_json_file(all_metric_res, os.path.join(args.output_dir, 'all_metric_res.json'))
        old_html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static/visual.html")
        new_html_file = os.path.join(args.output_dir, os.path.basename(os.path.abspath(args.output_dir)) + '_visual_macro_source.html')
        insert_data_to_visual_html(all_metric_res, ori_html=old_html_file, new_html=new_html_file)

        
        if args.save_split_results:
            visual_dir = os.path.join(args.output_dir, 'visual')
            png_dir = os.path.join(visual_dir, 'png')
            html_dir = os.path.join(visual_dir, 'html')
            os.makedirs(visual_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(html_dir, exist_ok=True)

        x_ckpts = sorted([int(ckpt) for ckpt in args.ckpts.split(',')])
        for task in args.run_tasks.split(','):
            metric_list = all_metric_res[task][str(x_ckpts[0])].keys()
            for metric in tqdm(metric_list):
                y_data = [float(all_metric_res[task][str(ckpt)][metric]) for ckpt in x_ckpts]
                if args.save_split_results:
                    fig_name = os.path.join(png_dir, f'{task}-{metric}.png')
                    html_name = os.path.join(html_dir, f'{task}-{metric}.html')
                else:
                    fig_name, html_name = None, None
                single_plot(x_data=x_ckpts, y_data=y_data, show_img=False,
                            x_title='Checkpoint', y_title='Value', title=f'{task}-{metric}',
                            fig_name=fig_name, html_name=html_name)
    if args.granularity=="type":
        all_metric_res = {}
        res_dict={}
        for task in args.run_tasks.split(','):
            if task not in all_metric_res:
                all_metric_res[task] = {}
            for ckpt in args.ckpts.split(','):
                if ckpt not in all_metric_res[task]:
                    all_metric_res[task][ckpt] = {}
                res_dict[task]=set()
                for match_type in args.run_match_types.split(','):
                    res_file = os.path.join(args.intermediate_data_dir, f'{task}/{ckpt}/result_type_{match_type}.json')
                    res = read_json_file(res_file)
                    new_sys1 = sorted(res.items(), key=lambda d: d[0], reverse=False)
                    res={}
                    for sys in new_sys1:
                        res[sys[0]]=sys[1]
                    for source_name, source_data in res.items():
                        
                        res_dict[task].add(source_name)
                        if source_name not in all_metric_res[task][ckpt]:
                            all_metric_res[task][ckpt][source_name] = {}
                        for metric_name,metric_data in source_data.items():
                            all_metric_res[task][ckpt][source_name][metric_name]=metric_data

        dump_json_file(all_metric_res, os.path.join(args.output_dir, 'all_metric_res.json'))
        old_html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static/visual_type.html")
        new_html_file = os.path.join(args.output_dir, os.path.basename(os.path.abspath(args.output_dir)) + '_visual_type.html')
        insert_data_to_visual_html(all_metric_res, ori_html=old_html_file, new_html=new_html_file)

        
        if args.save_split_results:
            visual_dir = os.path.join(args.output_dir, 'visual')
            png_dir = os.path.join(visual_dir, 'png')
            html_dir = os.path.join(visual_dir, 'html')
            os.makedirs(visual_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(html_dir, exist_ok=True)

        x_ckpts = sorted([int(ckpt) for ckpt in args.ckpts.split(',')])
        for task in args.run_tasks.split(','):
            source_list=list(res_dict[task])
            for source in source_list:
                if source not in all_metric_res[task][str(x_ckpts[0])].keys():
                    continue
                metric_list = all_metric_res[task][str(x_ckpts[0])][source].keys()
                for metric in tqdm(metric_list):
                    flag=False
                    for ckpt in x_ckpts:
                        if source not in all_metric_res[task][str(ckpt)].keys():
                            flag=True
                    if flag==True:
                        continue
                    y_data = [float(all_metric_res[task][str(ckpt)][source][metric]) for ckpt in x_ckpts]
                    if args.save_split_results:
                        fig_name = os.path.join(png_dir, f'{task}-{metric}.png')
                        html_name = os.path.join(html_dir, f'{task}-{metric}.html')
                    else:
                        fig_name, html_name = None, None
                    single_plot(x_data=x_ckpts, y_data=y_data, show_img=False,
                                x_title='Checkpoint', y_title='Value', title=f'{task}-{metric}',
                                fig_name=fig_name, html_name=html_name) 
    if args.granularity=="source":
        all_metric_res = {}
        res_dict={}
        for task in args.run_tasks.split(','):
            if task not in all_metric_res:
                all_metric_res[task] = {}
            for ckpt in args.ckpts.split(','):
                if ckpt not in all_metric_res[task]:
                    all_metric_res[task][ckpt] = {}
                    res_dict[task]=set()
                for match_type in args.run_match_types.split(','):
                    res_file = os.path.join(args.intermediate_data_dir, f'{task}/{ckpt}/result_source_{match_type}.json')
                    res = read_json_file(res_file)
                    new_sys1 = sorted(res.items(), key=lambda d: d[0], reverse=False)
                    res={}
                    for sys in new_sys1:
                        res[sys[0]]=sys[1]
                    for source_name, source_data in res.items():
                        print(source_name)
                        print(source_name.split("/"))
                        if "/" in source_name:
                            source_name=source_name.split("/")[3]
                        else:
                            pass
                        res_dict[task].add(source_name)
                        all_metric_res[task][ckpt][source_name] = {}
                        for metric_name,metric_data in source_data.items():
                            all_metric_res[task][ckpt][source_name][metric_name]=metric_data

        dump_json_file(all_metric_res, os.path.join(args.output_dir, 'all_metric_res.json'))
        old_html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static/visual_source.html")
        new_html_file = os.path.join(args.output_dir, os.path.basename(os.path.abspath(args.output_dir)) + '_visual_source.html')
        insert_data_to_visual_html(all_metric_res, ori_html=old_html_file, new_html=new_html_file)

        
        if args.save_split_results:
            visual_dir = os.path.join(args.output_dir, 'visual')
            png_dir = os.path.join(visual_dir, 'png')
            html_dir = os.path.join(visual_dir, 'html')
            os.makedirs(visual_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(html_dir, exist_ok=True)

        x_ckpts = sorted([int(ckpt) for ckpt in args.ckpts.split(',')])
        for task in args.run_tasks.split(','):
            source_list=list(res_dict[task])
            for source in source_list:
                metric_list = all_metric_res[task][str(x_ckpts[0])][source].keys()
                for metric in tqdm(metric_list):
                    y_data = [float(all_metric_res[task][str(ckpt)][source][metric]) for ckpt in x_ckpts]
                    if args.save_split_results:
                        fig_name = os.path.join(png_dir, f'{task}-{metric}.png')
                        html_name = os.path.join(html_dir, f'{task}-{metric}.html')
                    else:
                        fig_name, html_name = None, None
                    single_plot(x_data=x_ckpts, y_data=y_data, show_img=False,
                                x_title='Checkpoint', y_title='Value', title=f'{task}-{metric}',
                                fig_name=fig_name, html_name=html_name) 
    if args.granularity=="type_dis":
        all_metric_res = {}
        res_dict={}
        for task in args.run_tasks.split(','):
            if task not in all_metric_res:
                all_metric_res[task] = {}
            for ckpt in args.ckpts.split(','):
                if ckpt not in all_metric_res[task]:
                    all_metric_res[task][ckpt] = {}
                res_dict[task]=set()
                for match_type in args.run_match_types.split(','):
                    res_file = os.path.join(args.intermediate_data_dir, f'{task}/{ckpt}/result_{match_type}_type.json')
                    res = read_json_file(res_file)
                    for source_name, source_data in res.items():
                        res_dict[task].add(source_name)
                        if source_name not in all_metric_res[task][ckpt]:
                            all_metric_res[task][ckpt][source_name] = {}
                        for metric_name,metric_data in source_data.items():
                            all_metric_res[task][ckpt][source_name][metric_name]=metric_data
        new_metric={}
        run_task=args.run_tasks.split(',')
        #with open("all_metric_res.json","r") as f:
        #    all_metric_res=json.load(f)
        for task in all_metric_res.keys():
            new_metric[task]={}
            for k,v in all_metric_res[task].items():
                for k1,v1 in v.items():
                    new_metric[task][k1]={}
            for k1, v1 in new_metric[task].items():
                #new_metric[task][k1] = {}
                for k,v in all_metric_res[task].items():
                    new_metric[task][k1][k]={}
            for k, v in all_metric_res[task].items():
                for k1, v1 in v.items():
                    for k2,v2 in v1.items():
                        if k2=="EM-EC-F1" or k2=="EM-RC-F1" or k2=="EM-AC-F1":
                            new_metric[task][k1][k][k2]=v2
        new_metric_trans={}
        for task in new_metric.keys():
            for k, v in new_metric[task].items():
                for k1, v1 in v.items():                
                    if v1=={}:
                        if task=="NER":
                            new_metric[task][k][k1]={'EM-EC-F1': '0.0'}
                        if task=="RE":
                            new_metric[task][k][k1]={'EM-RC-F1': '0.0'}
                        if task=="EE":
                            new_metric[task][k][k1]={'EM-AC-F1': '0.0'}

        ckpts=args.ckpts.split(',')
        new_metric_trans={}
        for ckpt in ckpts:
            new_metric_trans[ckpt]={}
            for task in run_task:
                new_metric_trans[ckpt][task]={}
                for type1 in new_metric[task].keys():
                    if task=="NER":
                        new_metric_trans[ckpt][task][type1]=new_metric[task][type1][ckpt]['EM-EC-F1']
                    if task=="EE":
                        new_metric_trans[ckpt][task][type1] = new_metric[task][type1][ckpt]['EM-AC-F1']
                    if task=="RE":
                        new_metric_trans[ckpt][task][type1] = new_metric[task][type1][ckpt]['EM-RC-F1']
        #print(new_metric_trans)
        dump_json_file(all_metric_res, os.path.join(args.output_dir, 'all_metric_res.json'))
        old_html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static/visual_type_dis.html")
        new_html_file = os.path.join(args.output_dir, os.path.basename(os.path.abspath(args.output_dir)) + '_visual_type_dis.html')
        insert_data_to_visual_html(new_metric_trans, ori_html=old_html_file, new_html=new_html_file)
        
        if args.save_split_results:
            visual_dir = os.path.join(args.output_dir, 'visual')
            png_dir = os.path.join(visual_dir, 'png')
            html_dir = os.path.join(visual_dir, 'html')
            os.makedirs(visual_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(html_dir, exist_ok=True)

        x_ckpts = sorted([int(ckpt) for ckpt in args.ckpts.split(',')])
        for task in args.run_tasks.split(','):
            source_list=list(res_dict[task])
            for source in source_list:
                if source not in all_metric_res[task][str(x_ckpts[0])].keys():
                    continue
                metric_list = all_metric_res[task][str(x_ckpts[0])][source].keys()
                for metric in tqdm(metric_list):
                    flag=False
                    for ckpt in x_ckpts:
                        if source not in all_metric_res[task][str(ckpt)].keys():
                            flag=True
                    if flag==True:
                        continue
                    y_data = [float(all_metric_res[task][str(ckpt)][source][metric]) for ckpt in x_ckpts]
                    if args.save_split_results:
                        fig_name = os.path.join(png_dir, f'{task}-{metric}.png')
                        html_name = os.path.join(html_dir, f'{task}-{metric}.html')
                    else:
                        fig_name, html_name = None, None
                    single_plot(x_data=x_ckpts, y_data=y_data, show_img=False,
                                x_title='Checkpoint', y_title='Value', title=f'{task}-{metric}',
                                fig_name=fig_name, html_name=html_name)
