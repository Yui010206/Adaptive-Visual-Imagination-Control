import os
import json
import time
from collections import defaultdict

from vln.data_utils import construct_instrs
from vln.env import R2RNavBatch
from vln.parser import parse_args

from utils.data import set_random_seed
from utils.logger import write_to_record_file

# from vln.gpt_agent import GPTNavAgent
from vln.gpt_agent_wm_calling import GPTNavAgent


def build_dataset(args, rank=0, is_test=True):
    dataset_class = R2RNavBatch
    split = args.split
    val_envs = {}

    if 'processed' in split:
        with open(os.path.join(args.anno_dir, split+'.json'), 'r') as f:
            val_instr_data = json.load(f)

        if args.end is None:
            args.end = len(val_instr_data)
        val_instr_data = val_instr_data[args.start:args.end]
        print(f'------------------ Evaluate {args.start}-{args.end} in {split} ------------------')

    else:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split],
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )

    val_env = dataset_class(
        val_instr_data, args.connectivity_dir, batch_size=args.batch_size,
        seed=args.seed+rank,
        name=split, args=args,
    )   # evaluation using all objects
    val_envs[split] = val_env

    return val_envs


def valid(args, val_envs, rank=0):

    default_gpu = None
    agent_class = GPTNavAgent
    agent = agent_class(args, list(val_envs.values())[0], rank=rank)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    files = os.listdir('/nas-ssd2/shoubin/code/worldTTS/MapGPT/datasets/exprs_map/debug/preds/')
    baseline_dir = '/nas-ssd2/shoubin/code/worldTTS/MapGPT/datasets/exprs_map/test/preds/'
    for env_name, env in val_envs.items():

        print(f"Start evaluating {env_name}")
        prefix = 'submit' if args.detailed_output is False else 'detail'
        if os.path.exists(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))):
            print('Path already exists...')
            continue
        agent.logs = defaultdict(list)
        agent.env = env
        all_preds = []
        all_preds_baseline = []
        select = ['case_InstrID_2513_1.json', 'case_InstrID_3209_2.json', 'case_InstrID_4121_2.json', 'case_InstrID_237_2.json', 'case_InstrID_7198_2.json', 'case_InstrID_5166_0.json', 'case_InstrID_6310_1.json', 'case_InstrID_4605_1.json', 'case_InstrID_2390_1.json', 'case_InstrID_6999_2.json', 'case_InstrID_2365_2.json', 'case_InstrID_2211_2.json', 'case_InstrID_5166_2.json', 'case_InstrID_4633_0.json', 'case_InstrID_3696_0.json', 'case_InstrID_5043_1.json', 'case_InstrID_7213_1.json', 'case_InstrID_6310_0.json', 'case_InstrID_5356_0.json', 'case_InstrID_2365_1.json', 'case_InstrID_4134_2.json', 'case_InstrID_237_0.json', 'case_InstrID_2211_0.json', 'case_InstrID_107_0.json', 'case_InstrID_3965_1.json', 'case_InstrID_4001_0.json', 'case_InstrID_257_2.json', 'case_InstrID_7213_2.json', 'case_InstrID_446_0.json', 'case_InstrID_818_2.json']
        select = ['case_InstrID_2513_1.json', 'case_InstrID_6310_2.json', 'case_InstrID_2365_0.json', 'case_InstrID_2211_1.json', 'case_InstrID_5166_0.json', 'case_InstrID_818_0.json', 'case_InstrID_2513_0.json', 'case_InstrID_6310_1.json', 'case_InstrID_7213_0.json', 'case_InstrID_2365_2.json', 'case_InstrID_2211_2.json', 'case_InstrID_5166_2.json', 'case_InstrID_7213_1.json', 'case_InstrID_818_1.json', 'case_InstrID_2513_2.json', 'case_InstrID_6310_0.json', 'case_InstrID_5166_1.json', 'case_InstrID_2365_1.json', 'case_InstrID_2211_0.json', 'case_InstrID_7213_2.json', 'case_InstrID_818_2.json']
         
        for file in files:
            
            # for scene in passed_scene:
            #     if scene in file:
            #         continue
            if file in select:
                continue
            
            if file.endswith('.json'):
                with open(os.path.join('/nas-ssd2/shoubin/code/worldTTS/MapGPT/datasets/exprs_map/debug/preds/', file), 'r') as f:
                    preds = json.load(f)
                
                with open(os.path.join(baseline_dir, file), 'r') as f:
                    preds_baseline = json.load(f)
                    
                all_preds.append(preds)
                all_preds_baseline.append(preds_baseline)
                
        
        score_summary, _ = env.eval_metrics(all_preds_baseline, args.dataset)
        loss_str = "All cases  -"
        for metric, val in score_summary.items():
            loss_str += '  %s: %.2f' % (metric, val)
        print('Baseline:', loss_str)
        
        score_summary, _ = env.eval_metrics(all_preds, args.dataset)
        loss_str = "All cases  -"
        for metric, val in score_summary.items():
            loss_str += '  %s: %.2f' % (metric, val)
        print('Ours:', loss_str)
        # record_file = os.path.join(args.log_dir, 'valid.txt')
        # write_to_record_file(loss_str + '\n', record_file)
        

def main():
    args = parse_args()
    set_random_seed(args.seed)
    val_envs = build_dataset(args)
    valid(args, val_envs)


if __name__ == '__main__':
    main()
