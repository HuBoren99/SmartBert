from telnetlib import GA
from transformers import BertConfig,BertTokenizer
from transformers.modeling_bert import BertModel
import argparse
import logging
import os
import torch
import numpy as np
import random
import time
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from thop import profile
from transformers import glue_compute_metrics as compute_metrics
from models import SmartBert
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def load_and_cache_examples(args, task, tokenizer, evaluate=False):

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm']:
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode= output_mode
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    if evaluate == False:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def train(args, train_dataset, model, tokenizer, is_first_stage = False,special_training_mode = None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)
    if is_first_stage==False:
        # second_stage
        t_total = len(train_dataloader) * args.second_stage_train_nums
    else:
        t_total = len(train_dataloader) * args.first_stage_train_nums
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay)],
        'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if is_first_stage==False:
        logger.info("***** Running Second Stage*****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.second_stage_train_nums)
        model.is_hard_weight = True
    else:
        logger.info("***** Running First Stage *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.first_stage_train_nums)          
    logger.info(" Total optimization steps = %d", t_total)
    train_iterator = (trange(int(args.second_stage_train_nums), desc="Epoch") if is_first_stage==False else trange(int(args.first_stage_train_nums), desc="Epoch"))
    model.zero_grad()
    set_seed(args)
    best_acc = -1
    for _ in train_iterator:
        if is_first_stage:
            if special_training_mode:   
                if _ <= 2:
                    model.set_hard_weight_mechanism(False)
                else:
                    model.set_hard_weight_mechanism(args.is_hard_weight)
            else:
                # only use the hard_weight mechanism
                model.set_hard_weight_mechanism(args.is_hard_weight)

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        total_loss = 0.0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if is_first_stage:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          'train_first_stage': True}
            else:
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            if step % args.log_step == 0:
                logger.info('step {}: loss = {}'.format(step,total_loss/args.log_step))
                total_loss = 0.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        if is_first_stage:
            result = evaluate(args, model, tokenizer, is_exit = False)
            result = get_wanted_result(result)
            if result >= best_acc:
                best_acc = result
                # save the model
                torch.save(model.state_dict(), args.output_path+'/first_model.bin')
    


def get_wanted_result(result):
    if "spearmanr" in result:
        print_result = result["spearmanr"]
    elif "f1" in result:
        print_result = result["f1"]
    elif "mcc" in result:
        print_result = result["mcc"]
    elif "acc" in result:
        print_result = result["acc"]
    else:
        print(result)
        exit(1)
    return print_result

def evaluate(args, model, tokenizer, is_exit = False):
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)
    # Eval!
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    total_flops = 0
    skipped_layer_count = {i+1:0 for i in range(model.num_layers)}
    if is_exit:
        exited_layer_count = {i+1:0 for i in range(model.num_layers)}
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3],
                      'is_exit':           is_exit
                      }
            if is_exit:
                outputs,skipped_layer_index,exited_layer_index = model(**inputs)
                for item in exited_layer_index:
                    exited_layer_count[item]+=1
                input_profile = (inputs['input_ids'],inputs['attention_mask'],inputs['token_type_ids'],None,is_exit,inputs['labels'])
                flops,params = profile(model, input_profile,verbose=False)
                total_flops += flops
            else:
                outputs,skipped_layer_index = model(**inputs)
            
            for item in skipped_layer_index:
                skipped_layer_count[item]+=1
            
            
            logits = outputs

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)   

    logger.info("skipped layers:{}".format(skipped_layer_count))

    preds = np.argmax(preds, axis=1)
    result = compute_metrics(args.task_name, preds, out_label_ids)
    if is_exit:
        logger.info("exited layers:{}".format(exited_layer_count))   
        logger.info("average FLOPs: %.2fM"%(total_flops/len(eval_dataloader)/1e6))   
        result['floaps'] = total_flops/len(eval_dataloader)/1e6  
    logger.info("Result: {}".format(result))
    
    return result

def init_model(model,args):
    # the hyper-parameters of skipping gates
    model.set_lamada(args.lamada)
    model.set_skipped_rate(args.skipped_rate)
    model.set_hard_weight_mechanism(args.is_hard_weight)
    # the hyper-parameters of contrastive learning
    model.set_eta1(args.eta1)
    model.set_eta2(args.eta2)
    model.set_t1(args.t1)
    model.set_t2(args.t2)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda_id", default='1', type = str,
                        help="Using the id of the cuda")
    parser.add_argument("--lamada", default=0.05, type=float,
                        help="the rate for inspiring the skipping gate")
    parser.add_argument("--eta1", default=0.1, type=float,
                        help="the parameters for the first stage of contrastive learning")
    parser.add_argument("--eta2", default=0.5, type=float,
                        help="the parameters for the second stage of contrastive learning")
    parser.add_argument("--t1", default=0.5, type=float,
                        help="the temperature for the first stage of contrastive learning")
    parser.add_argument("--t2", default=0.55, type=float,
                        help="the temperature for the second stage of contrastive learning")
    parser.add_argument("--special_training_mode", default=True, type = bool,
                        help = "The special training trick")
    parser.add_argument("--is_hard_weight", default=True, type = bool,
                        help="Whether or not using the hard_weight_mechanism for the skipping gate")
    parser.add_argument("--skipped_rate", default=0.5, type = float,
                        help="the rate for the skipping")   
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name", type=str, required=True,
                        help="The name of task for training")
    parser.add_argument("--output_path", default='./output/smartbert_{}_{}_{}_{}_{}_{}_{}', type=str, required=False,
                        help="The path for saving the bert")
    parser.add_argument("--do_train", default=True, type = bool, 
                        help="whether to train the model")
    parser.add_argument("--do_eval", default=True, type = bool, 
                        help="whether to evaluate the model")  

    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False,
                        help="Path to pre-trained model or shortcut name such as bert-base-uncased")

    parser.add_argument('--seed', type=int, default=50,
                        help="random seed for initialization")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--first_stage_train_nums", type=int, default=5,
                        help="the nums of first stage")
    parser.add_argument("--second_stage_train_nums", type=int, default=4,
                        help="the nums of second stage")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-9, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--log_step", default=100, type=int,
                        help="the logging step for training")
    parser.add_argument("--max_seq_length", default=128, type = int,
                        help = "the max length of sequence")
    args = parser.parse_args()
    args.output_path = args.output_path.format(args.skipped_rate,args.task_name,('hard' if args.is_hard_weight else 'soft'),args.learning_rate,('special_mode'if args.special_training_mode else 'not_special'),args.lamada,args.seed)
    set_seed(args)


    # Check output_dir
    if os.path.exists(args.output_path) == False:
        os.makedirs(args.output_path)
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    rq = 'smartbert'+ '_' + ('special_mode'if args.special_training_mode else 'not_special') +'_' + str(args.skipped_rate) +'_'+('hard' if args.is_hard_weight else 'soft')+'_'+args.task_name.lower() +'_'+str(args.learning_rate) + '_' + str(args.lamada) + '_' + str(args.seed) + '_'+time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = './logs/'
    if os.path.exists(log_path) == False:
        os.mkdir(log_path)
    log_name = log_path + rq + '.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)  
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("Args:{}".format(args))

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config = BertConfig.from_pretrained(args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          output_attentions=True,
                                        )
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertModel.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        )
    model = SmartBert(model,config)
    # set cuda
    device = torch.device("cuda:"+args.cuda_id if torch.cuda.is_available() else "cpu")
    args.device = device
    model.to(device)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        init_model(model,args)        
        train(args, train_dataset, model, tokenizer,is_first_stage = True, special_training_mode=args.special_training_mode)
        model.load_state_dict(torch.load(args.output_path + '/first_model.bin', map_location='cpu'), strict=False)
        init_model(model,args)
        train(args, train_dataset, model, tokenizer,is_first_stage = False)
        # save the final model
        torch.save(model.state_dict(), args.output_path + '/final_model.bin')
    
    # evaluate
    if args.do_eval:
        model.load_state_dict(torch.load(args.output_path + '/final_model.bin', map_location='cpu'), strict=False)
        init_model(model,args)
        model.set_skipped_rate(1.0)
        brr = []
        flag = False
        for speed in [0,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]:
            if speed == 0:
                if flag == False:
                    flag = True 
                else:
                    model.set_skipped_rate(args.skipped_rate)
            model.threshold = speed
            logger.info("speed:{}".format(speed))
            result = evaluate(args, model, tokenizer, is_exit = True) 
            if "f1" in result:
                brr.append({'f1':result['f1'],'floaps':result['floaps']})
            elif "mcc" in result:
                brr.append({'mcc':result['mcc'],'floaps':result['floaps']})
            elif "acc" in result:
                brr.append({'acc':result['acc'],'floaps':result['floaps']})
        np.save(args.output_path+'/'+args.task_name + '_' + str(args.learning_rate),brr)
if __name__ == '__main__':
    main()