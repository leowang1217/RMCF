import os
import logging
import argparse
import torch
import pickle
# import json
import datetime
from tqdm import tqdm

from models import setup_model
from data.dataset import MultiRMCFDataset,SingleRMCFDataset
# from data.dataloader import GraphVAECollater
from trainer.training_args import TrainingArguments
from trainer.trainer import Trainer
from random import seed
from transformers.trainer_utils import get_last_checkpoint
from callback import RMCFCallback
from segmentor.frag_seg import FragSeg
from data.vocabulary import FragVocab
from processor.frag_processor import FragProcessor
from torch_geometric.loader.dataloader import Collater
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

def fix_seed():
    torch.manual_seed(1)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(43)


def load_dataset(args,vocab,split="train"):
    if split == "train":
        load_path=args.train_prefix
    elif split == "valid":
        load_path=args.valid_prefix
    elif split == "test":
        load_path=args.test_prefix
    else:
        raise ValueError("Split mush be train, valid or test.")
    if split=="train":
        train_slice = args.local_rank if not args.debug else 0
        dataset = MultiRMCFDataset( num_workers = args.num_workers,
                                    load_path = f'{load_path}{train_slice}.json',
                                    vocab = vocab,
                                    angle_inv=args.angle_intervals
                                ) 
    else:
        dataset = SingleRMCFDataset(load_path = f'{load_path}.json',
                                    vocab = vocab,
                                    angle_inv=args.angle_intervals
                                ) 

    return dataset

        
def main(args):
    # torch.distributed.init_process_group(backend='nccl', init_method='env://',timeout=datetime.timedelta(seconds=36000))
    print('----------------------------------------------------------------')
    fix_seed()
    # os.environ["CUDA_LAUNCH_BLOCKING"]='1'
    args.local_rank = int(os.environ["LOCAL_RANK"]) if not args.debug else -1
    print(args.local_rank)
    seg = FragSeg(hit_vocab=pickle.load(open(f'{args.seg_vocab_path}','rb')))
    vocab = FragVocab(vocab_file=pickle.load(open(f'{args.vocab_path}','rb')))
    processor  = FragProcessor(segmentor= seg, vocab =vocab )
    args.vocab_2d_size = vocab.get_2d_vocab_size()
    args.vocab_3d_size = vocab.get_3d_vocab_size()
    args.iface_size = vocab.get_iface_size()
    train_dataset,valid_dataset = load_dataset(args,vocab,split="train"), load_dataset(args,vocab,split="valid")
    
    
    model = setup_model(args,processor = processor)
    
    
    if args.debug:
        train_args = TrainingArguments(
            output_dir=args.model_dir,
            overwrite_output_dir = True,
            do_train=True,
            do_eval=False,
            report_to="none",
            ignore_data_skip=True,
            # run_name=run_name,
            max_grad_norm=1,
            max_steps=args.max_steps,
            per_device_train_batch_size = args.batch_size,
            per_device_eval_batch_size = args.batch_size,
            gradient_accumulation_steps=args.update_freq,
            learning_rate=args.learning_rate,
            num_train_epochs=1,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            evaluation_strategy="no",
            save_total_limit=args.save_total_limit,
            fp16=args.fp16,
            dataloader_num_workers=1,
            remove_unused_columns=True 
        )
    else:
        train_args =TrainingArguments(
            output_dir=args.model_dir,
            overwrite_output_dir = True,
            do_train=True,
            do_eval=False,
            report_to="none",
            # run_name=run_name,
            max_grad_norm=1,
            ignore_data_skip=True,
            max_steps=args.max_steps,
            per_device_train_batch_size = args.batch_size,
            per_device_eval_batch_size = args.batch_size,
            gradient_accumulation_steps=args.update_freq,
            learning_rate=args.learning_rate,
            num_train_epochs=1,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            evaluation_strategy="no",
            save_total_limit=args.save_total_limit,
            fp16=args.fp16,
            sharded_ddp= "simple" ,
            ddp_find_unused_parameters=True,
            local_rank=args.local_rank,
            dataloader_num_workers=args.num_workers,
            remove_unused_columns=True 
        )
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=Collater(None,None),
    )
    if args.debug:
        args.eval_steps = 60000 
    eval_callback = RMCFCallback(trainer=trainer,
                                config=args,
                                eval_rank=0 if not args.debug else -1
                                )

    trainer.add_callback(eval_callback)
    last_checkpoint = get_last_checkpoint(args.model_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # trainer.save_model()

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn', force=True)
    # TODO: 对args分组！
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-prefix', type=str)
    parser.add_argument('--valid-prefix', type=str)
    parser.add_argument('--test-prefix', type=str)
    parser.add_argument('--model-dir', type=str)


    parser.add_argument('--max-steps', type=int, default=360000)
    parser.add_argument('--batch-size', type=int, default=1280)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--logging-steps', type=int, default=500)
    parser.add_argument('--eval-steps', type=int, default=2500,)
    parser.add_argument('--save-steps', type=int, default=10000)
    parser.add_argument('--save-total-limit', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--update-freq', type=int, default=1, help='update')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank')

    parser.add_argument('--model-name', type=str, default='rmcf')
    parser.add_argument('--dim-h', type=int, default=512, help='dimension of the hidden')
    parser.add_argument('--dim-node', type=int, default=512, help='dimension of the nodes')
    parser.add_argument('--dim-edge', type=int, default=512, help='dimension of the edges')
    parser.add_argument('--mpnn-steps', type=int, default=3, help='number of mpnn steps')
    parser.add_argument('--num-attn-heads', type=int, default=8)
    
    parser.add_argument('--angle-intervals', type=float, default=5.0)
    parser.add_argument('--sampling-strategy', type=str, default='random',choices=['random', 'clustering'])
    parser.add_argument('--cov-thres', type=float, default=1.25)
    parser.add_argument('--vocab-path', type=str,default='geom-drugs/vocab.pkl')
    parser.add_argument('--seg-vocab-path', type=str,default='geom-drugs/hit.pkl')
    
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
   