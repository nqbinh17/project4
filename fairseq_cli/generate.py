#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)


    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
    
    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)
    
    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x
    obj = {
    'graph_transformer': {'path': "/content/graph_transformer_st_valid1_full.out", 'name': 'graph_transformer'},
    'scaled_transformer': {'path': "/content/scaled_transformer_valid1_full.out", 'name': 'scaled_transformer'},
    'standard_transformer': {'path': "/content/standard_transformer_valid1_full.out", 'name': 'standard_transformer'}
    }
    import re
    def clean_ref(string):
      string = ' '.join(string.replace('\n', '').split()[1:])
      return string

    def clean_cand(string):
      string = ' '.join(string.replace('\n', '').split()[2:])
      return string

    # Clean & Store references and candidate sentences for each model

    for key in obj.keys():
      holder = obj[key]
      with open(holder['path'], 'r') as f:
        data = f.readlines()
        holder['data'] = data 
      assert len(data) % 5 == 0
      holder['ref'] = []
      holder['cand'] = []
      for i in range(0, len(data), 5):
        tok_ref = tgt_dict.encode_line(
            clean_ref(data[i+1]), add_if_not_exist=True
        )
        tok_cand = tgt_dict.encode_line(
            clean_cand(data[i+2]), add_if_not_exist=True
        )
        holder['ref'].append(tok_ref)
        holder['cand'].append(tok_cand)
    
    from numpy.random import default_rng

    # generate 10000 random indices (80% of samples) for significance test

    rng = default_rng()
    numbers = rng.choice(20, size=10, replace=False)
    size = len(holder['ref'])
    print('total samples', size)
    def indices(size, n=10000, p=0.8):
      num_remove = int(size * p)
      random_indices = [rng.choice(size, size=num_remove, replace=False) for _ in range(n)]
      return random_indices
    index = np.array(indices(size), dtype=int)
    obj['significance_id'] = index
    # compute bleu score for each sample in significance test
    for key in obj.keys():
      if type(obj[key]) != dict:
        continue

      holder = obj[key]
      holder['significance_test'] = []

      ref_array = np.array(holder['ref'], dtype=object)
      cand_array = np.array(holder['cand'], dtype=object)
      for idx in obj['significance_id']:
        scorer = scoring.build_scorer(cfg.scoring, tgt_dict)
        
        new_ref = ref_array[idx]
        new_cand = cand_array[idx]
        for ref, cand in zip(new_ref, new_cand):
          scorer.add(ref, cand)
        holder['significance_test'].append(scorer.score())
      holder['significance_test'] = np.array(holder['significance_test'])

    # Final results
    def compare_significance_test(obj1, obj2):
      total = len(obj1['significance_test'])
      def format(num):
        return round(num, 2)
      def to_string(mean, std):
        return '{}±{}'.format(mean, std/2)
      v1 = obj1['significance_test']
      v2 = obj2['significance_test']
      mean1, std1 = format(v1.mean()), format(np.std(v1, axis=0))
      mean2, std2 = format(v2.mean()), format(np.std(v2, axis=0))
      cnt = sum(v1 >= v2)
      
      return {
          obj1['name']: {'mean': mean1, 'std': std1, 'str': to_string(mean1, std1)},
          obj2['name']: {'mean': mean2, 'std': std2, 'str': to_string(mean2, std2)},
          'significance': format(100 * cnt / total),
          'samples': total
      }
    print(compare_significance_test(obj['graph_transformer'], obj['standard_transformer']))
    print(compare_significance_test(obj['graph_transformer'], obj['scaled_transformer']))


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
