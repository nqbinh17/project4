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
import pymeteor.pymeteor as pymeteor

class MeteorScorer:
    def __init__(self):
        self.cnt_str = 0
        self.total_meteor_str = 0.0

        self.cnt_tok = 0
        self.total_meteor_tok = 0.0

    def to_string(self, tensor):
        return ' '.join(map(lambda x: str(x), tensor.tolist()))

    def add_string(self, target, hypo):
        self.cnt_str += 1
        self.total_meteor_str += pymeteor.meteor(target, hypo)

    def add_tok(self, target_tok, hypo_tok):
        self.cnt_tok += 1
        self.total_meteor_tok += pymeteor.meteor(self.to_string(target_tok), self.to_string(hypo_tok))

    def score_str(self):
        return '{:.4f}'.format(100 * self.total_meteor_str / self.cnt_str)

    def score_tok(self):
        return '{:.4f}'.format(100 * self.total_meteor_tok / self.cnt_tok)

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

    with open(cfg.task.significance_index_path, "r") as f:
      datas = f.readlines()
      significance_index_datas = []
      for d in datas:
        processed = list(map(lambda x: int(x), d.split()))
        significance_index_datas.append(processed)

    for i, significance_index_data in enumerate(significance_index_datas):
        print("Significance Testing {}".format(i+1))
        task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task, significance_index_data = significance_index_data)
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
        scorer = scoring.build_scorer(cfg.scoring, tgt_dict)
        meteor_scorer = MeteorScorer()

        num_sentences = 0
        has_target = True
        wps_meter = TimeMeter()
        for sample in progress:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if "net_input" not in sample:
                continue

            prefix_tokens = None
            if cfg.generation.prefix_size > 0:
                prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

            constraints = None
            if "constraints" in sample:
                constraints = sample["constraints"]

            gen_timer.start()
            hypos = task.inference_step(
                generator,
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
            )
            num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample["id"].tolist()):
                has_target = sample["target"] is not None

                # Remove padding
                if "src_tokens" in sample["net_input"]:
                    src_tokens = utils.strip_pad(
                        sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                    )
                else:
                    src_tokens = None

                target_tokens = None
                if has_target:
                    target_tokens = (
                        utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                    )

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
                        sample_id
                    )
                    target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                        sample_id
                    )
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(
                            target_tokens,
                            cfg.common_eval.post_process,
                            escape_unk=True,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                                generator
                            ),
                        )

                src_str = decode_fn(src_str)
                if has_target:
                    target_str = decode_fn(target_str)

                # Process top predictions
                for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=src_str,
                        alignment=hypo["alignment"],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=cfg.common_eval.post_process,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                    )
                    detok_hypo_str = decode_fn(hypo_str)
                    if not cfg.common_eval.quiet:
                        score = hypo["score"] / math.log(2)  # convert to base 2

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or cfg.common_eval.post_process is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(
                                target_str, add_if_not_exist=True
                            )
                            hypo_tokens = tgt_dict.encode_line(
                                detok_hypo_str, add_if_not_exist=True
                            )
                        if hasattr(scorer, "add_string"):
                            scorer.add_string(target_str, detok_hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)
                        
                        meteor_scorer.add_string(target_str, detok_hypo_str)
                        meteor_scorer.add_tok(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            progress.log({"wps": round(wps_meter.avg)})
            num_sentences += (
                sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
            )
        if has_target:

            print(
                "Generate {} with beam={}: {}, meteor_tok: {}, meteor_str: {}".format(
                    cfg.dataset.gen_subset, cfg.generation.beam, scorer.result_string(), meteor_scorer.score_str(), meteor_scorer.score_tok()
                ),
                file=output_file,
            )

    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
