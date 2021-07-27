import torch
import fairseq
from fairseq import trainer
from transformers import RobertaTokenizer


def roberta(vocab_dir, max_length=512, arch="roberta_base"):
    INPUT_ARGS = f"{vocab_dir} --task masked_lm \
        --criterion masked_lm --arch {arch} \
        --max-positions {max_length}"  # --shorten-method truncate
    parser = fairseq.options.get_training_parser()
    args = fairseq.options.parse_args_and_arch(
        parser, input_args=INPUT_ARGS.split())

    task = fairseq.tasks.setup_task(args)
    return task.build_model(args)


if __name__ == "__main__":
    model = roberta(vocab_dir="/home/ehoffer/PyTorch/fairseq_projects/roberta", arch="roberta_base")
    # x['net_input']['src_tokens']
    
    # inputs = {"input_ids"
    print(model)