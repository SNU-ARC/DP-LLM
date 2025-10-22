import torch
from any_precision import DPLLMForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from any_precision.evaluate.helpers import dataloader
from any_precision.evaluate.helpers.utils import vprint, logprint

@torch.no_grad()
def auto_model_load(model_path, device='cuda', dtype=torch.float16, verbose=True, precisions=[3,4,5,6],
                    estimator_results=None):
    """
    Args:
        model_path: path of the model to evaluate
        device: the device to use for evaluation, either 'cuda' or 'cpu'
        dtype: the dtype to use for evaluation, either torch.float16 or torch.float32
        verbose: whether to print progress
        precisions: list of precisions to use
        estimator_results: path to estimator results

    Returns:
        (tokenizer_type, tokenizer, model) tuple loaded from the given path, with the given device and dtype.
    """
    logprint(verbose, "Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    max_mem_dict = torch.load(os.path.join(estimator_results, "max_mem_dict.pt"))
    linear_reg_d = torch.load(os.path.join(estimator_results, "linear_reg_d.pt"))
    jl_d = torch.load(os.path.join(estimator_results, "jl_d.pt"))
    T_d = torch.load(os.path.join(estimator_results, "T_d.pt"))
    model = DPLLMForCausalLM.from_quantized(model_path, precisions=precisions, prefill_by_decode=True,
                                            max_mem_dict=max_mem_dict, linear_reg_d=linear_reg_d, jl_d=jl_d, T_d=T_d)
    
    model = model.eval().cuda()

    logprint(verbose, f"{model_path} model loaded to device: {model.device}")

    tokenizer_type = None

    return tokenizer_type, tokenizer, model


@torch.no_grad()
def evaluate_ppl(model, tokenizer, testcases, verbose=True, chunk_size=2048, tokenizer_type=None):
    """
    Args:
        model: model to evaluate
        tokenizer: tokenizer to use
        testcases: testcases names to evaluate on, passed on to dataloader.get_loaders
        verbose: whether to print progress
        chunk_size: the size of the chunks into which the test set is split
        tokenizer_type: set to llama, llama-2, or opt to use cached input tokens
                        for the corresponding test set

    Returns:
        A dictionary of perplexity scores, with keys being the testcases names and values being the perplexity scores.

    Note that the perplexity scores are calculated over non-overlapping chunks of the test set.
    """

    results = {}

    for testcase_name in testcases:
        vprint(verbose, f"---------------------- {testcase_name} ----------------------")
        try:
            model.clear_comp_count()
            logprint(verbose, f"<<<< Resetting comp count >>>>")
        except:
            pass

        input_tokens = load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose)

        input_tokens.to("cuda:0")

        logprint(verbose, "Calculating perplexity...")

        seq_len = input_tokens.input_ids.size(1)
        nsamples = seq_len // chunk_size  # floor(seq_len / chunk_size)

        neg_log_likelihoods = []
        for i in tqdm(range(nsamples), disable=not verbose):
            begin_loc = i * chunk_size

            input_ids = input_tokens.input_ids[:, begin_loc:begin_loc + chunk_size]

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss
                neg_log_likelihoods.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(neg_log_likelihoods).mean())
        logprint(verbose, f"Perplexity: {ppl.item()}")

        results[f"{testcase_name}"] = ppl.item()
        try:
            eb = model.get_effective_bits()
            results[f"{testcase_name}_eb"] = eb
            logprint(verbose, f"effective bits: {eb}")
        except:
            pass

    return results


def load_input_tokens(tokenizer_type, testcase_name, tokenizer, verbose):
    logprint(verbose, "Loading test set...")

    raw_text = dataloader.get_loaders(testcase_name)

    logprint(verbose, "Tokenizing test set...")

    input_tokens = tokenizer(raw_text, return_tensors='pt')

    return input_tokens


import json
import argparse

parser = argparse.ArgumentParser(description="Test perplexity for DP-LLM.")
parser.add_argument("ap_model_path", type=str,
                    help="Path to AnyPrecision model.")
parser.add_argument('--estimator_results', type=str, required=True,
                    help="Path to finetuned results. For default, try using ./estimator_private_values/<ap_model_name>/<finetuned...>")

parser.add_argument("--min_prec", type=int, default=3,
                    help="Minimum precision to utilize.")
parser.add_argument("--max_prec", type=int, default=6,
                    help="Maximum precision to utilize.")


parser.add_argument('--output_file', type=str, default='pp_results.json',
                    help="Path to perplexity result output file.")

args = parser.parse_args()

prec_arr = [bit for bit in range(args.min_prec, args.max_prec+1)]


datasets = ['wikitext2', 'c4_new']

# read previous results
if os.path.exists(args.output_file):
    with open(args.output_file) as f:
        all_results = json.load(f)
else:
    all_results = {}

new_results = {}  # results that are newly calculated, to be printed at the end

total_tests_to_run = {}  # tasks to be run will be stored here
skipped_models = []  # models that are skipped will be stored here

# Check which models/testcases need to be run
# This is done first so that we know how many tasks there are in total,
# and thus we can print the progress
for (model_path, estimator_results) in [(args.ap_model_path, args.estimator_results)]:
    model_jobs = {'to_print': [], 'ppl': []}

    datasets_with_results = [dataset for dataset in datasets if all_results.get(f"{model_path}_{estimator_results}")]

    model_jobs['ppl'] = [testcase for testcase in datasets if testcase not in datasets_with_results]
    if not model_jobs['ppl']:
        # All results of the target model/testcases and model/tasks combination exist, skip
        skipped_models.append(model_path)
        continue
    else:
        if datasets_with_results:
            model_jobs['to_print'].append(f"Skipping datasets: "
                                            f"{datasets_with_results} because results already exist")
            
    model_jobs['to_print'].append(f"Running datasets: {model_jobs['ppl']}")
    total_tests_to_run[(model_path, estimator_results)] = model_jobs

total_ppl_job_count = sum(len(model_tasks['ppl']) for model_tasks in total_tests_to_run.values())
if skipped_models:
    print(f">> {len(skipped_models)} models will be skipped because all dataset results already exist.")

# Run all tasks
for i, (model_path, estimator_results) in enumerate(total_tests_to_run):
    model_jobs = total_tests_to_run[(model_path, estimator_results)]
    to_print = model_jobs['to_print']
    datasets_to_evaluate = model_jobs['ppl']
    print("==================================================")
    print(f" Model: {model_path}, Prec: {prec_arr}")
    print(f" Finetuned: {estimator_results}")
    print(f"Progress: {i + 1}/{len(total_tests_to_run)}")
    print("==================================================")
    datasets_with_results = [testcase for testcase in datasets if testcase in all_results.get(model_path, {})]

    for line in to_print:
        print('>> ' + line)

    ppl_results = {}

    # Run evaluation
    tokenizer_type, tokenizer, model = auto_model_load(model_path, precisions=prec_arr, estimator_results=estimator_results)

    if datasets_to_evaluate:
        ppl_results = evaluate_ppl(model, tokenizer, datasets_to_evaluate, verbose=True,
                                    chunk_size=2048, tokenizer_type=tokenizer_type)
    # Update new results
    new_results[f"{model_path}_{estimator_results}"] = {}
    if ppl_results:
        new_results[f"{model_path}_{estimator_results}"]['ppl'] = ppl_results

    # read previous results
    if os.path.exists(args.output_file):
        with open(args.output_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Update all results
    if ppl_results:
        all_results.setdefault(f"{model_path}_{estimator_results}", {}).setdefault('ppl', {}).update(ppl_results)

    # save results
    with open(args.output_file, 'w') as f:
        all_results = dict(sorted(all_results.items()))  # sort by key
        json.dump(all_results, f, indent=4)

    print()

    del model  # clear memory
    torch.cuda.empty_cache()


print("---------------------- All Results ----------------------")
# print new results as formatted json
print(json.dumps(new_results, indent=4))

if len(total_tests_to_run) == 0:
    exit(1)
