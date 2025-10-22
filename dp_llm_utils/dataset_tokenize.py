from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Define datasets here.
load_dataset_lambda_d = {
    # 10000 samples is enough for c4
    "c4": lambda : load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train[:10000]'),
    "wikitext2": lambda : load_dataset('wikitext', 'wikitext-2-raw-v1', split='train'),
}

def getPossibleDatasets() -> list[str]:
    """
    Get a list of defined datasets.

    Returns:
        a list of defined datasets
    """
    return [name for name in load_dataset_lambda_d.keys()]

def getDataLoader(dataset_name:str, tokenizer:AutoTokenizer,
                  context_length:int, dataset_length:int)->DataLoader:
    """
    Get a DataLoader for the given dataset name.

    Args:
        dataset_name: name of dataset to load
        tokenizer: tokenizer to use
        context_length: context length to use
        dataset_length: number of samples to use

    Returns:
        DataLoader for the given dataset name
    """

    if dataset_name not in load_dataset_lambda_d.keys():
        raise RuntimeError(f"Unknown dataset {dataset_name}.")
    
    print(f"Loading Dataset {dataset_name}...", end="", flush=True)
    dataset = load_dataset_lambda_d[dataset_name]()
    print("Done.")

    # remove empty texts from dataset
    dataset = dataset.filter(lambda e: len(e['text']) >= 1) 

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_dataset = dataset.map(
        tokenize, batched=True, remove_columns=dataset.column_names
    )

    training_dataset = tokenized_dataset.select(range(dataset_length))
    training_dataset.set_format(type='torch')
    
    return DataLoader(training_dataset, batch_size=1)