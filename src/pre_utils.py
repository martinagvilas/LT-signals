import json

import pandas as pd


def load_eureka_report(model_name, dataset, path, n_repeats=5):
    """
    Load the eureka report for a specific model and dataset.

    Args:
        model_name (str): The name of the model.
        dataset (str): The name of the dataset.
        path (Path): The path to the project folder.
        n_repeats (int): The number of repeats to consider.

    Returns:
        pd.DataFrame: The loaded eureka report data.
    """

    # Open report
    report = path / f'results/eureka_reports/{dataset}/{model_name}.jsonl'
    with open(report, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    data = pd.DataFrame(data)

    # Filter by number of repeats
    data = data.loc[data['data_repeat_id'].isin([f'repeat_{i}' for i in range(n_repeats)])]

    # Sample from different question types if MAZE or TSP
    if dataset == 'MAZE':
        ids = (data.drop_duplicates('data_point_id')[['data_point_id', 'question_type']]
                .groupby('question_type', group_keys=False)
                .apply(lambda g: g.sample(n=40, random_state=0))
                ['data_point_id'])
        data = data[data['data_point_id'].isin(ids)]
    elif dataset == 'TSP':
        ids = (data
                .drop_duplicates('data_point_id')[['data_point_id', 'category']]
                .groupby('category', group_keys=False)
                .apply(lambda g: g.sample(n=20, random_state=0))
                ['data_point_id']
                )
        data = data[data['data_point_id'].isin(ids)]

    return data