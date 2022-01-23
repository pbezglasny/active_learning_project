from datasets import load_dataset
from datasets import DatasetDict, Dataset


def explode_dataset(dataset):
    as_dict = dataset.to_dict()
    dialog, act, dialog_id = [], [], []
    for i, (d_arr, act_arr) in enumerate(zip(as_dict['dialog'], as_dict['act'])):
        if len(d_arr) != len(act_arr):
            raise ValueError('Size of arrays should be equal')
        dialog += d_arr
        act += act_arr
        dialog_id += [i] * len(d_arr)
    if len(dialog) != len(act):
        raise ValueError('')
    return {'dialog': dialog, 'act': act, 'dialog_id': dialog_id}


if __name__ == '__main__':
    dataset = load_dataset("daily_dialog")

    train = Dataset.from_dict(explode_dataset(dataset['train']))
    test = Dataset.from_dict(explode_dataset(dataset['test']))
    validation = Dataset.from_dict(explode_dataset(dataset['validation']))

    exploded_dataset = DatasetDict(train=train, test=test, validation=validation)

    exploded_dataset.save_to_disk('exploded_dataset')
