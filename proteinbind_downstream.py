import torch
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOWNSTREAM_FILE_PATHS = 'downstream_tasks/downstream_files/'

def read_labels(file_path):
    with open(file_path, "r") as my_file:
        data_into_list = my_file.read().split("\n")
    labels = [1 if entry == 'PLP' else 0 if entry == 'BLB' else None for entry in data_into_list]
    return labels

def process_data(model, file_path):
    data = torch.load(file_path)
    output = model({'aa': data.type(torch.float32).to(device)})
    return output['aa']

def eval_linear_task(model):
    model.eval()

    # Load labels
    train_labels = read_labels(f"{DOWNSTREAM_FILE_PATHS}/train_mt/seq-cleaned-train_mt_labels.txt")
    test_labels = read_labels(f"{DOWNSTREAM_FILE_PATHS}/test_mt/seq-cleaned-test_mt-labels.txt")

    # Load and process data
    train_wt_aa = process_data(model, f'{DOWNSTREAM_FILE_PATHS}/train_wt/seq-cleaned-train_wt-embeddings.pt')
    train_mt_aa = process_data(model, f'{DOWNSTREAM_FILE_PATHS}/train_mt/seq-cleaned-train_mt_embeddings.pt')

    test_wt_aa = process_data(model, f'{DOWNSTREAM_FILE_PATHS}/test_wt/seq-cleaned-test_wt-embeddings.pt')
    test_mt_aa = process_data(model, f'{DOWNSTREAM_FILE_PATHS}/test_mt/seq-cleaned-test_mt-embeddings.pt')

    # Concatenate outputs
    train_concat = torch.cat((train_wt_aa, train_mt_aa), 1)
    test_concat = torch.cat((test_wt_aa, test_mt_aa), 1)

    # Linear classifier
    clf = LogisticRegression(random_state=0).fit(train_concat.detach().numpy(), train_labels)

    predictions = clf.predict(test_concat.detach().numpy())
    score = clf.score(test_concat.detach().numpy(), test_labels)

    cm = metrics.confusion_matrix(test_labels, predictions)
    print(cm)
    return score

def dms_eval_task(model):
    model.eval()

    # Load and process data
    dms_wt_aa = process_data(model, f'{DOWNSTREAM_FILE_PATHS}/DMS/data_wt/data-cleaned-wt-embeddings.pt')
    dms_mt_aa = process_data(model, f'{DOWNSTREAM_FILE_PATHS}/DMS/data_mt/data-cleaned-mt-embeddings.pt')

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(dms_wt_aa, dms_mt_aa)

    return output
