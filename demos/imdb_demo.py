import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

from opt_client import Client
# from datasets import load_dataset


def batcher(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def generate_dataset_activations(split, dataset):

    print("Generating Activations: " + split)

    client = Client(host="172.17.8.62", port="6969")

    module_names = [
        'decoder.layers.11.fc2'
    ]

    activations = []
    BATCH_SIZE = 16
    for batch in tqdm(batcher(dataset, BATCH_SIZE), total=len(dataset)/BATCH_SIZE):
        prompts = batch['text']
        activations.append(client.get_activations(prompts, module_names))

    parsed_activations = []
    for batch in activations:
        for prompt_activation in batch:
            parsed_activations.append(prompt_activation['decoder.layers.11.fc2'])

    cached_activations = {
        'activations': parsed_activations,
        'labels': dataset['label']
    }

    with open(split + '_activations_test.pkl', 'wb') as handle:
        pickle.dump(cached_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_imdb_activations():

    imdb = load_dataset("imdb")

    small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
    small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])

    generate_dataset_activations('train', small_train_dataset)
    generate_dataset_activations('test', small_test_dataset)


def load_activations(split):

    print("Loading Activations: " + split)

    with open(split + '_activations.pkl', 'rb') as handle:
        activations = pickle.load(handle)

    return activations

class ActivationDataset(Dataset):

    def __init__(self, activations, labels):
        self.activations = activations
        self.labels = labels

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]

def pad_batch(batch):
    (x, y) = zip(*batch)
    
    # Return only last token
    sequence_lengths = [seq.shape[0] - 1 for seq in x]
    x = torch.stack([seq[-1] for seq in x])

    # Return padded sequence
    # PAD_TOKEN = -1000
    # x = rnn.pad_sequence(x, batch_first=True, padding_value=PAD_TOKEN)

    return x, y, sequence_lengths

class SequenceClassifier(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg['embedding_dim'], cfg['hidden_dim'], bias=False)
        self.out = nn.Linear(cfg['hidden_dim'], cfg['label_dim'])

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.out(x)  
        return x

if __name__ == "__main__":

    # generate_imdb_activations()

    cached_train_activations = load_activations('train')
    cached_test_activations = load_activations('test')

    train_dataset = ActivationDataset(cached_train_activations['activations'], cached_train_activations['labels'])
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=pad_batch)

    test_dataset = ActivationDataset(cached_test_activations['activations'], cached_test_activations['labels'])
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=pad_batch)

    model = SequenceClassifier({
        "embedding_dim": 768,
        "hidden_dim": 128,
        "label_dim": 2
    })
    model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    NUM_EPOCHS = 10
    pbar = tqdm(range(NUM_EPOCHS))
    for epoch_idx in pbar:
        
        pbar.set_description("Epoch: %s" % epoch_idx)
        training_params = {
            "Train-Loss": 0.0,
            "Test-Accuracy": 0.0
        }
        pbar.set_postfix(training_params)

        model.train()
        for batch in train_dataloader:

            activations, labels, seq_lengths = batch
            activations = activations.float().cuda()
            labels = torch.tensor(labels).cuda()

            optimizer.zero_grad()

            logits = model(activations)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            training_params["Train-Loss"] = loss.detach().item()
            pbar.set_postfix(training_params)

        model.eval()
        with torch.no_grad():
            predictions = []
            for batch in test_dataloader:
                activations, labels, seq_lengths = batch
                activations = activations.float().cuda()
                labels = torch.tensor(labels).cuda()

                logits = model(activations)
                predictions.extend((logits.argmax(dim=1) == labels)) 


            accuracy = torch.stack(predictions).sum() / len(predictions)
            
            training_params["Test-Accuracy"] = accuracy.detach().item()
            pbar.set_postfix(training_params)
