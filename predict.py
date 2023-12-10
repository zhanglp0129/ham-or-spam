import torch
import models
import data
import conf

def predict(model:models.HamSpamModel, sentence:str)->str:
    model.eval()
    sequence = data.dictionary.sentence2Sequence(sentence)
    input = torch.tensor(sequence)
    input = input.to(conf.device)
    input = input.unsqueeze(0)
    with torch.no_grad():
        pred = model(input)
        pred = 'spam' if (pred>0.5).item() else 'ham'
    return pred

if __name__ == "__main__":
    model = models.HamSpamModel()
    model.load_state_dict(torch.load(conf.model_load_path, map_location=conf.device))
    model = model.to(conf.device)
    with open('./data/testdata.csv', 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            if line[-1] == '\n' or line[-1] == '\r':
                line = line[:-1]
            if predict(model, line) == 'spam':
                print(line)