import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from data import load_data
from model import DiffusionModel_FNN, DiffusionModel_LSTM
from config import get_config
from transformers import logging, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            self.base_model = AutoModel.from_pretrained('roberta-base')
        else:
            raise ValueError('unknown model')

        if args.method_name == 'fnn':
            self.model = DiffusionModel_FNN(self.base_model, args.num_classes)
        elif args.method_name == 'lstm':
            self.model = DiffusionModel_LSTM(self.base_model, args.num_classes)
        else:
            raise ValueError('unknown method')

        self.model.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer,scheduler):
        train_loss, n_correct, n_train = 0, 0, 0

        self.model.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            predicts = self.model(inputs)
            loss = criterion(predicts, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)

        return train_loss / n_train, n_correct / n_train

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        self.model.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                predicts = self.model(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)

        return test_loss / n_test, n_correct / n_test

    def run(self):
        train_dataloader, test_dataloader = load_data(dataset=self.args.dataset,
                                                      data_dir=self.args.data_dir,
                                                      tokenizer=self.tokenizer,
                                                      train_batch_size=self.args.train_batch_size,
                                                      test_batch_size=self.args.test_batch_size,
                                                      model_name=self.args.model_name,
                                                      method_name=self.args.method_name,
                                                      workers=0)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        # Define the criterion
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay,eps=self.args.eps)
        # Warm up
        total_steps = len(train_dataloader) * self.args.num_epoch
        warmup_steps = 0.1 * len(train_dataloader)
        scheduler =get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        best_loss, best_acc = 0, 0

        l_acc, l_epo = [], []
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer,scheduler)
            test_loss, test_acc = self._test(test_dataloader, criterion)
            l_epo.append(epoch), l_acc.append(test_acc)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
                # Save model
                # torch.save(self.model.state_dict(), './model.pkl')
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc * 100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc * 100))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        plt.plot(l_epo, l_acc)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('image.png')
        plt.show()


if __name__ == '__main__':
    logging.set_verbosity_error()

    # ??????????????????
    args, logger = get_config()

    # ???????????????????????????
    ins = Instructor(args, logger)

    # ??????????????????
    ins.run()
