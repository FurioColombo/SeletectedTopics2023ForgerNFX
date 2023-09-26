import os
import datetime
import shutil
import torch
import numpy as np
from training.config import config
from training.modules.models import loss
from training.modules.training import training
from training.scripts.evaluation import evaluate


class Trainer:

    def __init__(self, model, writer, run_name):
        self.model = model
        # https://github.com/Alec-Wright/Automated-GuitarAmpModelling/blob/main/dist_model_recnet.py
        print("Creating optimiser")
        self.optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        print("Creating optimizer's scheduler")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, 'min', factor=0.5, patience=5,
                                                                    verbose=True)
        print("Creating loss function")
        self.loss_functions = loss.LossWrapper()
        self.device = training.get_device()
        self.writer = writer
        self.run_name = run_name

        self.lowest_val_loss = 0
        self.best_loss = False
        self.start_time = datetime.datetime.now()
        self.epoch_time = None
        self.val_mse = np.zeros(shape=(1, 2))
        self.train_mse = np.zeros(shape=(1, 2))

    def train_model(self, dataloader):
        print("About to train")
        start_epoch = config.START_EPOCH or 0

        for epoch in range(start_epoch, config.MAX_EPOCHS):
            self._train_epoch(
                dataloader=dataloader,
                epoch=epoch,
            )

    def _train_epoch(self, dataloader, epoch):
        ep_loss = training.train_epoch_interval(
            self.model,
            dataloader,
            self.loss_functions,
            self.optimiser,
            device=self.device,
            sub_batch_seq_len=-1,
            warm_up_len=1000
        )

        # ep_loss = training.train_epoch(model, train_dl, loss_functions, optimiser, device=device)
        val_loss = training.compute_batch_loss(self.model, dataloader, self.loss_functions, device=self.device)
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("Loss/train", ep_loss, epoch)

        # check if we have beaten our best loss to date
        if self.lowest_val_loss == 0:  # first run
            self.lowest_val_loss = val_loss
        elif val_loss < self.lowest_val_loss:  # new record
            self.lowest_val_loss = val_loss
            self.best_loss = True
        else:  # no improvement
            self.best_loss = False
        if self.best_loss:  # save best model so far
            print("Record loss - saving at ", epoch)
            # save checkpoints for python and RTNeural
            checkpoint_path = self.model.save_model(
                out_dir=os.path.join(config.OUTPUT_FOLDER_PATH),
                curr_epoch=epoch,
                optimizer=self.optimiser,
                curr_loss=self.best_loss
            )
            evaluate.test_model(
                model=self.model,
                indata=dataloader,
                out_dir=checkpoint_path,
                file_name=self.run_name + str(epoch) + "singleNote.wav"
            )
            # if epoch % 10 == 0:  # save an example processed audio file
            evaluate.run_file_through_model(
                model=self.model,
                infile=config.TEST_FILE_PATH,
                out_dir=checkpoint_path,
                file_name=self.run_name + str(epoch) + ".wav"
            )

            self._save_training_data(
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                ep_loss=ep_loss,
                val_loss=val_loss
            )
            self._save_config(checkpoint_path=checkpoint_path)

        print("epoch, train, val ", epoch, ep_loss, val_loss)

    def _save_training_data(self, checkpoint_path, epoch, ep_loss, val_loss):
        # training data folder
        training_data_checkpoints_folder_path = os.path.join(checkpoint_path, 'training_data')  # /checkpoints/pytorch
        if not os.path.isdir(training_data_checkpoints_folder_path):
            os.mkdir(training_data_checkpoints_folder_path)

        self.epoch_time = datetime.datetime.now()
        self.train_mse = np.append(self.train_mse, np.array([[epoch, val_loss]]), axis=0)
        self.val_mse = np.append(self.val_mse, np.array([[epoch, ep_loss]]), axis=0)

        np.save(os.path.join(training_data_checkpoints_folder_path, 'train_mse'), self.train_mse)
        np.save(os.path.join(training_data_checkpoints_folder_path, 'eval_mse'), self.val_mse)

        training_time_s = (self.epoch_time - self.start_time).total_seconds()
        training_time_min = round(training_time_s / 60, 3)
        np.save(os.path.join(training_data_checkpoints_folder_path, 'training_time'), np.array(training_time_min))

    def _save_config(self, checkpoint_path):
        # config folder
        config_checkpoints_folder_path = os.path.join(checkpoint_path, 'config')  # /checkpoints/pytorch
        if not os.path.isdir(config_checkpoints_folder_path):
            os.mkdir(config_checkpoints_folder_path)
        shutil.copy(src=os.path.abspath(config.__file__), dst=config_checkpoints_folder_path)
