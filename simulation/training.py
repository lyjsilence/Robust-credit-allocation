import time
import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim


def train_DR_model(args, model, dl_train, dl_test, device, alpha_input=None, exp_comp=False, verbose=True):
    best_loss, best_model = float('inf'), None
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5,
                                                     patience=10, verbose=False, min_lr=1e-4)
    # loss function
    loss_fun = nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tic = time.time()

        loss_train_list = []

        for i, train_data in enumerate(dl_train):
            X_train, y_train = train_data[0], train_data[1]
            X_train, y_train = X_train.to(torch.float32).to(device), y_train.to(torch.float32).to(device)

            pred_train = model(X_train[:, 1:], X_train[:, 0].reshape(-1, 1))
            loss_train = loss_fun(pred_train, y_train)

            loss_train_list.append(loss_train.cpu().detach().numpy())

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            if verbose:
                print("\rEpoch [{}/{}], Batch [{}/{}], Loss = {:.4f}, time elapsed = {:.2f}, "
                      .format(epoch, args.epochs, i + 1, len(dl_train),
                              np.sum(loss_train_list) / len(loss_train_list), time.time() - tic), end="")

        with torch.no_grad():
            model.eval()
            for test_data in dl_test:
                X_test, y_test = test_data[0], test_data[1]
                X_test, y_test = X_test.to(torch.float32).to(device), y_test.to(torch.float32).to(device)
                pred_y_test = model(X_test[:, 1:], X_test[:, 0].reshape(-1, 1))
                loss_test = loss_fun(pred_y_test, y_test)

            if verbose:
                print("Test: MAE={:.4f}".format(loss_test.cpu().detach().numpy()))

            if loss_test.item() < best_loss:
                best_loss = loss_test.item()
                # best_model = copy.deepcopy(model)
                best_model = model.state_dict()

        scheduler.step(loss_test)

    if best_model is not None:
        prefix = "exponential_" if exp_comp else "tr_mlp_DR_"
        filename = f"{prefix}dis_Y_alpha={alpha_input}.pth" if exp_comp else f"{prefix}dis.pth"

        full_path = os.path.join("./result", filename)
        torch.save(best_model, full_path)
        return full_path


def train_Prob_model(args, model, dl_train, dl_test, device, prop_score, verbose=True):
    best_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5,
                                                     patience=10, verbose=True, min_lr=1e-4)
    best_model = None

    softmax_fun = nn.Softmax(dim=1)
    loss_fun = nn.CrossEntropyLoss()
    for epoch in range(1, args.density_epochs + 1):
        model.train()
        tic = time.time()

        loss_train_list = []
        acc_train_list = []

        for i, train_data in enumerate(dl_train):
            X_train, y_train = train_data[0], train_data[1]
            X_train, y_train = X_train.to(torch.float32).to(device), y_train.to(torch.float32).to(device)
            # predict the probability
            pred_train = model(X_train)
            loss_train = loss_fun(pred_train, torch.squeeze(y_train).long())

            loss_train_list.append(loss_train.cpu().detach().numpy())
            # Compute the accuracy
            forecasted_prob_train = softmax_fun(pred_train)
            _, forecasted_train = torch.max(forecasted_prob_train, 1)
            correct = (forecasted_train == torch.squeeze(y_train).long()).sum().item()
            accuracy = 100 * correct / y_train.size(0)
            acc_train_list.append(accuracy)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            if verbose:
                print("\rEpoch [{}/{}], Batch [{}/{}], Loss = {:.4f}, Acc = {:.4f}%, time elapsed = {:.2f}, "
                      .format(epoch, args.density_epochs, i + 1, len(dl_train),
                              np.mean(loss_train_list), np.mean(acc_train_list), time.time() - tic), end="")

        with torch.no_grad():
            model.eval()
            acc_val_list = []
            for test_data in dl_test:
                X_test, y_test = test_data[0], test_data[1]
                X_test, y_test = X_test.to(torch.float32).to(device), y_test.to(torch.float32).to(device)
                pred_y_test = model(X_test)
                loss_test = loss_fun(pred_y_test, torch.squeeze(y_test).long())
                # Compute the accuracy
                forecasted_prob_test = softmax_fun(pred_y_test)
                _, forecasted_test = torch.max(forecasted_prob_test, 1)

                '''check for the estimated propensity score'''
                '''
                prop_temp = forecasted_prob_test.detach().cpu().numpy()
                y_temp = y_test.detach().cpu().numpy().astype(int)
                prop_score_test = np.array([prop_temp[i, y_temp[i, 0]]
                                            for i in range(y_temp.shape[0])]).reshape([y_temp.shape[0], -1])
                prop_score_test = np.concatenate([prop_score, prop_score_test], axis=1)
                '''
                correct = (forecasted_test == torch.squeeze(y_test).long()).sum().item()
                accuracy = 100 * correct / y_test.size(0)
                acc_val_list.append(accuracy)

            if verbose:
                print("Validation: Acc={:.4f}%".format(np.mean(acc_val_list)))

            if loss_test.item() < best_loss:
                best_loss = loss_test.item()
                # best_model = copy.deepcopy(model)
                best_model = model.state_dict()

        scheduler.step(loss_test)
    if best_model is not None:
        full_path = os.path.join("./results", "best_prop_score_trained_model.pth")
        torch.save(best_model, full_path)
        return full_path

