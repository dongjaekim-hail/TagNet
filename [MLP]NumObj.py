import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.data_loader import data_loader
import wandb
import os

class MLP(nn.Module):
    def __init__(self, num_classes=10, hidden_size=1024):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3 * 32 * 32, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=128)

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    args = parser.parse_args()
    num_epochs = args.epoch

    wandb_run = wandb.init(entity="hails",
                           project="TagNet - NumObj dk",
                           config=args.__dict__,
                           name="[MLP]NumObj_lr:" + str(args.lr)
                                + "_Batch:" + str(args.batch_size)
                                + "_Hidden:" + str(args.hidden_size)
                           )

    save_dir = f"./checkpoints/{wandb_run.name}"
    os.makedirs(save_dir, exist_ok=True)
    best_avg_acc = 0.0
    save_interval = 100
    min_save_epoch = 10

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    cifar_loader, cifar_loader_test = data_loader('CIFAR10', args.batch_size)
    stl_loader, stl_loader_test = data_loader('STL10', args.batch_size)

    print("Data load complete, start training")

    model = MLP(num_classes=args.num_classes, hidden_size=args.hidden_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.opt_decay)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()

        total_mnist_loss, total_mnist_correct = 0, 0
        total_svhn_loss, total_svhn_correct = 0, 0
        total_cifar_loss, total_cifar_correct = 0, 0
        total_stl_loss, total_stl_correct = 0, 0
        total_label_loss, total_loss = 0, 0

        total_samples_m, total_samples_s, total_samples_c, total_samples_stl = 0, 0, 0, 0

        train_loader_zip = zip(mnist_loader, svhn_loader, cifar_loader, stl_loader)

        for i, (mnist_data, svhn_data, cifar_data, stl_data) in enumerate(train_loader_zip):
            mnist_images, mnist_labels = mnist_data
            mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
            svhn_images, svhn_labels = svhn_data
            svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
            cifar_images, cifar_labels = cifar_data
            cifar_images, cifar_labels = cifar_images.to(device), cifar_labels.to(device)
            stl_images, stl_labels = stl_data
            stl_images, stl_labels = stl_images.to(device), stl_labels.to(device)

            optimizer.zero_grad()

            bs_m, bs_s, bs_c, bs_stl = mnist_images.size(0), svhn_images.size(0), cifar_images.size(0), stl_images.size(0)
            all_images = torch.cat((mnist_images, svhn_images, cifar_images, stl_images), dim=0)

            all_outputs = model(all_images)

            mnist_out_part = all_outputs[:bs_m]
            svhn_out_part = all_outputs[bs_m: bs_m + bs_s]
            cifar_out_part = all_outputs[bs_m + bs_s: bs_m + bs_s + bs_c]
            stl_out_part = all_outputs[bs_m + bs_s + bs_c:]

            mnist_label_loss = criterion(mnist_out_part, mnist_labels)
            svhn_label_loss = criterion(svhn_out_part, svhn_labels)
            cifar_label_loss = criterion(cifar_out_part, cifar_labels)
            stl_label_loss = criterion(stl_out_part, stl_labels)

            label_loss = (mnist_label_loss + svhn_label_loss) / 2 + (cifar_label_loss + stl_label_loss) / 2
            loss = label_loss

            loss.backward()
            optimizer.step()

            total_label_loss += label_loss.item()
            total_mnist_loss += mnist_label_loss.item()
            total_svhn_loss += svhn_label_loss.item()
            total_cifar_loss += cifar_label_loss.item()
            total_stl_loss += stl_label_loss.item()
            total_loss += loss.item()

            total_mnist_correct += (torch.argmax(mnist_out_part, dim=1) == mnist_labels).sum().item()
            total_svhn_correct += (torch.argmax(svhn_out_part, dim=1) == svhn_labels).sum().item()
            total_cifar_correct += ((torch.argmax(cifar_out_part, dim=1) == cifar_labels).sum().item())
            total_stl_correct += ((torch.argmax(stl_out_part, dim=1) == stl_labels).sum().item())

            total_samples_m += bs_m
            total_samples_s += bs_s
            total_samples_c += bs_c
            total_samples_stl += bs_stl

        if total_samples_m == 0: 
            total_samples_m = 1
        if total_samples_s == 0: 
            total_samples_s = 1
        if total_samples_c == 0: 
            total_samples_c = 1
        if total_samples_stl == 0:
            total_samples_stl = 1

        total_samples_all = total_samples_m + total_samples_s + total_samples_c + total_samples_stl

        # TODO just so wrong. Not every data go all the way to the end of dataloader.
        # mnist_avg_loss = total_mnist_loss / len(mnist_loader)
        # svhn_avg_loss = total_svhn_loss / len(svhn_loader)
        # cifar_avg_loss = total_cifar_loss / len(cifar_loader)
        # stl_avg_loss = total_stl_loss / len(stl_loader)
        
        mnist_avg_loss = total_mnist_loss / total_samples_m
        svhn_avg_loss = total_svhn_loss / total_samples_s
        cifar_avg_loss = total_cifar_loss / total_samples_c
        stl_avg_loss = total_stl_loss / total_samples_stl

        # label_avg_loss = total_label_loss / (len(mnist_loader) * 4)
        # total_avg_loss = total_loss / (len(mnist_loader) * 4)
        label_avg_loss = total_label_loss / total_samples_all
        total_avg_loss = total_loss / total_samples_all

        mnist_acc_epoch = total_mnist_correct / total_samples_m * 100
        svhn_acc_epoch = total_svhn_correct / total_samples_s * 100
        cifar_acc_epoch = total_cifar_correct / total_samples_c * 100
        stl_acc_epoch = total_stl_correct / total_samples_stl * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  [Acc]    MNIST: {mnist_acc_epoch:<6.2f}% | SVHN: {svhn_acc_epoch:<6.2f}% | CIFAR: {cifar_acc_epoch:<6.2f}% | STL: {stl_acc_epoch:<6.2f}%')
        print(f'  [Loss]   Label: {label_avg_loss:<8.4f} | Total: {total_avg_loss:<8.4f}')
        print(f'  [Label]  MNIST: {mnist_avg_loss:<6.4f} | SVHN: {svhn_avg_loss:<6.4f} | CIFAR: {cifar_avg_loss:<6.4f} | STL: {stl_avg_loss:<6.4f}')

        wandb.log({
            'Train/Label MNIST Accuracy': mnist_acc_epoch,
            'Train/Label SVHN Accuracy': svhn_acc_epoch,
            'Train/Label CIFAR Accuracy': cifar_acc_epoch,
            'Train/Label STL Accuracy': stl_acc_epoch,
            'TrainLoss/Label MNIST Loss': mnist_avg_loss,
            'TrainLoss/Label SVHN Loss': svhn_avg_loss,
            'TrainLoss/Label CIFAR Loss': cifar_avg_loss,
            'TrainLoss/Label STL Loss': stl_avg_loss,
            'TrainLoss/Label Loss': label_avg_loss,
            'TrainLoss/Total Loss': total_avg_loss,
            'Parameters/Learning Rate': optimizer.param_groups[0]['lr'],
        }, step=epoch + 1)

        model.eval()

        with ((torch.no_grad())):

            def evaluate_cnn_dataset(loader):
                total_loss, total_correct, total_samples = 0, 0, 0
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    bs = images.size(0)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    total_loss += (loss.item() * bs)
                    total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                    total_samples += bs

                if total_samples == 0: total_samples = 1
                avg_loss = total_loss / total_samples
                acc = (total_correct / total_samples) * 100
                return avg_loss, acc

            test_mnist_avg_loss, test_mnist_acc_epoch = evaluate_cnn_dataset(mnist_loader_test)
            test_svhn_avg_loss, test_svhn_acc_epoch = evaluate_cnn_dataset(svhn_loader_test)
            test_cifar_avg_loss, test_cifar_acc_epoch = evaluate_cnn_dataset(cifar_loader_test)
            test_stl_avg_loss, test_stl_acc_epoch = evaluate_cnn_dataset(stl_loader_test)

            test_label_avg_loss = (test_mnist_avg_loss + test_svhn_avg_loss + test_cifar_avg_loss + test_stl_avg_loss) / 4.0
            test_total_avg_loss = test_label_avg_loss
            current_avg_acc = (test_mnist_acc_epoch + test_svhn_acc_epoch + test_cifar_acc_epoch + test_stl_acc_epoch) / 4.0

            print(f'Epoch [{epoch + 1}/{num_epochs}] (Test)')
            print(f'  [Acc]    MNIST: {test_mnist_acc_epoch:<6.2f}% | SVHN: {test_svhn_acc_epoch:<6.2f}% | CIFAR: {test_cifar_acc_epoch:<6.2f}% | STL: {test_stl_acc_epoch:<6.2f}%')
            print(f'  [Loss]   Label: {test_label_avg_loss:<8.4f} | Total: {test_total_avg_loss:<8.4f}')
            print(f'  [Label]  MNIST: {test_mnist_avg_loss:<6.4f} | SVHN: {test_svhn_avg_loss:<6.4f} | CIFAR: {test_cifar_avg_loss:<6.4f} | STL: {test_stl_avg_loss:<6.4f}')

            if (epoch + 1) >= min_save_epoch and current_avg_acc > best_avg_acc:
                best_avg_acc = current_avg_acc
                save_path = os.path.join(save_dir, "best_model.pt")
                torch.save(model.state_dict(), save_path)
                print(f"*** New best model saved to {save_path} (Epoch: {epoch + 1}, Avg Acc: {current_avg_acc:.2f}%) ***")
                wandb.log({"Test/Best Avg Accuracy": best_avg_acc}, step=epoch + 1)

            if (epoch + 1) % save_interval == 0:
                periodic_save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save(model.state_dict(), periodic_save_path)
                print(f"--- Periodic checkpoint saved to {periodic_save_path} ---")

            wandb.log({
                'Test/Label MNIST Accuracy': test_mnist_acc_epoch,
                'Test/Label SVHN Accuracy': test_svhn_acc_epoch,
                'Test/Label CIFAR Accuracy': test_cifar_acc_epoch,
                'Test/Label STL Accuracy': test_stl_acc_epoch,
                'TestLoss/Label MNIST Loss': test_mnist_avg_loss,
                'TestLoss/Label SVHN Loss': test_svhn_avg_loss,
                'TestLoss/Label CIFAR Loss': test_cifar_avg_loss,
                'TestLoss/Label STL Loss': test_stl_avg_loss,
                'TestLoss/Label Loss': test_label_avg_loss,
                'TestLoss/Total Loss': test_total_avg_loss,
            }, step=epoch + 1)

    final_save_path = os.path.join(save_dir, f"final_model_epoch_{num_epochs}.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"--- Final model saved to {final_save_path} ---")


if __name__ == '__main__':
    main()