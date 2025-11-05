import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from functions.GumbelTauScheduler import GumbelTauScheduler
from model.TagNet import TagNet32, TagNet32_woLayernorm, TagNet_weights
from dataloader.data_loader import data_loader
import math
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_label_partition_log_data(label_partition_counts, domain_name, num_classes, num_partition, prefix):
    log_data = {}
    total_counts_per_label = label_partition_counts.sum(dim=1)

    for label_idx in range(num_classes):
        total_for_label = total_counts_per_label[label_idx].item()
        for part_idx in range(num_partition):
            count = label_partition_counts[label_idx, part_idx].item()

            if total_for_label > 0:
                percentage = (count / total_for_label) * 100
            else:
                percentage = 0.0

            log_key = f"Partition {domain_name} {prefix}/Partition:{part_idx}/Label:{label_idx}"
            log_data[log_key] = percentage
    return log_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_partition', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_domains', type=int, default=4)
    parser.add_argument('--pre_classifier_out', type=int, default=1024)
    parser.add_argument('--part_layer', type=int, default=1024)

    # tau scheduler
    parser.add_argument('--init_tau', type=float, default=2.0)
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--tau_decay', type=float, default=0.97)

    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)

    # parameter lr amplifier
    parser.add_argument('--prefc_lr', type=float, default=1.0)
    parser.add_argument('--fc_lr', type=float, default=1.0)
    parser.add_argument('--disc_lr', type=float, default=0.2)
    parser.add_argument('--switcher_lr', type=float, default=0.2)

    # regularization
    parser.add_argument('--reg_alpha', type=float, default=0.2)
    parser.add_argument('--reg_beta', type=float, default=1.0)
    parser.add_argument('--lambda_p', type=float, default=5e-2)

    args = parser.parse_args()
    init_lambda = args.lambda_p
    num_epochs = args.epoch

    wandb_run = wandb.init(entity="hails",
                           project="TagNet - NumObj dk",
                           config=args.__dict__,
                           name="[TagnetMLP]NumObj_UniqueDomain_lr:" + str(args.lr)
                                + "_Batch:" + str(args.batch_size)
                                + "_PLayer:" + str(args.part_layer)
                                + "_spe:" + str(args.reg_alpha)
                                + "_div:" + str(args.reg_beta)
                                + "_lr(d)" + str(args.disc_lr)
                                + "_lr(s)" + str(args.switcher_lr)
                                + "_lambda_p:" + str(args.lambda_p)
                           )

    mnist_loader, mnist_loader_test = data_loader('MNIST', args.batch_size)
    svhn_loader, svhn_loader_test = data_loader('SVHN', args.batch_size)
    cifar_loader, cifar_loader_test = data_loader('CIFAR10', args.batch_size)
    stl_loader, stl_loader_test = data_loader('STL10', args.batch_size)

    print("Data load complete, start training")

    model = TagNet32_woLayernorm(num_classes=args.num_classes,
                     pre_classifier_out=args.pre_classifier_out,
                     n_partition=args.num_partition,
                     part_layer=args.part_layer,
                     num_domains=args.num_domains,
                     device=device
                     )

    save_dir = f"./checkpoints/{wandb_run.name}"
    os.makedirs(save_dir, exist_ok=True)
    best_avg_acc = 0.0
    save_interval = 50
    min_save_epoch = 150

    optimizer = optim.Adam(TagNet_weights(
        model,
        lr=args.lr,
        pre_weight=args.prefc_lr,
        disc_weight=args.disc_lr,
        fc_weight=args.fc_lr,
        switcher_weight=args.switcher_lr
    ), lr=args.lr, weight_decay=args.opt_decay
    )

    tau_scheduler = GumbelTauScheduler(initial_tau=args.init_tau, min_tau=args.min_tau, decay_rate=args.tau_decay)
    domain_criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        phi = (1 + math.sqrt(5)) / 2
        lambda_p = init_lambda / phi ** (epoch / 50)
        tau = tau_scheduler.get_tau()

        total_mnist_domain_loss, total_mnist_domain_correct, total_mnist_loss, total_mnist_correct = 0, 0, 0, 0
        total_svhn_domain_loss, total_svhn_domain_correct, total_svhn_loss, total_svhn_correct = 0, 0, 0, 0
        total_cifar_domain_loss, total_cifar_domain_correct, total_cifar_loss, total_cifar_correct = 0, 0, 0, 0
        total_stl_domain_loss, total_stl_domain_correct, total_stl_loss, total_stl_correct = 0, 0, 0, 0
        total_domain_loss, total_label_loss, total_loss = 0, 0, 0
        total_specialization_loss, total_diversity_loss = 0, 0

        mnist_partition_counts = torch.zeros(args.num_partition, device=device)
        svhn_partition_counts = torch.zeros(args.num_partition, device=device)
        cifar_partition_counts = torch.zeros(args.num_partition, device=device)
        stl_partition_counts = torch.zeros(args.num_partition, device=device)

        total_samples_m, total_samples_s, total_samples_c, total_samples_stl = 0, 0, 0, 0

        mnist_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
        svhn_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
        cifar_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
        stl_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)

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

            # [TODO] each domain has different domain label
            mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
            svhn_dlabels = torch.full((svhn_images.size(0),), 1, dtype=torch.long, device=device)
            cifar_dlabels = torch.full((cifar_images.size(0),), 2, dtype=torch.long, device=device)
            stl_dlabels = torch.full((stl_images.size(0),), 3, dtype=torch.long, device=device)

            optimizer.zero_grad()

            bs_m, bs_s, bs_c, bs_stl = mnist_images.size(0), svhn_images.size(0), cifar_images.size(0), stl_images.size(
                0)
            all_images = torch.cat((mnist_images, svhn_images, cifar_images, stl_images), dim=0)

            out_part, domain_out, part_idx, part_gumbel = model(all_images, alpha=lambda_p, tau=tau, inference=False)

            mnist_out_part = out_part[:bs_m]
            svhn_out_part = out_part[bs_m: bs_m + bs_s]
            cifar_out_part = out_part[bs_m + bs_s: bs_m + bs_s + bs_c]
            stl_out_part = out_part[bs_m + bs_s + bs_c:]

            mnist_domain_out = domain_out[:bs_m]
            svhn_domain_out = domain_out[bs_m: bs_m + bs_s]
            cifar_domain_out = domain_out[bs_m + bs_s: bs_m + bs_s + bs_c]
            stl_domain_out = domain_out[bs_m + bs_s + bs_c:]

            mnist_part_idx = part_idx[:bs_m]
            svhn_part_idx = part_idx[bs_m: bs_m + bs_s]
            cifar_part_idx = part_idx[bs_m + bs_s: bs_m + bs_s + bs_c]
            stl_part_idx = part_idx[bs_m + bs_s + bs_c:]

            mnist_part_gumbel = part_gumbel[:bs_m]
            svhn_part_gumbel = part_gumbel[bs_m: bs_m + bs_s]
            cifar_part_gumbel = part_gumbel[bs_m + bs_s: bs_m + bs_s + bs_c]
            stl_part_gumbel = part_gumbel[bs_m + bs_s + bs_c:]

            if i % 1 == 0:
                print(f"--- [Epoch {epoch + 1}, Batch {i}] Partition Stats ---")
                mnist_counts = torch.bincount(mnist_part_idx, minlength=args.num_partition)
                svhn_counts = torch.bincount(svhn_part_idx, minlength=args.num_partition)
                cifar_counts = torch.bincount(cifar_part_idx, minlength=args.num_partition)
                stl_counts = torch.bincount(stl_part_idx, minlength=args.num_partition)
                print(
                    f"MNIST : {mnist_counts.cpu().numpy()} / SVHN  : {svhn_counts.cpu().numpy()} / CIFAR : {cifar_counts.cpu().numpy()} / STL : {stl_counts.cpu().numpy()}")
                print(
                    f"Switcher Weight Mean: {model.partition_switcher.weight.data.mean():.8f}, Bias Mean: {model.partition_switcher.bias.data.mean():.8f}")

            for l_idx in range(mnist_labels.size(0)):
                label_val = mnist_labels[l_idx].item()
                if 0 <= label_val < args.num_classes:
                    mnist_label_partition_counts[label_val, mnist_part_idx[l_idx].item()] += 1
            for l_idx in range(svhn_labels.size(0)):
                label_val = svhn_labels[l_idx].item()
                if 0 <= label_val < args.num_classes:
                    svhn_label_partition_counts[label_val, svhn_part_idx[l_idx].item()] += 1
            for l_idx in range(cifar_labels.size(0)):
                label_val = cifar_labels[l_idx].item()
                if 0 <= label_val < args.num_classes:
                    cifar_label_partition_counts[label_val, cifar_part_idx[l_idx].item()] += 1
            for l_idx in range(stl_labels.size(0)):
                label_val = stl_labels[l_idx].item()
                if 0 <= label_val < args.num_classes:
                    stl_label_partition_counts[label_val, stl_part_idx[l_idx].item()] += 1

            mnist_label_loss = criterion(mnist_out_part, mnist_labels)
            svhn_label_loss = criterion(svhn_out_part, svhn_labels)
            cifar_label_loss = criterion(cifar_out_part, cifar_labels)
            stl_label_loss = criterion(stl_out_part, stl_labels)

            numbers_part_gumbel = torch.cat((mnist_part_gumbel, svhn_part_gumbel))
            objects_part_gumbel = torch.cat((cifar_part_gumbel, stl_part_gumbel))

            avg_prob_numbers = torch.mean(numbers_part_gumbel, dim=0)
            avg_prob_objects = torch.mean(objects_part_gumbel, dim=0)

            loss_specialization_numbers = -torch.sum(avg_prob_numbers * torch.log(avg_prob_numbers + 1e-8))
            loss_specialization_objects = -torch.sum(avg_prob_objects * torch.log(avg_prob_objects + 1e-8))
            loss_specialization = loss_specialization_numbers + loss_specialization_objects
            all_probs = torch.cat((numbers_part_gumbel, objects_part_gumbel), dim=0)
            avg_prob_global = torch.mean(all_probs, dim=0)

            loss_diversity = torch.sum(avg_prob_global * torch.log(avg_prob_global + 1e-8))

            label_loss = ((mnist_label_loss + svhn_label_loss) / 2 + (cifar_label_loss + stl_label_loss) / 2
                          + args.reg_alpha * loss_specialization + args.reg_beta * loss_diversity)
            mnist_domain_loss = domain_criterion(mnist_domain_out, mnist_dlabels)
            svhn_domain_loss = domain_criterion(svhn_domain_out, svhn_dlabels)
            cifar_domain_loss = domain_criterion(cifar_domain_out, cifar_dlabels)
            stl_domain_loss = domain_criterion(stl_domain_out, stl_dlabels)

            domain_loss = (mnist_domain_loss + svhn_domain_loss) / 2 + (cifar_domain_loss + stl_domain_loss) / 2

            loss = label_loss + domain_loss
            loss.backward()

            entries = []
            for name, param in model.partition_switcher.named_parameters():
                if param.grad is None:
                    grad_str = 'None'
                else:
                    grad_str = f"{torch.mean(torch.abs(param.grad)).item():.6f}"
                data_str = f"{torch.mean(torch.abs(param.data)).item():.6f}"
                entries.append(f"{name}: {grad_str}, {data_str}")

            print(" | ".join(entries) + f" | loss: {loss.item():.6f} ")

            optimizer.step()

            mnist_partition_counts += torch.bincount(mnist_part_idx, minlength=args.num_partition).to(device)
            svhn_partition_counts += torch.bincount(svhn_part_idx, minlength=args.num_partition).to(device)
            cifar_partition_counts += torch.bincount(cifar_part_idx, minlength=args.num_partition).to(device)
            stl_partition_counts += torch.bincount(stl_part_idx, minlength=args.num_partition).to(device)

            total_label_loss += label_loss.item() * (bs_m + bs_s + bs_c + bs_stl)  # °¡Áß Æò±ÕÀ» À§ÇØ ¹èÄ¡ Å©±â °öÇÔ
            total_mnist_loss += mnist_label_loss.item() * bs_m
            total_svhn_loss += svhn_label_loss.item() * bs_s
            total_cifar_loss += cifar_label_loss.item() * bs_c
            total_stl_loss += stl_label_loss.item() * bs_stl

            total_domain_loss += domain_loss.item() * (bs_m + bs_s + bs_c + bs_stl)
            total_mnist_domain_loss += mnist_domain_loss.item() * bs_m
            total_svhn_domain_loss += svhn_domain_loss.item() * bs_s
            total_cifar_domain_loss += cifar_domain_loss.item() * bs_c
            total_stl_domain_loss += stl_domain_loss.item() * bs_stl

            total_specialization_loss += loss_specialization.item() * (bs_m + bs_s + bs_c + bs_stl)
            total_diversity_loss += loss_diversity.item() * (bs_m + bs_s + bs_c + bs_stl)
            total_loss += loss.item() * (bs_m + bs_s + bs_c + bs_stl)

            total_mnist_correct += (torch.argmax(mnist_out_part, dim=1) == mnist_labels).sum().item()
            total_svhn_correct += (torch.argmax(svhn_out_part, dim=1) == svhn_labels).sum().item()
            total_cifar_correct += ((torch.argmax(cifar_out_part, dim=1) == cifar_labels).sum().item())
            total_stl_correct += ((torch.argmax(stl_out_part, dim=1) == stl_labels).sum().item())

            total_mnist_domain_correct += (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
            total_svhn_domain_correct += (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()
            total_cifar_domain_correct += ((torch.argmax(cifar_domain_out, dim=1) == cifar_dlabels).sum().item())
            total_stl_domain_correct += ((torch.argmax(stl_domain_out, dim=1) == stl_dlabels).sum().item())

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

        mnist_train_partition_log = get_label_partition_log_data(
            mnist_label_partition_counts, 'MNIST', args.num_classes, args.num_partition, prefix="Train"
        )
        svhn_train_partition_log = get_label_partition_log_data(
            svhn_label_partition_counts, 'SVHN', args.num_classes, args.num_partition, prefix="Train"
        )
        cifar_train_partition_log = get_label_partition_log_data(
            cifar_label_partition_counts, 'CIFAR', args.num_classes, args.num_partition, prefix="Train"
        )
        stl_train_partition_log = get_label_partition_log_data(
            stl_label_partition_counts, 'STL', args.num_classes, args.num_partition, prefix="Train"
        )

        mnist_partition_ratios = mnist_partition_counts / total_samples_m * 100
        svhn_partition_ratios = svhn_partition_counts / total_samples_s * 100
        cifar_partition_ratios = cifar_partition_counts / total_samples_c * 100
        stl_partition_ratios = stl_partition_counts / total_samples_stl * 100

        mnist_partition_ratio_str = " | ".join(
            [f"Partition {p}: {mnist_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
        svhn_partition_ratio_str = " | ".join(
            [f"Partition {p}: {svhn_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
        cifar_partition_ratio_str = " | ".join(
            [f"Partition {p}: {cifar_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
        stl_partition_ratio_str = " | ".join(
            [f"P {p}: {stl_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])

        mnist_domain_avg_loss = total_mnist_domain_loss / total_samples_m
        svhn_domain_avg_loss = total_svhn_domain_loss / total_samples_s
        cifar_domain_avg_loss = total_cifar_domain_loss / total_samples_c
        stl_domain_avg_loss = total_stl_domain_loss / total_samples_stl

        mnist_avg_loss = total_mnist_loss / total_samples_m
        svhn_avg_loss = total_svhn_loss / total_samples_s
        cifar_avg_loss = total_cifar_loss / total_samples_c
        stl_avg_loss = total_stl_loss / total_samples_stl

        domain_avg_loss = total_domain_loss / total_samples_all
        label_avg_loss = total_label_loss / total_samples_all
        specialization_loss = total_specialization_loss / total_samples_all
        diversity_loss = total_diversity_loss / total_samples_all
        total_avg_loss = total_loss / total_samples_all

        mnist_acc_epoch = total_mnist_correct / total_samples_m * 100
        svhn_acc_epoch = total_svhn_correct / total_samples_s * 100
        cifar_acc_epoch = total_cifar_correct / total_samples_c * 100
        stl_acc_epoch = total_stl_correct / total_samples_stl * 100

        mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples_m * 100
        svhn_domain_acc_epoch = total_svhn_domain_correct / total_samples_s * 100
        cifar_domain_acc_epoch = total_cifar_domain_correct / total_samples_c * 100
        stl_domain_acc_epoch = total_stl_domain_correct / total_samples_stl * 100

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(
            f'  [Ratios] MNIST: [{mnist_partition_ratio_str}] | SVHN: [{svhn_partition_ratio_str}] | CIFAR: [{cifar_partition_ratio_str}] | STL: [{stl_partition_ratio_str}]')
        print(
            f'  [Acc]    MNIST: {mnist_acc_epoch:<6.2f}% | SVHN: {svhn_acc_epoch:<6.2f}% | CIFAR: {cifar_acc_epoch:<6.2f}% | STL: {stl_acc_epoch:<6.2f}%')
        print(
            f'  [DomAcc] MNIST: {mnist_domain_acc_epoch:<6.2f}% | SVHN: {svhn_domain_acc_epoch:<6.2f}% | CIFAR: {cifar_domain_acc_epoch:<6.2f}% | STL: {stl_domain_acc_epoch:<6.2f}%')
        print(f'  [Reg]    Spec:  {specialization_loss:<8.4f} | Div:    {diversity_loss:<8.4f} | Tau: {tau:<5.3f}')
        print(
            f'  [Loss]   Label: {label_avg_loss:<8.4f} | Domain: {domain_avg_loss:<8.4f} | Total: {total_avg_loss:<8.4f}')
        print(
            f'  [Label]  MNIST: {mnist_avg_loss:<6.4f} | SVHN: {svhn_avg_loss:<6.4f} | CIFAR: {cifar_avg_loss:<6.4f} | STL: {stl_avg_loss:<6.4f}')
        print(
            f'  [Domain] MNIST: {mnist_domain_avg_loss:<6.4f} | SVHN: {svhn_domain_avg_loss:<6.4f} | CIFAR: {cifar_domain_avg_loss:<6.4f} | STL: {stl_domain_avg_loss:<6.4f}')

        wandb.log({
            **{f"Train/Partition {p} MNIST Ratio": mnist_partition_ratios[p].item() for p in range(args.num_partition)},
            **{f"Train/Partition {p} SVHN Ratio": svhn_partition_ratios[p].item() for p in range(args.num_partition)},
            **{f"Train/Partition {p} CIFAR Ratio": cifar_partition_ratios[p].item() for p in range(args.num_partition)},
            **{f"Train/Partition {p} STL Ratio": stl_partition_ratios[p].item() for p in range(args.num_partition)},
            'Train/Label MNIST Accuracy': mnist_acc_epoch,
            'Train/Label SVHN Accuracy': svhn_acc_epoch,
            'Train/Label CIFAR Accuracy': cifar_acc_epoch,
            'Train/Label STL Accuracy': stl_acc_epoch,
            'Train/Domain MNIST Accuracy': mnist_domain_acc_epoch,
            'Train/Domain SVHN Accuracy': svhn_domain_acc_epoch,
            'Train/Domain CIFAR Accuracy': cifar_domain_acc_epoch,
            'Train/Domain STL Accuracy': stl_domain_acc_epoch,
            'TrainLoss/Label MNIST Loss': mnist_avg_loss,
            'TrainLoss/Label SVHN Loss': svhn_avg_loss,
            'TrainLoss/Label CIFAR Loss': cifar_avg_loss,
            'TrainLoss/Label STL Loss': stl_avg_loss,
            'TrainLoss/Label Loss': label_avg_loss,
            'TrainLoss/Domain MNIST Loss': mnist_domain_avg_loss,
            'TrainLoss/Domain SVHN Loss': svhn_domain_avg_loss,
            'TrainLoss/Domain CIFAR Loss': cifar_domain_avg_loss,
            'TrainLoss/Domain STL Loss': stl_domain_avg_loss,
            'TrainLoss/Domain Loss': domain_avg_loss,
            'TrainLoss/Specialization Loss': specialization_loss,
            'TrainLoss/Diversity Loss': diversity_loss,
            'TrainLoss/Total Loss': total_avg_loss,
            'Parameters/Tau': tau,
            'Parameters/Learning Rate': optimizer.param_groups[0]['lr'],
            'Parameters/Lambda_p': lambda_p,
            **mnist_train_partition_log,
            **svhn_train_partition_log,
            **cifar_train_partition_log,
            **stl_train_partition_log,
        }, step=epoch + 1)

        model.eval()

        with ((torch.no_grad())):

            def evaluate_dataset(loader, d_label):
                total_loss, total_correct, total_samples = 0, 0, 0
                total_domain_loss, total_domain_correct = 0, 0

                partition_counts = torch.zeros(args.num_partition, device=device)
                label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)

                gumbel_outputs_list = []

                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    dlabels = torch.full_like(labels, fill_value=d_label, device=device)
                    bs = images.size(0)

                    out_part, domain_out, part_idx, part_gumbel = model(images, alpha=0, tau=tau, inference=False)

                    gumbel_outputs_list.append(part_gumbel)

                    label_loss = criterion(out_part, labels)
                    domain_loss = domain_criterion(domain_out, dlabels)

                    total_loss += (label_loss.item() * bs)
                    total_domain_loss += (domain_loss.item() * bs)

                    total_correct += (torch.argmax(out_part, dim=1) == labels).sum().item()
                    total_domain_correct += (torch.argmax(domain_out, dim=1) == dlabels).sum().item()
                    total_samples += bs

                    partition_counts += torch.bincount(part_idx, minlength=args.num_partition)

                    for l_idx in range(bs):
                        label_val = labels[l_idx].item()
                        if 0 <= label_val < args.num_classes:
                            label_partition_counts[label_val, part_idx[l_idx].item()] += 1

                if total_samples == 0: total_samples = 1

                avg_loss = total_loss / total_samples
                avg_domain_loss = total_domain_loss / total_samples
                acc = (total_correct / total_samples) * 100
                domain_acc = (total_domain_correct / total_samples) * 100
                ratios = partition_counts / total_samples * 100

                all_gumbel_outputs = torch.cat(gumbel_outputs_list, dim=0)

                return avg_loss, avg_domain_loss, acc, domain_acc, ratios, label_partition_counts, all_gumbel_outputs

            test_mnist_avg_loss, test_mnist_dom_avg_loss, test_mnist_acc_epoch, test_mnist_domain_acc_epoch, test_mnist_ratios, test_mnist_label_counts, mnist_gumbel = evaluate_dataset(
                mnist_loader_test, 0)
            test_svhn_avg_loss, test_svhn_dom_avg_loss, test_svhn_acc_epoch, test_svhn_domain_acc_epoch, test_svhn_ratios, test_svhn_label_counts, svhn_gumbel = evaluate_dataset(
                svhn_loader_test, 0)
            test_cifar_avg_loss, test_cifar_dom_avg_loss, test_cifar_acc_epoch, test_cifar_domain_acc_epoch, test_cifar_ratios, test_cifar_label_counts, cifar_gumbel = evaluate_dataset(
                cifar_loader_test, 1)
            test_stl_avg_loss, test_stl_dom_avg_loss, test_stl_acc_epoch, test_stl_domain_acc_epoch, test_stl_ratios, test_stl_label_counts, stl_gumbel = evaluate_dataset(
                stl_loader_test, 1)

            current_avg_acc = (
                                          test_mnist_acc_epoch + test_svhn_acc_epoch + test_cifar_acc_epoch + test_stl_acc_epoch) / 4.0
            test_label_avg_loss = (
                                              test_mnist_avg_loss + test_svhn_avg_loss + test_cifar_avg_loss + test_stl_avg_loss) / 4.0
            test_domain_avg_loss = (
                                               test_mnist_dom_avg_loss + test_svhn_dom_avg_loss + test_cifar_dom_avg_loss + test_stl_dom_avg_loss) / 4.0

            numbers_part_gumbel = torch.cat((mnist_gumbel, svhn_gumbel), dim=0)
            objects_part_gumbel = torch.cat((cifar_gumbel, stl_gumbel), dim=0)

            avg_prob_numbers = torch.mean(numbers_part_gumbel, dim=0)
            avg_prob_objects = torch.mean(objects_part_gumbel, dim=0)

            loss_specialization_numbers = -torch.sum(avg_prob_numbers * torch.log(avg_prob_numbers + 1e-8))
            loss_specialization_objects = -torch.sum(avg_prob_objects * torch.log(avg_prob_objects + 1e-8))
            test_specialization_loss = (loss_specialization_numbers + loss_specialization_objects).item()

            all_probs = torch.cat((numbers_part_gumbel, objects_part_gumbel), dim=0)
            avg_prob_global = torch.mean(all_probs, dim=0)
            test_diversity_loss = torch.sum(avg_prob_global * torch.log(avg_prob_global + 1e-8)).item()

            test_total_avg_loss = test_label_avg_loss + test_domain_avg_loss + (
                        args.reg_alpha * test_specialization_loss) + (args.reg_beta * test_diversity_loss)

            test_mnist_train_partition_log = get_label_partition_log_data(
                test_mnist_label_counts, 'MNIST', args.num_classes, args.num_partition, prefix="Test"
            )
            test_svhn_train_partition_log = get_label_partition_log_data(
                test_svhn_label_counts, 'SVHN', args.num_classes, args.num_partition, prefix="Test"
            )
            test_cifar_train_partition_log = get_label_partition_log_data(
                test_cifar_label_counts, 'CIFAR', args.num_classes, args.num_partition, prefix="Test"
            )
            test_stl_train_partition_log = get_label_partition_log_data(
                test_stl_label_counts, 'STL', args.num_classes, args.num_partition, prefix="Test"
            )

            test_mnist_partition_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(test_mnist_ratios)])
            test_svhn_partition_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(test_svhn_ratios)])
            test_cifar_partition_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(test_cifar_ratios)])
            test_stl_partition_ratio_str = " | ".join([f"P{p}: {r:.2f}%" for p, r in enumerate(test_stl_ratios)])

            if (epoch + 1) >= min_save_epoch and current_avg_acc > best_avg_acc:
                best_avg_acc = current_avg_acc
                save_path = os.path.join(save_dir, "best_model.pt")
                torch.save(model.state_dict(), save_path)
                print(
                    f"*** New best model saved to {save_path} (Epoch: {epoch + 1}, Avg Acc: {current_avg_acc:.2f}%) ***")
                wandb.log({"Test/Best Avg Accuracy": best_avg_acc}, step=epoch + 1)

            if (epoch + 1) % save_interval == 0:
                periodic_save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save(model.state_dict(), periodic_save_path)
                print(f"--- Periodic checkpoint saved to {periodic_save_path} ---")

            print(f'Epoch [{epoch + 1}/{num_epochs}] (Test)')
            print(
                f'  [Ratios] MNIST: [{test_mnist_partition_ratio_str}] | SVHN: [{test_svhn_partition_ratio_str}] | CIFAR: [{test_cifar_partition_ratio_str}] | STL: [{test_stl_partition_ratio_str}]')
            print(
                f'  [Acc]    MNIST: {test_mnist_acc_epoch:<6.2f}% | SVHN: {test_svhn_acc_epoch:<6.2f}% | CIFAR: {test_cifar_acc_epoch:<6.2f}% | STL: {test_stl_acc_epoch:<6.2f}%')
            print(
                f'  [DomAcc] MNIST: {test_mnist_domain_acc_epoch:<6.2f}% | SVHN: {test_svhn_domain_acc_epoch:<6.2f}% | CIFAR: {test_cifar_domain_acc_epoch:<6.2f}% | STL: {test_stl_domain_acc_epoch:<6.2f}%')
            print(
                f'  [Reg]    Spec:  {test_specialization_loss:<8.4f} | Div:    {test_diversity_loss:<8.4f} | Tau: {tau:<5.3f}')
            print(
                f'  [Loss]   Label: {test_label_avg_loss:<8.4f} | Domain: {test_domain_avg_loss:<8.4f} | Total: {test_total_avg_loss:<8.4f}')
            print(
                f'  [Label]  MNIST: {test_mnist_avg_loss:<6.4f} | SVHN: {test_svhn_avg_loss:<6.4f} | CIFAR: {test_cifar_avg_loss:<6.4f} | STL: {test_stl_avg_loss:<6.4f}')
            print(
                f'  [Domain] MNIST: {test_mnist_dom_avg_loss:<6.4f} | SVHN: {test_svhn_dom_avg_loss:<6.4f} | CIFAR: {test_cifar_dom_avg_loss:<6.4f} | STL: {test_stl_dom_avg_loss:<6.4f}')

            wandb.log({
                **{f"Test/Partition {p} MNIST Ratio": r.item() for p, r in enumerate(test_mnist_ratios)},
                **{f"Test/Partition {p} SVHN Ratio": r.item() for p, r in enumerate(test_svhn_ratios)},
                **{f"Test/Partition {p} CIFAR Ratio": r.item() for p, r in enumerate(test_cifar_ratios)},
                **{f"Test/Partition {p} STL Ratio": r.item() for p, r in enumerate(test_stl_ratios)},
                'Test/Label MNIST Accuracy': test_mnist_acc_epoch,
                'Test/Label SVHN Accuracy': test_svhn_acc_epoch,
                'Test/Label CIFAR Accuracy': test_cifar_acc_epoch,
                'Test/Label STL Accuracy': test_stl_acc_epoch,
                'Test/Domain MNIST Accuracy': test_mnist_domain_acc_epoch,
                'Test/Domain SVHN Accuracy': test_svhn_domain_acc_epoch,
                'Test/Domain CIFAR Accuracy': test_cifar_domain_acc_epoch,
                'Test/Domain STL Accuracy': test_stl_domain_acc_epoch,
                'TestLoss/Label MNIST Loss': test_mnist_avg_loss,
                'TestLoss/Label SVHN Loss': test_svhn_avg_loss,
                'TestLoss/Label CIFAR Loss': test_cifar_avg_loss,
                'TestLoss/Label STL Loss': test_stl_avg_loss,
                'TestLoss/Label Loss': test_label_avg_loss,
                'TestLoss/Domain MNIST Loss': test_mnist_dom_avg_loss,
                'TestLoss/Domain SVHN Loss': test_svhn_dom_avg_loss,
                'TestLoss/Domain CIFAR Loss': test_cifar_dom_avg_loss,
                'TestLoss/Domain STL Loss': test_stl_dom_avg_loss,
                'TestLoss/Domain Loss': test_domain_avg_loss,
                'TestLoss/Specialization Loss': test_specialization_loss,
                'TestLoss/Diversity Loss': test_diversity_loss,
                'TestLoss/Total Loss': test_total_avg_loss,
                **test_mnist_train_partition_log,
                **test_svhn_train_partition_log,
                **test_cifar_train_partition_log,
                **test_stl_train_partition_log,
            }, step=epoch + 1)

        tau_scheduler.step()

    final_save_path = os.path.join(save_dir, f"final_model_epoch_{num_epochs}.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"--- Final model saved to {final_save_path} ---")


if __name__ == '__main__':
    main()