# sweep for learning rate scheduling as well
# wandb agent 'hails/TagNet MSC sweep everything schedule 16 fixedlambdap/gx75rzjf'
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from functions.GumbelTauScheduler import GumbelTauScheduler
from model.TagNet import TagNet32, TagNet32_woLayernorm, TagNet_weights, TagNet_ClassDomain_weights
from dataloader.data_loader import data_loader
from torch.optim.lr_scheduler import LambdaLR
import math
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--num_partition', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_domains', type=int, default=4)
parser.add_argument('--hidden_size', type=int, default=16)
parser.add_argument('--pre_classifier_out', type=int, default=16)
parser.add_argument('--part_layer', type=int, default=16)

# tau scheduler
parser.add_argument('--init_tau', type=float, default=1)
parser.add_argument('--min_tau', type=float, default=1)
parser.add_argument('--tau_decay', type=float, default=0.97)

# Optimizer
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--opt_decay', type=float, default=1e-6)

# Lr scheduler
parser.add_argument('--lr_domain', type=float, default=1e-3)
parser.add_argument('--lr_domain_min', type=float, default=1e-6)
parser.add_argument('--lr_domain_prop', type=float, default=0.4)

# parameter lr amplifier
parser.add_argument('--prefc_lr', type=float, default=1.0)
parser.add_argument('--fc_lr', type=float, default=1.0)
parser.add_argument('--disc_lr', type=float, default=1.0)
parser.add_argument('--switcher_lr', type=float, default=0.05)

# regularization
parser.add_argument('--reg_alpha', type=float, default=0.01)
parser.add_argument('--reg_beta', type=float, default=0.01)
parser.add_argument('--lambda_p', type=float, default=0.1)

args = parser.parse_args()
args.pre_classifier_out = args.hidden_size
args.part_layer = args.hidden_size

num_epochs = args.epoch

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

def train_step(epoch, model, args, optimizers, criterion, domain_criterion, data_loader_zip, phi, lambda_p, tau, inference = False):

    if not inference:
        model.train()
    else:
        model.eval()

    total_mnist_domain_loss, total_mnist_domain_correct, total_mnist_loss, total_mnist_correct = 0, 0, 0, 0
    total_svhn_domain_loss, total_svhn_domain_correct, total_svhn_loss, total_svhn_correct = 0, 0, 0, 0
    total_cifar_domain_loss, total_cifar_domain_correct, total_cifar_loss, total_cifar_correct = 0, 0, 0, 0
    total_domain_loss, total_label_loss, total_loss = 0, 0, 0
    total_specialization_loss, total_diversity_loss = 0, 0

    mnist_partition_counts = torch.zeros(args.num_partition, device=device)
    svhn_partition_counts = torch.zeros(args.num_partition, device=device)
    cifar_partition_counts = torch.zeros(args.num_partition, device=device)

    total_samples_m, total_samples_s, total_samples_c  = 0, 0, 0 

    mnist_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
    svhn_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
    cifar_label_partition_counts = torch.zeros(args.num_classes, args.num_partition, device=device)
    
    for i, (mnist_data, svhn_data, cifar_data) in enumerate(data_loader_zip):
        
        
        mnist_images, mnist_labels = mnist_data
        mnist_images, mnist_labels = mnist_images.to(device), mnist_labels.to(device)
        svhn_images, svhn_labels = svhn_data
        svhn_images, svhn_labels = svhn_images.to(device), svhn_labels.to(device)
        cifar_images, cifar_labels = cifar_data
        cifar_images, cifar_labels = cifar_images.to(device), cifar_labels.to(device)

        # [TODO] each domain has different domain label
        mnist_dlabels = torch.full((mnist_images.size(0),), 0, dtype=torch.long, device=device)
        svhn_dlabels = torch.full((svhn_images.size(0),), 1, dtype=torch.long, device=device)
        cifar_dlabels = torch.full((cifar_images.size(0),), 2, dtype=torch.long, device=device)

        if not inference:
            optimizers[0].zero_grad()
            optimizers[1].zero_grad()

        bs_m, bs_s, bs_c= mnist_images.size(0), svhn_images.size(0), cifar_images.size(0)
        all_images = torch.cat((mnist_images, svhn_images, cifar_images), dim=0)

        # TODO here, I think part_gumbel is meaningless. it must be the probability before gumbel sampling
        out_part, domain_out, part_idx, part_gumbel = model(all_images, alpha=lambda_p, tau=tau, inference=False)

        mnist_out_part = out_part[:bs_m]
        svhn_out_part = out_part[bs_m: bs_m + bs_s]
        cifar_out_part = out_part[bs_m + bs_s: bs_m + bs_s + bs_c]

        mnist_domain_out = domain_out[:bs_m]
        svhn_domain_out = domain_out[bs_m: bs_m + bs_s]
        cifar_domain_out = domain_out[bs_m + bs_s: bs_m + bs_s + bs_c]

        mnist_part_idx = part_idx[:bs_m]
        svhn_part_idx = part_idx[bs_m: bs_m + bs_s]
        cifar_part_idx = part_idx[bs_m + bs_s: bs_m + bs_s + bs_c]

        mnist_part_gumbel = part_gumbel[:bs_m]
        svhn_part_gumbel = part_gumbel[bs_m: bs_m + bs_s]
        cifar_part_gumbel = part_gumbel[bs_m + bs_s: bs_m + bs_s + bs_c]

        if i % 5 == 0:
            print(f"--- [Epoch {epoch + 1}, Batch {i}] Partition Stats ---")
            mnist_counts = torch.bincount(mnist_part_idx, minlength=args.num_partition)
            svhn_counts = torch.bincount(svhn_part_idx, minlength=args.num_partition)
            cifar_counts = torch.bincount(cifar_part_idx, minlength=args.num_partition)
            print(
                f"MNIST : {mnist_counts.cpu().numpy()} / SVHN  : {svhn_counts.cpu().numpy()} / CIFAR : {cifar_counts.cpu().numpy()}")
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

        mnist_label_loss = criterion(mnist_out_part, mnist_labels)
        svhn_label_loss = criterion(svhn_out_part, svhn_labels)
        cifar_label_loss = criterion(cifar_out_part, cifar_labels)

        # TODO it is cheating. it must be each domain std
        # numbers_part_gumbel = torch.cat((mnist_part_gumbel, svhn_part_gumbel))
        # objects_part_gumbel = torch.cat((cifar_part_gumbel, stl_part_gumbel))
        # avg_prob_numbers = torch.mean(numbers_part_gumbel, dim=0)
        # avg_prob_objects = torch.mean(objects_part_gumbel, dim=0)

        # TODO if you do this like it before, then the resulting avg_prob_numbers is shape of 2... which is the number of partitions.
        # this is genuinely wrong. it must be the probability of the partition idx. 
        # loss_specialization_numbers = -torch.sum(avg_prob_numbers * torch.log(avg_prob_numbers + 1e-8))
        # loss_specialization_objects = -torch.sum(avg_prob_objects * torch.log(avg_prob_objects + 1e-8))
        # TODO v2 I dont remember why we decided to use entropy? as there is no meaning of using it since we want each task's data to be specifically target one partition.
        # loss_specialization_mnist =  -torch.sum(mnist_part_gumbel[:, mnist_part_idx] * torch.log(mnist_part_gumbel[:, mnist_part_idx] + 1e-8))  
        # loss_specialization_svhn =  -torch.sum(svhn_part_gumbel[:, svhn_part_idx] * torch.log(svhn_part_gumbel[:, svhn_part_idx] + 1e-8))   
        # loss_specialization_cifar =  -torch.sum(cifar_part_gumbel[:, cifar_part_idx] * torch.log(cifar_part_gumbel[:, cifar_part_idx] + 1e-8))
        # loss_specialization_stl =  -torch.sum(stl_part_gumbel[:, stl_part_idx] * torch.log(stl_part_gumbel[:, stl_part_idx] + 1e-8))
        
        # this is fixed one, the lower the better. 
        loss_specialization_mnist = -torch.sum(mnist_part_gumbel*torch.log(mnist_part_gumbel), axis=1).mean()
        loss_specialization_svhn = -torch.sum(svhn_part_gumbel*torch.log(svhn_part_gumbel), axis=1).mean()
        loss_specialization_cifar = -torch.sum(cifar_part_gumbel*torch.log(cifar_part_gumbel), axis=1).mean()

        # loss_specialization = loss_specialization_numbers + loss_specialization_objects
        loss_specialization = loss_specialization_mnist + loss_specialization_svhn + loss_specialization_cifar 
        
        # check if it is nan because all batch are the same
        if torch.isnan(loss_specialization):
            print('caution')
        
        # TODO here, it is wrong again. you applied it for partitions not data.
        # all_probs = torch.cat((numbers_part_gumbel, objects_part_gumbel), dim=0)
        # avg_prob_global = torch.mean(all_probs, dim=0)
        # loss_diversity = torch.sum(avg_prob_global * torch.log(avg_prob_global + 1e-8))
        
        # TODO v2 I dont remember why we decided to use entropy? as there is no meaning of using it since we want each task's data to be specifically target one partition.
        # loss_diversity = 0
        # for part in range(args.num_partition):
        #     loss_diversity += torch.sum(part_gumbel[:, part] * torch.log(part_gumbel[:, part] + 1e-8))
        
        loss_diversity = torch.sum(part_gumbel.mean(0) * torch.log(part_gumbel.mean(0) + 1e-8))
        
        if torch.isnan(loss_diversity):
            print('caution diversity')

        label_loss = (mnist_label_loss + svhn_label_loss) + cifar_label_loss
        mnist_domain_loss = domain_criterion(mnist_domain_out, mnist_dlabels)
        svhn_domain_loss = domain_criterion(svhn_domain_out, svhn_dlabels)
        cifar_domain_loss = domain_criterion(cifar_domain_out, cifar_dlabels)
        
        domain_loss = (mnist_domain_loss + svhn_domain_loss) + cifar_domain_loss
        loss = label_loss + domain_loss + args.reg_alpha * loss_specialization + args.reg_beta * loss_diversity
        
        # if not inference:
        #     loss.backward(retain_graph=True)

        entries = []
        for name, param in model.partition_switcher.named_parameters():
            if param.grad is None:
                grad_str = 'None'
            else:
                grad_str = f"{torch.mean(torch.abs(param.grad)).item():.6f}"
            data_str = f"{torch.mean(torch.abs(param.data)).item():.6f}"
            entries.append(f"{name}: {grad_str}, {data_str}")

        print(" | ".join(entries) + f" | loss: {loss.item():.6f} ")

        if not inference:
            # 1. Classification optimizer 업데이트
            optimizers[0].zero_grad()
            optimizers[1].zero_grad()
            # 2. Domain optimizer 업데이트
            loss.backward()
            optimizers[0].step()
            optimizers[1].step()
        mnist_partition_counts += torch.bincount(mnist_part_idx, minlength=args.num_partition).to(device)
        svhn_partition_counts += torch.bincount(svhn_part_idx, minlength=args.num_partition).to(device)
        cifar_partition_counts += torch.bincount(cifar_part_idx, minlength=args.num_partition).to(device)

        total_label_loss += label_loss.item() * (bs_m + bs_s + bs_c)  # °¡Áß Æò±ÕÀ» À§ÇØ ¹èÄ¡ Å©±â °öÇÔ
        total_mnist_loss += mnist_label_loss.item() * bs_m
        total_svhn_loss += svhn_label_loss.item() * bs_s
        total_cifar_loss += cifar_label_loss.item() * bs_c

        total_domain_loss += domain_loss.item() * (bs_m + bs_s + bs_c)
        total_mnist_domain_loss += mnist_domain_loss.item() * bs_m
        total_svhn_domain_loss += svhn_domain_loss.item() * bs_s
        total_cifar_domain_loss += cifar_domain_loss.item() * bs_c

        total_specialization_loss += loss_specialization.item() * (bs_m + bs_s + bs_c)
        total_diversity_loss += loss_diversity.item() * (bs_m + bs_s + bs_c)
        total_loss += loss.item() * (bs_m + bs_s + bs_c)

        total_mnist_correct += (torch.argmax(mnist_out_part, dim=1) == mnist_labels).sum().item()
        total_svhn_correct += (torch.argmax(svhn_out_part, dim=1) == svhn_labels).sum().item()
        total_cifar_correct += ((torch.argmax(cifar_out_part, dim=1) == cifar_labels).sum().item())

        total_mnist_domain_correct += (torch.argmax(mnist_domain_out, dim=1) == mnist_dlabels).sum().item()
        total_svhn_domain_correct += (torch.argmax(svhn_domain_out, dim=1) == svhn_dlabels).sum().item()
        total_cifar_domain_correct += ((torch.argmax(cifar_domain_out, dim=1) == cifar_dlabels).sum().item())

        total_samples_m += bs_m
        total_samples_s += bs_s
        total_samples_c += bs_c

    
    if total_samples_m == 0: 
        total_samples_m = 1
    if total_samples_s == 0: 
        total_samples_s = 1
    if total_samples_c == 0: 
        total_samples_c = 1

    if not inference:
        prefix = 'Train'
    else:
        prefix = 'Test'
        
    total_samples_all = total_samples_m + total_samples_s + total_samples_c

    mnist_train_partition_log = get_label_partition_log_data(
        mnist_label_partition_counts, 'MNIST', args.num_classes, args.num_partition, prefix=prefix
    )
    svhn_train_partition_log = get_label_partition_log_data(
        svhn_label_partition_counts, 'SVHN', args.num_classes, args.num_partition, prefix=prefix
    )
    cifar_train_partition_log = get_label_partition_log_data(
        cifar_label_partition_counts, 'CIFAR', args.num_classes, args.num_partition, prefix=prefix
    )
    

    mnist_partition_ratios = mnist_partition_counts / total_samples_m * 100
    svhn_partition_ratios = svhn_partition_counts / total_samples_s * 100
    cifar_partition_ratios = cifar_partition_counts / total_samples_c * 100

    mnist_partition_ratio_str = " | ".join(
        [f"Partition {p}: {mnist_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
    svhn_partition_ratio_str = " | ".join(
        [f"Partition {p}: {svhn_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
    cifar_partition_ratio_str = " | ".join(
        [f"Partition {p}: {cifar_partition_ratios[p]:.2f}%" for p in range(args.num_partition)])
    
    mnist_domain_avg_loss = total_mnist_domain_loss / total_samples_m
    svhn_domain_avg_loss = total_svhn_domain_loss / total_samples_s
    cifar_domain_avg_loss = total_cifar_domain_loss / total_samples_c

    mnist_avg_loss = total_mnist_loss / total_samples_m
    svhn_avg_loss = total_svhn_loss / total_samples_s
    cifar_avg_loss = total_cifar_loss / total_samples_c

    domain_avg_loss = total_domain_loss / total_samples_all
    label_avg_loss = total_label_loss / total_samples_all
    specialization_loss = total_specialization_loss / total_samples_all
    diversity_loss = total_diversity_loss / total_samples_all
    total_avg_loss = total_loss / total_samples_all

    mnist_acc_epoch = total_mnist_correct / total_samples_m * 100
    svhn_acc_epoch = total_svhn_correct / total_samples_s * 100
    cifar_acc_epoch = total_cifar_correct / total_samples_c * 100

    mnist_domain_acc_epoch = total_mnist_domain_correct / total_samples_m * 100
    svhn_domain_acc_epoch = total_svhn_domain_correct / total_samples_s * 100
    cifar_domain_acc_epoch = total_cifar_domain_correct / total_samples_c * 100
    
    best_avg_acc = (mnist_acc_epoch + svhn_acc_epoch + cifar_acc_epoch) / 3 

    if not inference:
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
    else:
        print(f'Epoch [{epoch + 1}/{num_epochs}] (Test)')
    print(
        f'  [Ratios] MNIST: [{mnist_partition_ratio_str}] | SVHN: [{svhn_partition_ratio_str}] | CIFAR: [{cifar_partition_ratio_str}]')
    print(
        f'  [Acc]    MNIST: {mnist_acc_epoch:<6.2f}% | SVHN: {svhn_acc_epoch:<6.2f}% | CIFAR: {cifar_acc_epoch:<6.2f}%')
    print(
        f'  [DomAcc] MNIST: {mnist_domain_acc_epoch:<6.2f}% | SVHN: {svhn_domain_acc_epoch:<6.2f}% | CIFAR: {cifar_domain_acc_epoch:<6.2f}%')
    print(f'  [Reg]    Spec:  {specialization_loss:<8.4f} | Div:    {diversity_loss:<8.4f} | Tau: {tau:<5.3f}')
    print(
        f'  [Loss]   Label: {label_avg_loss:<8.4f} | Domain: {domain_avg_loss:<8.4f} | Total: {total_avg_loss:<8.4f}')
    print(
        f'  [Label]  MNIST: {mnist_avg_loss:<6.4f} | SVHN: {svhn_avg_loss:<6.4f} | CIFAR: {cifar_avg_loss:<6.4f}')
    print(
        f'  [Domain] MNIST: {mnist_domain_avg_loss:<6.4f} | SVHN: {svhn_domain_avg_loss:<6.4f} | CIFAR: {cifar_domain_avg_loss:<6.4f}')

    wandb.log({f"{prefix}/Best Avg Accuracy": best_avg_acc}, step=epoch + 1)

    wandb.log({
        **{f"{prefix}/Partition {p} MNIST Ratio": mnist_partition_ratios[p].item() for p in range(args.num_partition)},
        **{f"{prefix}/Partition {p} SVHN Ratio": svhn_partition_ratios[p].item() for p in range(args.num_partition)},
        **{f"{prefix}/Partition {p} CIFAR Ratio": cifar_partition_ratios[p].item() for p in range(args.num_partition)},
        f'{prefix}/Label MNIST Accuracy': mnist_acc_epoch,
        f'{prefix}/Label SVHN Accuracy': svhn_acc_epoch,
        f'{prefix}/Label CIFAR Accuracy': cifar_acc_epoch,
        f'{prefix}/Domain MNIST Accuracy': mnist_domain_acc_epoch,
        f'{prefix}/Domain SVHN Accuracy': svhn_domain_acc_epoch,
        f'{prefix}/Domain CIFAR Accuracy': cifar_domain_acc_epoch,
        f'{prefix}Loss/Label MNIST Loss': mnist_avg_loss,
        f'{prefix}Loss/Label SVHN Loss': svhn_avg_loss,
        f'{prefix}Loss/Label CIFAR Loss': cifar_avg_loss,
        f'{prefix}Loss/Label Loss': label_avg_loss,
        f'{prefix}Loss/Domain MNIST Loss': mnist_domain_avg_loss,
        f'{prefix}Loss/Domain SVHN Loss': svhn_domain_avg_loss,
        f'{prefix}Loss/Domain CIFAR Loss': cifar_domain_avg_loss,
        f'{prefix}Loss/Domain Loss': domain_avg_loss,
        f'{prefix}Loss/Specialization Loss': specialization_loss,
        f'{prefix}Loss/Diversity Loss': diversity_loss,
        f'{prefix}Loss/Total Loss': total_avg_loss,
        'Parameters/Tau': tau,
        'Parameters/Learning Rate': optimizers[0].param_groups[0]['lr'],
        'Parameters/Learning Rate Domain': optimizers[1].param_groups[0]['lr'],
        'Parameters/Lambda_p': lambda_p,
        **mnist_train_partition_log,
        **svhn_train_partition_log,
        **cifar_train_partition_log,
    }, step=epoch + 1)


def get_cosine_decay_with_hold(current_epoch):
    decay_epochs = int(args.epoch * args.lr_domain_prop)
    min_lr_ratio = args.lr_domain_min / args.lr_domain
    if current_epoch < decay_epochs:
        # 1.0 (시작) 에서 min_lr_ratio (끝) 까지 코사인 곡선으로 감소
        progress = float(current_epoch) / float(decay_epochs)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # 1.0 과 min_lr_ratio 사이를 cosine_factor로 보간
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor
    else:
        # decay_epochs 이후에는 최소 배율 유지
        return min_lr_ratio


def main():
    init_lambda = args.lambda_p
    num_epochs = args.epoch

    wandb_run = wandb.init(
                           config=args.__dict__,
                           project="TagNet MSC sweep everything 128",
                           name="[TagnetMLP]MSC_UniqueDomain_LpFixed_probGB_Entropy_lr:" + str(args.lr)
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
    cifar_loader, cifar_loader_test = data_loader('CIFAR10', args.batch_size*2)

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
    save_interval = 100 # TODO changed
    min_save_epoch = 10 # TODO changed
    
    weights_class, weights_domain = TagNet_ClassDomain_weights(
        model,
        lr=args.lr,
        pre_weight=args.prefc_lr,
        disc_weight=args.disc_lr,
        fc_weight=args.fc_lr,
        switcher_weight=args.switcher_lr
    )

    class_optimizer = optim.Adam(weights_class, lr=args.lr, weight_decay=args.opt_decay)
    domain_optimizer = optim.Adam(weights_domain, lr=args.lr_domain, weight_decay=args.opt_decay)
    
    # LambdaLR 스케줄러 생성
    scheduler_cosine = LambdaLR(domain_optimizer, lr_lambda=get_cosine_decay_with_hold)

    tau_scheduler = GumbelTauScheduler(initial_tau=args.init_tau, min_tau=args.min_tau, decay_rate=args.tau_decay)
    domain_criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loader_zip = zip(mnist_loader, svhn_loader, cifar_loader)
        test_loader_zip = zip(mnist_loader_test, svhn_loader_test, cifar_loader_test)
        phi = (1 + math.sqrt(5)) / 2
        # lambda_p = init_lambda / phi ** (epoch / 10)
        lambda_p = init_lambda
        tau = tau_scheduler.get_tau()

        train_step(epoch, model, args, (class_optimizer, domain_optimizer), criterion, domain_criterion, train_loader_zip, phi, lambda_p, tau, inference = False)
        train_step(epoch, model, args, (class_optimizer, domain_optimizer), criterion, domain_criterion, test_loader_zip, phi, lambda_p, tau, inference = True)
        tau_scheduler.step()
        scheduler_cosine.step()

    final_save_path = os.path.join(save_dir, f"final_model_epoch_{num_epochs}.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"--- Final model saved to {final_save_path} ---")


if __name__ == '__main__':
    main()