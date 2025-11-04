import torch
import torch.nn as nn
from functions.ReverseLayerF import ReverseLayerF
from torch.nn.functional import gumbel_softmax

class TagNet(nn.Module):
    def __init__(self, num_classes=10, pre_classifier_out=1024, n_partition=2, part_layer=128, num_domains=2,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TagNet, self).__init__()
        self.device = device
        self.n_partition = n_partition
        self.disc_hidden = 3 * num_classes * num_domains * n_partition

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2), # 224 -> 112
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # 112-> 56
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # 56 -> 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # 28 -> 14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.pre_classifier = nn.Sequential(
            nn.Linear(64 * 14 * 14, pre_classifier_out),
            nn.LayerNorm(pre_classifier_out),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(pre_classifier_out, part_layer),
            nn.BatchNorm1d(part_layer),
            nn.ReLU(),
            nn.Linear(part_layer, num_classes)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(pre_classifier_out, self.disc_hidden),
            nn.BatchNorm1d(self.disc_hidden),
            nn.ReLU(),
        )

        self.discriminator_fc = nn.Linear(self.disc_hidden, num_domains)

        self.partition_switcher = nn.Linear(self.disc_hidden, n_partition)

        self.create_partitioned_classifier()
        self.sync_classifier_with_subnetworks()
        self.to(self.device)

    # Method to partition the classifier into sub-networks
    def create_partitioned_classifier(self):
        self.partitioned_classifier = nn.ModuleList()

        linear_layers = []
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)

        for p_i in range(self.n_partition):
            partitioned_layer = nn.ModuleList()

            for i, linear_layer in enumerate(linear_layers):
                if i == 0:
                    input_size = linear_layer.in_features
                    output_size = linear_layer.out_features
                    partition_size = output_size // self.n_partition

                    sublayer = nn.Linear(input_size, partition_size)

                    partitioned_layer.append(sublayer)
                    partitioned_layer.append(nn.ReLU(inplace=True))

                elif i == len(linear_layers) - 1:
                    input_size = linear_layer.in_features
                    output_size = linear_layer.out_features
                    partition_size = input_size // self.n_partition

                    sublayer = nn.Linear(partition_size, output_size)
                    partitioned_layer.append(sublayer)

            self.partitioned_classifier.append(partitioned_layer)

    def sync_classifier_with_subnetworks(self):
        linear_layers = [layer for layer in self.classifier if isinstance(layer, nn.Linear)]
        linear_layers_subnet = [[layer for layer in partitioned_classifier if isinstance(layer, nn.Linear)] for
                                partitioned_classifier in self.partitioned_classifier]

        for i, linear_layer in enumerate(linear_layers):
            ws_ = []
            bs_ = []

            # Subnet weights/biases 수집
            for j, subnet_layer in enumerate(linear_layers_subnet):
                ws_.append(subnet_layer[i].weight.data)
                if i == 1:
                    bs_.append(subnet_layer[i].bias.data)

            with torch.no_grad():
                if i == 0:
                    ws_cat = torch.cat(ws_, dim=0)
                    linear_layer.weight.copy_(ws_cat)

                elif i == 1:
                    ws_cat = torch.cat(ws_, dim=1)
                    bs_cat = torch.sum(torch.stack(bs_), dim=0)

                    linear_layer.weight.copy_(ws_cat)

                    if len(bs_) > 0:
                        linear_layer.bias.copy_(bs_cat)

    def forward(self, input_data, alpha=1.0, tau=0.1, inference=False):
        feature = self.features(input_data)
        # feature = input_data
        feature = feature.view(feature.size(0), -1)
        feature = self.pre_classifier(feature)

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_penul = self.discriminator(reverse_feature)
        domain_output = self.discriminator_fc(domain_penul)
        partition_switcher_output = self.partition_switcher(domain_penul)

        if inference:
            partition_gumbel_or_probs = torch.softmax(partition_switcher_output, dim=1)
            partition_idx = torch.argmax(partition_gumbel_or_probs, dim=1)
        else:
            partition_gumbel_or_probs = gumbel_softmax(partition_switcher_output, tau=tau, hard=True)
            partition_idx = torch.argmax(partition_gumbel_or_probs, dim=1)

        # class_output_partitioned = torch.zeros(feature.size(0), self.classifier[-1].out_features, device=self.device)
        class_output_partitioned = torch.zeros(feature.size(0), self.partitioned_classifier[0][-1].out_features, device=self.device)
        for p_i in range(self.n_partition):
            indices = torch.where(partition_idx == p_i)[0]

            if len(indices) == 0:
                continue

            selected_features = feature[indices]

            xx = selected_features
            for layer in self.partitioned_classifier[p_i]:
                xx = layer(xx)

            class_output_partitioned[indices] = xx

        if self.training:
            self.sync_classifier_with_subnetworks()

        return class_output_partitioned, domain_output, partition_idx, partition_gumbel_or_probs


class TagNet32(nn.Module):
    def __init__(self, num_classes=10, pre_classifier_out=1024, n_partition=2, part_layer=128, num_domains=2,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TagNet32, self).__init__()
        self.device = device
        self.n_partition = n_partition
        self.disc_hidden = 3 * num_classes * num_domains * n_partition

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=0), # 32 to 15
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0), # 15 to 7
        )

        self.pre_classifier = nn.Sequential(
            nn.Linear(3 * 32 * 32, pre_classifier_out),
            nn.LayerNorm(pre_classifier_out),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(pre_classifier_out, part_layer),
            nn.BatchNorm1d(part_layer),
            nn.ReLU(),
            nn.Linear(part_layer, num_classes)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(pre_classifier_out, self.disc_hidden),
            nn.BatchNorm1d(self.disc_hidden),
            nn.ReLU(),
        )

        self.discriminator_fc = nn.Linear(self.disc_hidden, num_domains)

        self.partition_switcher = nn.Linear(self.disc_hidden, n_partition)

        self.create_partitioned_classifier()
        self.sync_classifier_with_subnetworks()
        self.to(self.device)


    def create_partitioned_classifier(self):
        self.partitioned_classifier = nn.ModuleList()

        linear_layers = []
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)

        for p_i in range(self.n_partition):
            partitioned_layer = nn.ModuleList()

            for i, linear_layer in enumerate(linear_layers):
                if i == 0:
                    input_size = linear_layer.in_features
                    output_size = linear_layer.out_features
                    partition_size = output_size // self.n_partition

                    sublayer = nn.Linear(input_size, partition_size)

                    partitioned_layer.append(sublayer)
                    partitioned_layer.append(nn.ReLU(inplace=True))

                elif i == len(linear_layers) - 1:
                    input_size = linear_layer.in_features
                    output_size = linear_layer.out_features
                    partition_size = input_size // self.n_partition

                    sublayer = nn.Linear(partition_size, output_size)
                    partitioned_layer.append(sublayer)

            self.partitioned_classifier.append(partitioned_layer)

    def sync_classifier_with_subnetworks(self):
        linear_layers = [layer for layer in self.classifier if isinstance(layer, nn.Linear)]
        linear_layers_subnet = [[layer for layer in partitioned_classifier if isinstance(layer, nn.Linear)] for
                                partitioned_classifier in self.partitioned_classifier]

        for i, linear_layer in enumerate(linear_layers):
            ws_ = []
            bs_ = []

            for j, subnet_layer in enumerate(linear_layers_subnet):
                ws_.append(subnet_layer[i].weight.data)
                if i == 1:
                    bs_.append(subnet_layer[i].bias.data)

            with torch.no_grad():
                if i == 0:
                    ws_cat = torch.cat(ws_, dim=0)
                    linear_layer.weight.copy_(ws_cat)

                elif i == 1:
                    ws_cat = torch.cat(ws_, dim=1)
                    bs_cat = torch.sum(torch.stack(bs_), dim=0)

                    linear_layer.weight.copy_(ws_cat)

                    if len(bs_) > 0:
                        linear_layer.bias.copy_(bs_cat)

    def forward(self, input_data, alpha=1.0, tau=0.1, inference=False):
        # feature = self.features(input_data)
        feature = input_data
        feature = feature.view(feature.size(0), -1)
        feature = self.pre_classifier(feature)

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_penul = self.discriminator(reverse_feature)
        domain_output = self.discriminator_fc(domain_penul)

        partition_switcher_output = self.partition_switcher(domain_penul)

        if inference:
            partition_gumbel_or_probs = torch.softmax(partition_switcher_output, dim=1)
            partition_idx = torch.argmax(partition_gumbel_or_probs, dim=1)
        else:
            partition_gumbel_or_probs = gumbel_softmax(partition_switcher_output, tau=tau, hard=True)
            partition_idx = torch.argmax(partition_gumbel_or_probs, dim=1)

        # class_output_partitioned = torch.zeros(feature.size(0), self.classifier[-1].out_features, device=self.device)
        class_output_partitioned = torch.zeros(feature.size(0), self.partitioned_classifier[0][-1].out_features, device=self.device)
        for p_i in range(self.n_partition):
            indices = torch.where(partition_idx == p_i)[0]

            if len(indices) == 0:
                continue

            selected_features = feature[indices]

            xx = selected_features
            for layer in self.partitioned_classifier[p_i]:
                xx = layer(xx)

            class_output_partitioned[indices] = xx

        if self.training:
            self.sync_classifier_with_subnetworks()

        return class_output_partitioned, domain_output, partition_idx, partition_gumbel_or_probs

class TagNet32_woLayernorm(nn.Module):
    def __init__(self, num_classes=10, pre_classifier_out=1024, n_partition=2, part_layer=128, num_domains=2,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(TagNet32, self).__init__()
        self.device = device
        self.n_partition = n_partition
        self.disc_hidden = 3 * num_classes * num_domains * n_partition

        self.pre_classifier = nn.Sequential(
            nn.Linear(3 * 32 * 32, pre_classifier_out),
            # nn.LayerNorm(pre_classifier_out),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(pre_classifier_out, part_layer),
            nn.BatchNorm1d(part_layer),
            nn.ReLU(),
            nn.Linear(part_layer, num_classes)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(pre_classifier_out, self.disc_hidden),
            nn.BatchNorm1d(self.disc_hidden),
            nn.ReLU(),
        )

        self.discriminator_fc = nn.Linear(self.disc_hidden, num_domains)

        self.partition_switcher = nn.Linear(self.disc_hidden, n_partition)

        self.create_partitioned_classifier()
        self.sync_classifier_with_subnetworks()
        self.to(self.device)


    def create_partitioned_classifier(self):
        self.partitioned_classifier = nn.ModuleList()

        linear_layers = []
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)

        for p_i in range(self.n_partition):
            partitioned_layer = nn.ModuleList()

            for i, linear_layer in enumerate(linear_layers):
                if i == 0:
                    input_size = linear_layer.in_features
                    output_size = linear_layer.out_features
                    partition_size = output_size // self.n_partition

                    sublayer = nn.Linear(input_size, partition_size)

                    partitioned_layer.append(sublayer)
                    partitioned_layer.append(nn.ReLU(inplace=True))

                elif i == len(linear_layers) - 1:
                    input_size = linear_layer.in_features
                    output_size = linear_layer.out_features
                    partition_size = input_size // self.n_partition

                    sublayer = nn.Linear(partition_size, output_size)
                    partitioned_layer.append(sublayer)

            self.partitioned_classifier.append(partitioned_layer)

    def sync_classifier_with_subnetworks(self):
        linear_layers = [layer for layer in self.classifier if isinstance(layer, nn.Linear)]
        linear_layers_subnet = [[layer for layer in partitioned_classifier if isinstance(layer, nn.Linear)] for
                                partitioned_classifier in self.partitioned_classifier]

        for i, linear_layer in enumerate(linear_layers):
            ws_ = []
            bs_ = []

            for j, subnet_layer in enumerate(linear_layers_subnet):
                ws_.append(subnet_layer[i].weight.data)
                if i == 1:
                    bs_.append(subnet_layer[i].bias.data)

            with torch.no_grad():
                if i == 0:
                    ws_cat = torch.cat(ws_, dim=0)
                    linear_layer.weight.copy_(ws_cat)

                elif i == 1:
                    ws_cat = torch.cat(ws_, dim=1)
                    bs_cat = torch.sum(torch.stack(bs_), dim=0)

                    linear_layer.weight.copy_(ws_cat)

                    if len(bs_) > 0:
                        linear_layer.bias.copy_(bs_cat)

    def forward(self, input_data, alpha=1.0, tau=0.1, inference=False):
        # feature = self.features(input_data)
        feature = input_data
        feature = feature.view(feature.size(0), -1)
        feature = self.pre_classifier(feature)

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_penul = self.discriminator(reverse_feature)
        domain_output = self.discriminator_fc(domain_penul)

        partition_switcher_output = self.partition_switcher(domain_penul)

        if inference:
            partition_gumbel_or_probs = torch.softmax(partition_switcher_output, dim=1)
            partition_idx = torch.argmax(partition_gumbel_or_probs, dim=1)
        else:
            partition_gumbel_or_probs = gumbel_softmax(partition_switcher_output, tau=tau, hard=True)
            partition_idx = torch.argmax(partition_gumbel_or_probs, dim=1)

        # class_output_partitioned = torch.zeros(feature.size(0), self.classifier[-1].out_features, device=self.device)
        class_output_partitioned = torch.zeros(feature.size(0), self.partitioned_classifier[0][-1].out_features, device=self.device)
        for p_i in range(self.n_partition):
            indices = torch.where(partition_idx == p_i)[0]

            if len(indices) == 0:
                continue

            selected_features = feature[indices]

            xx = selected_features
            for layer in self.partitioned_classifier[p_i]:
                xx = layer(xx)

            class_output_partitioned[indices] = xx

        if self.training:
            self.sync_classifier_with_subnetworks()

        return class_output_partitioned, domain_output, partition_idx, partition_gumbel_or_probs


def TagNet_weights(model, lr, pre_weight=1.0, fc_weight=1.0, disc_weight=1.0, switcher_weight=1.0):
    return [
        {'params': model.features.parameters(), 'lr': lr},
        {'params': model.pre_classifier.parameters(), 'lr': lr * pre_weight},
        {'params': model.discriminator.parameters(), 'lr': lr * disc_weight},
        {'params': model.discriminator_fc.parameters(), 'lr': lr * disc_weight},
        {'params': model.partitioned_classifier.parameters(), 'lr': lr * fc_weight},
        {'params': model.partition_switcher.parameters(), 'lr': lr * switcher_weight},
    ]
