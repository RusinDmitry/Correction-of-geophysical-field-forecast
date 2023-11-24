import torch


class StandardScaler:
    def __init__(self):
        self.channel_means = None
        self.channel_stddevs = None
        self.channels = None
        self.mean = None
        self.stddev = None
        self.channels_dim = None

    def fit_transform(self, tensor):
        self.mean = tensor.mean(0, keepdim=True)
        self.stddev = tensor.std(0, unbiased=False, keepdim=True)
        tensor -= self.mean
        tensor /= self.stddev
        return tensor

    def channel_fit_transform(self, tensor, channels=None, channels_dim=1):
        self.channels_dim = channels_dim
        self.channel_means = []
        self.channel_stddevs = []
        if not channels:
            self.channels = range(tensor.shape[1])
        else:
            self.channels = channels
        tensor = list(torch.split(tensor, 1, dim=channels_dim))
        for i, channel in enumerate(tensor):
            if i in self.channels:
                self.channel_means.append(torch.mean(channel))
                self.channel_stddevs.append(torch.std(channel))
                channel -= self.channel_means[-1]
                channel /= self.channel_stddevs[-1]
                tensor[i] = channel
        tensor = torch.cat(tensor, dim=channels_dim)
        return tensor

    def channel_inverse_transform(self, tensor, channels_dim=None):
        if not channels_dim:
            channels_dim = self.channels_dim
        tensor = list(torch.split(tensor, 1, dim=channels_dim))
        for i in range(min(len(self.channels), len(tensor))):
            tensor[self.channels[i]] = tensor[self.channels[i]] * self.channel_stddevs[i]
            tensor[self.channels[i]] = tensor[self.channels[i]] + self.channel_means[i]
        tensor = torch.cat(tensor, dim=channels_dim)
        return tensor

    def channel_fit(self, tensor, channels=None, channels_dim=1):
        self.channels_dim = channels_dim
        self.channel_means = []
        self.channel_stddevs = []
        if not channels:
            self.channels = range(tensor.shape[channels_dim])
        else:
            self.channels = channels
        tensor = torch.split(tensor, 1, dim=self.channels_dim)
        for i, channel in enumerate(tensor):
            if i in self.channels:
                self.channel_means.append(torch.mean(channel))
                self.channel_stddevs.append(torch.std(channel))

    def apply_scaler_channel_params(self, means, stds, channels=None):
        self.channel_means = means
        self.channel_stddevs = stds
        if channels is None:
            self.channels = list(range(len(means)))

    def channel_transform(self, tensor, channels_dim=None):
        if not channels_dim:
            channels_dim = self.channels_dim
        tensor = list(torch.split(tensor, 1, dim=channels_dim))
        j = 0
        for i, channel in enumerate(tensor):
            if i in self.channels:
                channel -= self.channel_means[j]
                channel /= self.channel_stddevs[j]
                tensor[i] = channel
                j += 1
        tensor = torch.cat(tensor, dim=channels_dim)
        return tensor

    def fit(self, tensor):
        self.mean = tensor.mean(0, keepdim=True)
        self.stddev = tensor.std(0, unbiased=False, keepdim=True)
        self.max = tensor.max()

    def transform(self, tensor):
        tensor -= self.mean
        tensor /= self.stddev
        return tensor

    def flip_transform(self, tensor):
        tensor = -tensor
        tensor += self.max
        tensor -= self.mean
        tensor /= self.stddev
        return tensor

    def inverse_transform(self, tensor):
        tensor *= self.stddev
        tensor += self.mean
        return tensor