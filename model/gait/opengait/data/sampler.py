import torch.utils.data as tordata


class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        indx_batch_per_rank = []

        for i in range(int(self.size)):
            indx_batch_per_rank.append(
                indices[i:i+1])

        self.idx_batch_this_rank = indx_batch_per_rank

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)