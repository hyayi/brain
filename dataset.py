class MRSDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, transforms):
        self.data_df = data_df
        self.transforms = transforms

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):

        image_dict = {'img' : self.data_df['image'][index]}
        label = torch.FloatTensor(self.data_df['label'][index])
        return self.transforms(image_dict), label
