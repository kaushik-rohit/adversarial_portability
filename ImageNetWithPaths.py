from torchvision import datasets

class ImageNetWithPaths(datasets.ImageNet):

    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageNetWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
