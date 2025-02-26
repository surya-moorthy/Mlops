import  torch
from torch import nn , save ,load
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# train = datasets.MNIST(root="data",download=True,train=True,transform=ToTensor())
# dataset = DataLoader(train, batch_size=32)

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,32,(3,3)),
            nn.ReLU(),
            nn.Conv2d(32,64,(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64,(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10)
        ) 
    def forward(self,x):
        return self.model(x)

clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(),lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    # for epoch in range(10):
    #     for batch in dataset:
    #         X,y = batch
    #         X,y = X.to('cuda'),y.to('cuda')
    #         yhat = clf(X)
    #         loss = loss_fn(yhat,y)

    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #     print(f"Epoch {epoch} loss is {loss.item()}")
    # with open('model_state.pt',"wb") as f:  ## write a binary file "wb" --> write binary
    #     save(clf.state_dict(),f)
    with open('model_state.pt',"rb") as f:
        clf.load_state_dict(load(f))
    img = Image.open('img_2.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    print(torch.argmax(clf(img_tensor)))