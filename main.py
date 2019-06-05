import skimage
import sys
import torch
import torchvision

print("Style Transfer")

preprocessing = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
postprocessing = torchvision.transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor(), preprocessing])
def loadImage(filename):
    try:
        image = skimage.io.imread(filename)
        return transform(image).unsqueeze(0)
    except Exception as e:
        print >> sys.stderr, "Error loading file " + filename + " ["+ str(e) +"]"

class StyleTransfer(torch.nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()
        # Load pretrained model
        vgg = torchvision.models.vgg19(pretrained=True).features
        print(vgg)
        # Re-assign layers as class members
        self.conv1_1 = vgg[0]
        self.conv1_2 = vgg[2]        
        self.conv2_1 = vgg[5]
        self.conv2_2 = vgg[7]
        self.conv3_1 = vgg[10]
        self.conv3_2 = vgg[12]
        self.conv3_3 = vgg[14]
        self.conv3_4 = vgg[16]
        self.conv4_1 = vgg[19]
        self.conv4_2 = vgg[21]
        self.conv4_3 = vgg[23]
        self.conv4_4 = vgg[25]
        self.conv5_1 = vgg[28]

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1_1(x))
        self.features1_1 = x
        x = torch.nn.functional.relu(self.conv1_2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2_1(x))
        self.features2_1 = x
        x = torch.nn.functional.relu(self.conv2_2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv3_1(x))
        self.features3_1 = x
        x = torch.nn.functional.relu(self.conv3_2(x))
        x = torch.nn.functional.relu(self.conv3_3(x))
        x = torch.nn.functional.relu(self.conv3_4(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv4_1(x))
        self.features4_1 = x
        x = torch.nn.functional.relu(self.conv4_2(x))
        x = torch.nn.functional.relu(self.conv4_3(x))
        x = torch.nn.functional.relu(self.conv4_4(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv5_1(x))
        self.features5_1 = x
        return x

    def gram(self, x):
        x = x.view(x.shape[1], -1)
        return torch.mm(x, x.t())

    def setStyle(self, x):
        self.forward(x)
        self.target1_1 = self.gram(self.features1_1).data.clone()
        self.target2_1 = self.gram(self.features2_1).data.clone()
        self.target3_1 = self.gram(self.features3_1).data.clone()
        self.target4_1 = self.gram(self.features4_1).data.clone()
        self.target5_1 = self.gram(self.features5_1).data.clone()

    def computeLoss(self, x):
        self.forward(x)
        loss = torch.nn.functional.mse_loss(self.gram(self.features1_1), self.target1_1)
        loss += torch.nn.functional.mse_loss(self.gram(self.features2_1), self.target2_1)
        loss += torch.nn.functional.mse_loss(self.gram(self.features3_1), self.target3_1)
        loss += torch.nn.functional.mse_loss(self.gram(self.features4_1), self.target4_1)
        loss += torch.nn.functional.mse_loss(self.gram(self.features5_1), self.target5_1)
        return loss

    def optimise(self):
        canvas = torch.randn((1, 3, 256, 256)).cuda()
        canvas.requires_grad = True
        optimizer = torch.optim.Adam([canvas], 1)
        for i in range(1000):
            optimizer.zero_grad()
            loss = self.computeLoss(canvas)
            print(i, loss.item())
            loss.backward()
            optimizer.step()
        return canvas[0]

model = StyleTransfer().cuda()
print(model)
    
# Load input
style = loadImage(sys.argv[1]).cuda()
print("Style", sys.argv[1], style.shape)

# Run input
model.setStyle(style)
result = model.optimise()
torchvision.utils.save_image(postprocessing(result), "result.png")
