import os
import skimage
import sys
import torch
import torchvision
from torch.utils.cpp_extension import load

print("Style Transfer")
if len(sys.argv) <= 1:
    print("Usage: " + sys.argv[0] + " STYLE_IMAGE_FILES ...")
    sys.exit(0)


preprocessing = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
postprocessing = torchvision.transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor(), preprocessing])
cpp = torch.utils.cpp_extension.load(name="histogram_cpp", sources=["histogram.cpp", "histogram.cu"])
def loadImage(filename):
    try:
        image = skimage.io.imread(filename)[:,:,:3]
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
        self.learning_rate = 0.05
        self.histogram = True

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

        self.min_1 = torch.min(self.features1_1[0].view(self.features1_1.shape[1], -1), 1)[0].data.clone()
        self.max_1 = torch.max(self.features1_1[0].view(self.features1_1.shape[1], -1), 1)[0].data.clone()
        self.hist_1 = cpp.computeHistogram(self.features1_1[0], 256)

        self.min_2 = torch.min(self.features2_1[0].view(self.features2_1.shape[1], -1), 1)[0].data.clone()
        self.max_2 = torch.max(self.features2_1[0].view(self.features2_1.shape[1], -1), 1)[0].data.clone()
        self.hist_2 = cpp.computeHistogram(self.features2_1[0], 256)

        self.min_3 = torch.min(self.features3_1[0].view(self.features3_1.shape[1], -1), 1)[0].data.clone()
        self.max_3 = torch.max(self.features3_1[0].view(self.features3_1.shape[1], -1), 1)[0].data.clone()
        self.hist_3 = cpp.computeHistogram(self.features3_1[0], 256)

        self.min_4 = torch.min(self.features4_1[0].view(self.features4_1.shape[1], -1), 1)[0].data.clone()
        self.max_4 = torch.max(self.features4_1[0].view(self.features4_1.shape[1], -1), 1)[0].data.clone()
        self.hist_4 = cpp.computeHistogram(self.features4_1[0], 256)

        self.min_5 = torch.min(self.features5_1[0].view(self.features5_1.shape[1], -1), 1)[0].data.clone()
        self.max_5 = torch.max(self.features5_1[0].view(self.features5_1.shape[1], -1), 1)[0].data.clone()
        self.hist_5 = cpp.computeHistogram(self.features5_1[0], 256)

    def computeHistogramMatchedActivation(self, t, h, minv, maxv):
        assert(len(t.shape) == 3)
        assert(len(minv.shape) == 1)
        assert(len(maxv.shape) == 1)
        assert(h.shape[0] == t.shape[0])
        assert(minv.shape[0] == t.shape[0])
        assert(maxv.shape[0] == t.shape[0])
        assert(h.shape[1] == 256)
        res = t.data.clone() # Clone, we don't want to change the values of features map or target histogram
        cpp.matchHistogram(res, h.clone())
        for c in range(res.size(0)):
            res[c].mul_(maxv[c] - minv[c]) # Values in range [0, max - min]
            res[c].add_(minv[c])           # Values in range [min, max]            
        return res.data.unsqueeze(0)

    def computeLoss(self, x):
        self.forward(x)

        loss = torch.nn.functional.mse_loss(self.gram(self.features1_1), self.target1_1)
        loss += torch.nn.functional.mse_loss(self.gram(self.features2_1), self.target2_1)
        loss += torch.nn.functional.mse_loss(self.gram(self.features3_1), self.target3_1)
        loss += torch.nn.functional.mse_loss(self.gram(self.features4_1), self.target4_1)
        loss += torch.nn.functional.mse_loss(self.gram(self.features5_1), self.target5_1)

        if self.histogram:
            histogramCorrectedTarget = self.computeHistogramMatchedActivation(self.features1_1[0], self.hist_1, self.min_1, self.max_1)
            assert(histogramCorrectedTarget.shape == self.features1_1.shape)
            loss += torch.nn.functional.mse_loss(self.features1_1, histogramCorrectedTarget) * 2000000000
            histogramCorrectedTarget = self.computeHistogramMatchedActivation(self.features2_1[0], self.hist_2, self.min_2, self.max_2)
            assert(histogramCorrectedTarget.shape == self.features2_1.shape)
            loss += torch.nn.functional.mse_loss(self.features2_1, histogramCorrectedTarget) * 500000000
            histogramCorrectedTarget = self.computeHistogramMatchedActivation(self.features3_1[0], self.hist_3, self.min_3, self.max_3)
            assert(histogramCorrectedTarget.shape == self.features3_1.shape)
            loss += torch.nn.functional.mse_loss(self.features3_1, histogramCorrectedTarget) * 100000000
            histogramCorrectedTarget = self.computeHistogramMatchedActivation(self.features4_1[0], self.hist_4, self.min_4, self.max_4)
            assert(histogramCorrectedTarget.shape == self.features4_1.shape)
            loss += torch.nn.functional.mse_loss(self.features4_1, histogramCorrectedTarget) * 35000000
            histogramCorrectedTarget = self.computeHistogramMatchedActivation(self.features5_1[0], self.hist_5, self.min_5, self.max_5)
            assert(histogramCorrectedTarget.shape == self.features5_1.shape)
            loss += torch.nn.functional.mse_loss(self.features5_1, histogramCorrectedTarget) * 35000000

        return loss


    def optimise(self, canvas = None):
        iterations = 1000
        if canvas is None:
            canvas = torch.randn((1, 3, 128, 128)).cuda()
        canvas.requires_grad = True
        optimizer = torch.optim.Adam([canvas], self.learning_rate)
        for i in range(iterations):
            optimizer.zero_grad()
            loss = self.computeLoss(canvas)
            print(self.histogram, canvas.shape[2], i, loss.item())
            loss.backward()
            optimizer.step()
        canvas.clamp(0, 1)
        return canvas.data

model = StyleTransfer().cuda()
print(model)

html = "<html><body><table style='margin:auto;'><tr><th>INPUT</th><th>GRAM ONLY</th><th>GRAM + HISTOGRAM</th></tr>"
for filename in sys.argv[1:]:
    print(filename)

    # Load input
    style = loadImage(filename).cuda()
    if style is not None:
        print("Style", filename, style.shape)
        html += "<tr><td><img src='" + filename + "'></td>"
        for histogram in [False, True]:
            model.histogram = histogram
            # Run input
            model.setStyle(torch.nn.functional.interpolate(style, scale_factor = 1.0/4))
            result = model.optimise()
            result = torch.nn.functional.interpolate(result, scale_factor = 2)
            model.setStyle(torch.nn.functional.interpolate(style, scale_factor = 1.0/2))
            result = model.optimise(result)
            result = torch.nn.functional.interpolate(result, scale_factor = 2)
            model.setStyle(torch.nn.functional.interpolate(style, scale_factor = 1))
            result = model.optimise(result)
            
            path = "dst/mode_" + str(histogram) + "_" + os.path.basename(filename)
            torchvision.utils.save_image(postprocessing(result[0]), path)
            html += "<td><img src='" + path + "'></td>"
            with open("results.html", "w") as f:
                f.write(html + "</table></body></html>")
