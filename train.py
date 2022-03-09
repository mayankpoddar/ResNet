import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from DatasetFetcher import DatasetFetcher
from project1_model import ResNet, BasicBlock 

if __name__ == "__main__":
    # Check available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fetching Dataset
    df = DatasetFetcher(dataset="CIFAR10", batch_size=64)
    df.addHorizontalFlipping()
    df.addRandomCrop(size=32, padding=3)
    df.addNormalizer()
    trainLoader, testLoader = df.getLoaders()
    
    # Get Model
    model = ResNet(BasicBlock, 3, [2, 2, 2])
    model = model.to(device)
    
    # Defining Loss Function, Learning Rate, Weight Decay, Optimizer) 
    lossFunction = torch.nn.CrossEntropyLoss()
    learningRate = 0.001
    weightDecay = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
    print(model.eval())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Trainable Parameters : %s"%(trainable_parameters))
    if trainable_parameters > 5*(10**6):
        raise Exception("Model not under budget!")
    
    # Setting up training
    EPOCHS=100
    globalBestAccuracy = 0.0
    trainingLoss = []
    testingLoss = []
    trainingAccuracy = []
    testingAccuracy = []
    
    # Training
    for i in tqdm(range(EPOCHS)):
        for phase in ['train', 'test']:
            if phase == "train":
                loader = trainLoader
                model.train()
                optimizer.zero_grad()
            else:
                loader = testLoader
                model.eval()
            runningLoss = 0.0
            runningCorrects = 0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = lossFunction(output, labels)
                predicted_labels = torch.argmax(output, dim=1)
                runningLoss = loss.item()*images.size(0)
                runningCorrects += torch.sum(predicted_labels == labels).float().item()
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                break
            epochLoss = runningLoss/len(loader.dataset)
            epochAccuracy = runningCorrects/len(loader.dataset)
            if phase == "train":
                trainingLoss.append(epochLoss)
                trainingAccuracy.append(epochAccuracy)
            else:
                testingLoss.append(epochLoss)
                testingAccuracy.append(epochAccuracy)
                if epochAccuracy > globalBestAccuracy:
                    globalBestAccuracy = epochAccuracy
                    model.saveToDisk()	
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    n = len(trainingLoss)
    ax1.plot(range(n), trainingLoss, '-', linewidth='3', label='Train Error')
    ax1.plot(range(n), testingLoss, '-', linewidth='3', label='Test Error')
    ax2.plot(range(n), trainingAccuracy, '-', linewidth='3', label='Train Accuracy')
    ax2.plot(range(n), testingAccuracy, '-', linewidth='3', label='Test Acuracy')
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    ax2.legend()
    f.savefig("./trainTestCurve.png")
