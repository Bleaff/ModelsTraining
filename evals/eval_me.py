import torch
from model_learn.utils.datasets.multilabel.dataset import DS
from model_learn.my_architecture.multilabel import MobileNetModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score
import seaborn as sns
import numpy as np
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='This script evaluates the performance of a pre-trained PyTorch image classification model on a test dataset. It computes standard metrics such as accuracy, precision, recall, and F1 score. Additionally, it visualizes confidence distributions, Receiver Operating Characteristic (ROC) curves, and Precision-Recall curves. Sample images from the test set are also displayed with their true and predicted labels.')
    parser.add_argument('valid_data_root', type=str, help='Root directory path of images')
    parser.add_argument('valid_data_yaml', type=str, help='Path to file contains yaml with labels')
    parser.add_argument('model_weights', type=str, help='Path to file contains weights.pth for model')
    args = parser.parse_args()
    return args.valid_data_root, args.valid_data_yaml, args.model_weights

if __name__ == '__main__':
    # parse arguments 
    valid_data_root, valid_data_yaml, model_weights = parse_arguments()

    # Load your model
    model = MobileNetModel() # Your model initialization here
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    test_dataset = DS(valid_data_root, valid_data_yaml, 224)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Predict and evaluate
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for idx, (images, labels, paths) in enumerate(tqdm(test_loader)):
            labels = torch.tensor([labels.flatten()[1]])
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            cases, incass = outputs#[item.flatten() for item in outputs]
            _, predicted = torch.max(incass, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Evaluation Metrics
    print(classification_report(all_targets, all_preds))
    f1 = f1_score(all_targets, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_targets, all_preds, multi_class='ovr')

    # Precision-Recall and ROC-AUC Curves
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    roc_auc_val = auc(fpr, tpr)

    # Plotting
    sns.histplot(all_preds, kde=True, stat="density", linewidth=0)
    plt.savefig('confidence_distribution.png')
    plt.close()  # Close the plot to free up memory

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_val)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('precision_recall_curve.png')
    plt.close()

    # Display some images with predictions
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    # Get a batch of test data
    inputs, classes = next(iter(test_loader))
    # Make predictions
    outputs = model(inputs.to(device))
    _, preds = torch.max(outputs, 1)

    # Plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        imshow(inputs.cpu().data[idx])
        ax.set_title("{} ({})".format(test_dataset.classes[preds[idx]], test_dataset.classes[classes[idx]]), 
                     color=("green" if preds[idx]==classes[idx].item() else "red"))
    plt.savefig('sample_predictions.png')
    plt.close()
