import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(metrics, metric, save_dir = None):

    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]

    if metric == "acc":
        title = "Accuracy"
        train_metric = metrics["train_acc"]
        val_metric = metrics["val_acc"]
        fig_name = "loss_accuracy_curves.png"
    elif metric == "miou":
        title = "Mean IoU"        
        train_metric = metrics["train_miou"]
        val_metric = metrics["val_miou"]
        fig_name = "loss_miou_curves.png"
    else: raise TypeError(f"{metric} doesn't exist!")

    
    # PLOT TRAINING AND ACCURACY CURVES
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2,1,1)
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title('Loss')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(train_metric, label='train')
    plt.plot(val_metric, label='validation')
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.legend()
    # FIFRST SAVE THEN SHOW!!!!
    if save_dir is not None:
        plt.savefig(save_dir / fig_name, dpi=200, bbox_inches="tight")
    plt.show()









def plot_object_parts(dataset, sample, id_to_name):
    
    points, object_class, seg_labels, _ = dataset[sample]

    num_parts = len(id_to_name)
    name_parts = [id_to_name[i] for i in range(num_parts)]
    parts_present = len(np.unique(seg_labels))       # equivalent to num_seg_labels but safer....
    # MAPPING PART COLORS
    seg_class_color = ["red", "blue", "green", "black", "orange"]
    points_color = [seg_class_color[i] for i in seg_labels]

    points = points.T

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(points[:,0], points[:,1], points[:,2], s=2)
    ax1.set_title(f"Sample class: {object_class} / Num present parts: {parts_present}/{num_parts}")
    ax1.set_aspect('equal')
  
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(points[:,0], points[:,1], points[:,2], c=points_color)
    ax2.set_title(f"Sample parts colored\n{seg_class_color[0:num_parts]}\n{name_parts}")
    ax2.set_aspect('equal')
    plt.show()



