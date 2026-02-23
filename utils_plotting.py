import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(metrics, task, save_dir = None):

    if task == "classification":

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax1.set_title(f"{task} / Loss")
        ax1.plot(metrics["train_loss"], label='train')
        ax1.plot(metrics["val_loss"], label='validation')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_box_aspect(1)
        ax1.legend()
    
        ax2 = fig.add_subplot(122)
        ax2.set_title(f"{task} / Accuracy")
        ax2.plot(metrics["train_acc"], label='train')
        ax2.plot(metrics["val_acc"], label='validation')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_box_aspect(1)
        ax2.legend()

        # FIFRST SAVE THEN SHOW!!!!
        if save_dir is not None:
            fig.savefig(save_dir / "metric_curves.png", dpi=200, bbox_inches="tight")
        plt.show()


    elif task == "segmentation":
    
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)
        ax1.set_title(f"{task} / Loss")
        ax1.plot(metrics["train_loss"], label='train')
        ax1.plot(metrics["val_loss"], label='validation')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_box_aspect(1)
        ax1.legend()
    
        ax2 = fig.add_subplot(132)
        ax2.set_title(f"{task} / Accuracy")
        ax2.plot(metrics["train_acc"], label='train')
        ax2.plot(metrics["val_acc"], label='validation')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_box_aspect(1)
        ax2.legend()

        ax3 = fig.add_subplot(133)
        ax3.set_title(f"{task} / Mean IoU")
        ax3.plot(metrics["train_miou"], label='train')
        ax3.plot(metrics["val_miou"], label='validation')
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Mean IoU")
        ax3.set_box_aspect(1)
        ax3.legend()

        # FIFRST SAVE THEN SHOW!!!!
        if save_dir is not None:
            fig.savefig(save_dir / "metric_curves.png", dpi=200, bbox_inches="tight")
        plt.show()

    else:  raise ValueError(f"Unknown task: {task}")











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



