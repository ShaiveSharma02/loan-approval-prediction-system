import matplotlib.pyplot as plt


def plot_confusion_matrix(matrix):
    fig, ax = plt.subplots()

    ax.imshow(matrix)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    labels = ["Rejected", "Approved"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center"
            )

    return fig