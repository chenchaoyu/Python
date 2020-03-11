import matplotlib.pyplot as plt
from dataset import FaceLandmarksDataset

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated



face_dataset=FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')
fig=plt.figure()
for i in range(len(face_dataset)):
    sample=face_dataset[i]
    print(i,sample['image'].shape,sample['landmarks'].shape)
    ax=plt.subplot(1,4,i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)
    if i==3:
        plt.show()
        break
