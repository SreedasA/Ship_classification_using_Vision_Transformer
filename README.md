# Ship_classification_using_Vision_Transformer
In this project, our primary goal was to achieve high accuracy in automatically categorizing ship images into predefined classes. Leveraging Vision Transformers (ViTs), originally designed for natural language processing (NLP), we adapted them for computer vision tasks. Our focus was on improving accuracy while maintaining computational efficiency.

GPU Specifications
CUDNN VERSION: 8700
Number of CUDA Devices: 1
CUDA Device Name: NVIDIA GeForce RTX 3050 Laptop GPU
CUDA Device Total Memory: 4.29 GB

Key Achievements:
Created a robust model using Vision Transformers.
Improved the confusion matrix significantly.

Key Metrics:
Accuracy: 98.25%
F1 Score: 0.9826

Impact:
Achieving an accuracy of 98.25% represents a drastic improvement in ship classification. Our work has practical applications in maritime surveillance, environmental monitoring, and navigation assistance.

Challenges:
Limited Annotated Data: Obtaining a diverse and well-annotated ship dataset was challenging.
Memory Efficiency: Balancing accuracy with memory requirements, especially for large ViT models.

Future Directions:
Investigate ensemble techniques (e.g., combining ViTs with CNNs) to further boost accuracy.
Explore transfer learning from related domains (e.g., aerial imagery, underwater scenes) for enhanced generalization.
