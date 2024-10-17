## Project Overview

This project implements three variations of GANs (Generative Adversarial Networks) to generate new images: DCGAN, WGAN, and WGAN-GP. The architectures include both conditional generators and discriminators, which allow controlling the output image class based on the label input. The models are trained on the FashionMNIST dataset and are capable of generating class-conditioned images.

## File Structure

```
/project_root
│
├── DL_A3_Q5_Architecture_1.ipynb  # Contains code for architecture 1
├── DL_A3_Q5_Architecture_2.ipynb  # Contains code for architecture 2
├── generate_new_images.ipynb      # Contains code to generate images from trained models
├── dcgan/                         # Folder for DCGAN model outputs
├── wgan/                          # Folder for WGAN model outputs
└── wgan-gp/                       # Folder for WGAN-GP model outputs
```

## Architectures

| Parameter                 | Architecture 1 (DL_A3_Q5_Architecture_1.ipynb) | Architecture 2 (DL_A3_Q5_Architecture_2.ipynb) |
| ------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| **Nonlinearity (G)**      | ReLU, Tanh                                     | ReLU, Tanh                                     |
| **Nonlinearity (D)**      | LeakyReLU                                      | LeakyReLU                                      |
| **Depth (G)**             | 4 convolutional layers                         | 4 convolutional layers                         |
| **Depth (D)**             | 4 convolutional layers                         | 4 convolutional layers                         |
| **Batch Norm (G)**        | True                                           | True                                           |
| **Batch Norm (D)**        | True (except first layer)                      | True (except first layer)                      |
| **Base Filter Count (G)** | 64                                             | 32                                             |
| **Base Filter Count (D)** | 64                                             | 32                                             |



## How to Train the Models

To train the models, you need to run the cells in the following notebooks:

1. **DCGAN**:
   - Open and run `DL_A3_Q5_Architecture_1.ipynb`.
   - The training process will use the architecture defined in the notebook with the following hyperparameters:
     - Generator optimizer: Adam with learning rate 1e-3, betas = (0.5, 0.999)
     - Discriminator optimizer: Adam with learning rate 1e-3, betas = (0.5, 0.999)
     - Number of epochs: 1
     - `z_test` is randomly initialized with 100 samples of noise for each class label.
2. **WGAN**:
   - Open and run `DL_A3_Q5_Architecture_2.ipynb`.
   - The architecture and hyperparameters are defined in the notebook with the following specifics:
     - Generator optimizer: Adam with learning rate 5e-4, betas = (0.5, 0.999)
     - Discriminator optimizer: Adam with learning rate 5e-4, betas = (0.5, 0.999)
     - Number of epochs: 1
     - Weight clipping set to `clip_value = 0.1`
     - Learning rate scheduling with `StepLR`
3. **WGAN-GP**:
   - Open and run `DL_A3_Q5_Architecture_2.ipynb` (same as WGAN architecture).
   - The hyperparameters are as follows:
     - Generator optimizer: Adam with learning rate 1e-3, betas = (0.5, 0.999)
     - Discriminator optimizer: Adam with learning rate 1e-3, betas = (0.5, 0.999)
     - Number of epochs: 1
     - Critic iterations (`n_critic`) set to 2 to ensure proper training balance between generator and critic.

## How to Generate New Images

Once the models are trained, you can generate new images using the `generate_new_images.ipynb` file. This notebook allows you to load trained models and generate new images based on user input.

### Steps to Generate Images:

1. Ensure the correct checkpoint path is set in the notebook.
2. Specify the desired label (class) and the number of images to generate.
3. Example:

```
Example: Generate 10 images for the label 5 (Sandal class in FashionMNIST)
label = 5  # Specify the class label
num_images = 10  # Number of images to generate
z_dim = g.dim_z  # Latent space dimension

# Generate images using the trained generator
generated_images = generate_images(g, label, num_images, z_dim, device)

# Display real vs generated images
real_images = get_real_images(train_dataset, label, num_images)
show_images(generated_images, real_images, label)
```

## Additional Notes

1. Running the Project on Colab
   - This project was tested and developed using Google Colab.
   - If running the code locally, you will need to adjust the file paths (`output_dir`) to point to the appropriate local checkpoint directories where `.pt` model files are stored.