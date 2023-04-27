library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)
install_tensorflow(extra_packages="pillow")
install_keras()

# Load the original model
model <- load_model_tf("/home/jupyter/332_data/dandelion_model")

# Define size of images and color channels
width <- 224
height <- 224
target_size <- c(width, height)
rgb <- 3

# Define epsilon value
epsilon <- 0.01

# Define function for FGSM attack
fgsm_attack <- function(image, label, epsilon, model) {
  # Convert the image to an array
  x <- image_to_array(image)
  # Reshape the array to match the input shape of the model
  x <- array_reshape(x, c(1, dim(x)))
  # Scale the pixel values to the range [0, 1]
  x <- x / 255
  # Compute the loss and gradients for the input images and labels
  label_dandelion <- c(1, 1)
  label_grass <- c(1, 2)
  loss_dandelion <- function(x) {
    k_categorical_crossentropy(model(x), label_dandelion)
  }
  loss_grass <- function(x) {
    k_categorical_crossentropy(model(x), label_grass)
  }
  grad_dandelion <- gradient(loss_dandelion, x)
  grad_grass <- gradient(loss_grass, x)
  # Compute the sign of the gradients
  sign_grad_dandelion <- sign(grad_dandelion)
  sign_grad_grass <- sign(grad_grass)
  # Compute the perturbation as the epsilon times the sign of the gradients
  if (model(x)[[1]]$class_name == "dandelion") {
    perturbation <- epsilon * sign_grad_grass
  } 
  else {
    perturbation <- epsilon * sign_grad_dandelion
  }
  # Add the perturbation to the input image
  x_adv <- x + perturbation
  # Make the pixel values fit in the range [0, 1]
  x_adv <- pmax(pmin(x_adv, 1), 0)
  # Scale the pixel values back to the range [0, 255]
  x_adv <- x_adv * 255
  # Convert the array back to an image
  adv_image <- array_to_image(x_adv, dim(x_adv)[-1])
  return(adv_image)
}

# Define function to modify only 1% of the pixels in the image
mod_image <- function(image, percent) {
  # Calculate the number of pixels in the image
  n_pixels <- prod(dim(I_adv)[-1])
  # Calculate the number of pixels to modify
  n_mod_pixels <- round(n_pixels * percent / 100)
  # Choose a random subset of pixels to modify
  mod_pixels <- sample(n_pixels, n_mod_pixels, replace = FALSE)
  # Modify the selected pixels by adding a small random value
  I_adv[[1]][mod_pixels] <- I_adv[[1]][mod_pixels] + runif(n_mod_pixels, -10, 10)

  # Return the modified adversarial image
  return(I_adv)
}
