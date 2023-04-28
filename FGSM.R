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
  # Reshape the array
  x <- array_reshape(x, c(1, dim(x)))
  # Scale the pixel values
  x <- x / 255
  # Compute the loss and gradients
  label_dandelion <- c(1, 1)
  label_grass <- c(1, 2)
  loss_dandelion <- function(x) {
    categorical_crossentropy(model(x), label_dandelion)
  }
  loss_grass <- function(x) {
    categorical_crossentropy(model(x), label_grass)
  }
  gradient_dandelion <- gradient(loss_dandelion, x)
  gradient_grass <- gradient(loss_grass, x)
  # Compute the sign of the gradients
  sign_gradient_dandelion <- sign(gradient_dandelion)
  sign_gradient_grass <- sign(gradient_grass)
  # Compute the perturbation
  if (model(x)[[1]]$class_name == "dandelion") {
    perturbation <- epsilon * sign_gradient_grass
  } 
  else {
    perturbation <- epsilon * sign_gradient_dandelion
  }
  # Add the perturbation to the input image
  x_adv <- x + perturbation
  # Fit pizels in the range [0, 1]
  x_adv <- pmax(pmin(x_adv, 1), 0)
  # Rescale the pixels to [0, 255]
  x_adv <- x_adv * 255
  # Change the array back to an image
  adv_image <- array_to_image(x_adv, dim(x_adv)[-1])
  return(adv_image)
}

# Modify only 1% of the pixels in the image
mod_image <- function(image, percent) {
  # Calculate the number of pixels
  num_pixels <- prod(dim(I_adv)[-1])
  # Calculate the number of pixels to modify
  num_modified_pixels <- round(num_pixels * percent / 100)
  # Choose random pixels to modify
  modified_pixels <- sample(num_pixels, num_modified_pixels, replace = FALSE)
  # Modify the selected pixels
  I_adv[[1]][modified_pixels] <- I_adv[[1]][modified_pixels] + runif(num_modified_pixels, -10, 10)

  # Return the modified adversarial image
  return(I_adv)
}
