# Load the model
model <- load_model_tf("/home/jupyter/332_data/dandelion_model")

# Define the binary labels
grass_label <- [1,2]
dandelion_label <- [1,1]

# Define the max percent of pixels that can be altered
max_pixel <- 0.01

# Load an image of the grass
x <- image_load("/home/jupyter/332_data/data-for-332/grass/",i,sep="")

# Perform the boundary attack for grass
for (i in 1:100) {
  # Find the gradient of the loss
  grad <- keras::gradients(binary_crossentropy(grass_label, model %>% predict(x/255)), x)
  
  # Compute the step to perturb the image
  step <- max_pixels * sign(grad)
  
  # Perturb the image
  x <- x + step
  
  # Change the image values
  x <- pmax(pmin(x, 1), 0)
  
  # Check if the classifier has been tricked
  if (model %>% predict_classes(x/255) == grass_label) break
}

# Load an image of the dandelions
x <- image_load("/home/jupyter/332_data/data-for-332/grass/",i,sep="")

# Perform the boundary attack for dandelions
for (i in 1:100) {
  # Find the gradient of the loss
  grad <- keras::gradients(binary_crossentropy(dandelion_labels, model %>% predict(x/255)), x)
  
  # Compute the step to perturb the image
  step <- max_pixels * sign(grad)
  
  # Perturb the image
  x <- x + step
  
  # Change the image value
  x <- pmax(pmin(x, 1), 0)
  
  # # Check if the classifier has been tricked
  if (model %>% predict_classes(x/255) == dandelion_label) break
}


