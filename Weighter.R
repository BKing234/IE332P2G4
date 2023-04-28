adversarial <- function(image, budget) {
    # Define weights for each adversarial image modifier
    weights <- c(0.2, 0.3, 0.1, 0.2, 0.2)    
    # Test the image to see what type it is originally classified as
    test_image <- image_load(image, target_size = target_size)
    x <- image_to_array(test_image)
    x <- array_reshape(x, c(1, dim(x)))
    x <- x/255
    pred <- model %>% predict(x)
    
    label <- matrix(0, nrow = 1, ncol = 2) # identifies whether the image is dandelion [1,0] or grass [0,1]
    # initializes label based on model outcome
    if(pred[1,2]>pred[1,1]){
        label[1,2] <- 1
    } else {
        label[1,1] <- 1
    }
    
    # Initialize variables to keep track of total pixel budget used and modified image
    used_pixels <- 0
    modified_image <- image_load(image, target_size = target_size) #uses already loaded form of image, saves processing
    modified_image <- image_to_array(modified_image)
    modified_image <- array_reshape(modified_image, c(dim(modified_image)))
    
    # Loop through each adversarial image modifier and modify the image
    for (i in 1:5) {
        # Calculate the expected number of pixels to be modified by the current algorithm
        expected_budget <- budget * nrow(modified_image) * ncol(modified_image)
        expected_pixels <- round(expected_budget * weights[i])
        
        # Check if there are enough pixels left in the budget to use this algorithm
        if (used_pixels + expected_pixels <= budget * nrow(modified_image) * ncol(modified_image)) {
            # Apply the adversarial image modifier to the image
            modified_image <- apply_modifier(modified_image, i, label)
            
            # Update the number of used pixels and the weights of the other algorithms
            weights <- c(0.2, 0.3, 0.1, 0.2, 0.2)
            used_pixels <- used_pixels + expected_pixels
            weights[-i] <- weights[-i] / sum(weights[-i])
        } else {
            # Stop modifying the image if the budget is exceeded
            break
        }
    }

    return(modified_image)
}

# Helper function to apply a specific adversarial image modifier to an image
apply_modifier <- function(image, algorithm, label) {
    # Call the appropriate adversarial image modifier function based on the input algorithm
    if (algorithm == 1) {
        return(FastGradSignMethod(image, budget, label))
    } else if (algorithm == 2) {
        return(Pixel(image, budget, label))
    } else if (algorithm == 3) {
        return(Spatial(image, budget, label))
    } else if (algorithm == 4) {
        return(Blurring(image, budget, label))
    } else {
        return(Noise(image, budget, label))
    }
}
