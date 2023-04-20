#** From: https://www.r-bloggers.com/2021/03/how-to-build-your-own-image-recognition-app-with-r-part-1/
# Directions: 
# Here's the model for Project 2 that you'll need to take a look at, including the data, and a tutorial 
# on how the model was built, and how you can load it into R. The model was validated, so all images are
# correctly identified at over 50% accuracy, and (as mentioned in the project) your job is to  write something
# that convinces the model otherwise that an image of grass is only 49% identified as grass, or the same
# for the dandelion model. You are allowed to know what type the image is before your model applies the changes.

# This tutorial will show you how the model was built, and how to import it into R, along with any dependencies you may need:
# https://www.r-bloggers.com/2021/03/how-to-build-your-own-image-recognition-app-with-r-part-1/

# The images are already classified into the appropriate folders, and you can use the following code after
# you've modified your images to determine if you made the classifier fail (very similar to what's in the
# tutorial, only slightly modified to let you check all the images in the grass or dandelion folders, respectively).

# res=c("","")
# f=list.files("./grass")
# for (i in f){
# test_image <- image_load(paste("./grass/",i,sep=""),
#                                   target_size = target_size)
# x <- image_to_array(test_image)
# x <- array_reshape(x, c(1, dim(x)))
# x <- x/255
# pred <- model %>% predict(x)
# if(pred[1,2]<0.50){
#  print(i)
# }
# }

# To open the tar files, use tar -xf filename in either powershell or mac's terminal. 

# If you have any questions, please let us know on Piazza!
# **From https://www.r-bloggers.com/2021/03/how-to-build-your-own-image-recognition-app-with-r-part-1/
install_tensorflow(extra_packages="pillow")
install_keras()

library(tidyverse)
library(keras)
library(tensorflow)
library(reticulate)

setwd("C:/users/my_name/desktop/birds")
label_list <- dir("train/")
output_n <- length(label_list)
save(label_list, file="label_list.R")

width <- 224
height<- 224
target_size <- c(width, height)
rgb <- 3 #color channels

width <- 224
height<- 224
target_size <- c(width, height)
rgb <- 3 #color channels

train_images <- flow_images_from_directory(path_train,
  train_data_gen,
  subset = 'training',
  target_size = target_size,
  class_mode = "categorical",
  shuffle=F,
  classes = label_list,
  seed = 2021)
  
validation_images <- flow_images_from_directory(path_train,
  train_data_gen, 
  subset = 'validation',
  target_size = target_size,
  class_mode = "categorical",
  classes = label_list,
  seed = 2021)
  
table(train_images$classes)

plot(as.raster(train_images[[1]][[1]][17,,,]))

mod_base <- application_xception(weights = 'imagenet', 
  include_top = FALSE, input_shape = c(width, height, 3))
  freeze_weights(mod_base) 

  model_function <- function(learning_rate = 0.001, 
dropoutrate=0.2, n_dense=1024){

k_clear_session()

model <- keras_model_sequential() %>%
  mod_base %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = n_dense) %>%
  layer_activation("relu") %>%
  layer_dropout(dropoutrate) %>%
  layer_dense(units=output_n, activation="softmax")

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(lr = learning_rate),
  metrics = "accuracy"
)

return(model)

}

model <- model_function()
model

batch_size <- 32
epochs <- 6
hist <- model %>% fit_generator(
  train_images,
  steps_per_epoch = train_images$n %/% batch_size, 
  epochs = epochs, 
  validation_data = validation_images,
  validation_steps = validation_images$n %/% batch_size,
  verbose = 2
)


path_test <- "/test/"
test_data_gen <- image_data_generator(rescale = 1/255)
test_images <- flow_images_from_directory(path_test,
   test_data_gen,
   target_size = target_size,
   class_mode = "categorical",
   classes = label_list,
   shuffle = F,
   seed = 2021)
model %>% evaluate_generator(test_images, 
                     steps = test_images$n)
                     

test_image <- image_load("Test images/Bald Eagle/index.jpg",
                                  target_size = target_size)
x <- image_to_array(test_image)
x <- array_reshape(x, c(1, dim(x)))
x <- x/255
pred <- model %>% predict(x)
pred <- data.frame("Bird" = label_list, "Probability" = t(pred))
pred <- pred[order(pred$Probability, decreasing=T),][1:5,]
pred$Probability <- paste(format(100*pred$Probability,2),"%")
pred


predictions <- model %>% 
  predict_generator(
    generator = test_generator,
    steps = test_generator$n
  ) %>% as.data.frame
names(predictions) <- paste0("Class",0:39)
predictions$predicted_class <- 
  paste0("Class",apply(predictions,1,which.max)-1)
predictions$true_class <- paste0("Class",test_generator$classes)
predictions %>% group_by(true_class) %>% 
  summarise(percentage_true = 100*sum(predicted_class == 
    true_class)/n()) %>% 
    left_join(data.frame(bird= names(test_generator$class_indices), 
    true_class=paste0("Class",0:39)),by="true_class") %>%
  select(bird, percentage_true) %>% 
  mutate(bird = fct_reorder(bird,percentage_true)) %>%
  ggplot(aes(x=bird,y=percentage_true,fill=percentage_true, 
    label=percentage_true)) +
  geom_col() + theme_minimal() + coord_flip() +
  geom_text(nudge_y = 3) + 
  ggtitle("Percentage correct classifications by bird species")
  
  
  tune_grid <- data.frame("learning_rate" = c(0.001,0.0001),
                        "dropoutrate" = c(0.3,0.2),
                        "n_dense" = c(1024,256))
tuning_results <- NULL
set.seed(2021)
for (i in 1:length(tune_grid$learning_rate)){
  for (j in 1:length(tune_grid$dropoutrate)){
      for (k in 1:length(tune_grid$n_dense)){
        
        model <- model_function(
          learning_rate = tune_grid$learning_rate[i],
          dropoutrate = tune_grid$dropoutrate[j],
          n_dense = tune_grid$n_dense[k])
        
        hist <- model %>% fit_generator(
          train_images,
          steps_per_epoch = train_images$n %/% batch_size, 
          epochs = epochs, 
          validation_data = validation_images,
          validation_steps = validation_images$n %/% 
            batch_size,
          verbose = 2
        )
        
        #Save model configurations
        tuning_results <- rbind(
          tuning_results,
          c("learning_rate" = tune_grid$learning_rate[i],
            "dropoutrate" = tune_grid$dropoutrate[j],
            "n_dense" = tune_grid$n_dense[k],
            "val_accuracy" = hist$metrics$val_accuracy))
        
    }
  }
}
tuning_results


best_results <- tuning_results[which( 
  tuning_results[,ncol(tuning_results)] == 
  max(tuning_results[,ncol(tuning_results)])),]
  
  
model <- model_function(learning_rate = 
  best_results["learning_rate"],
  dropoutrate = best_results["dropoutrate"],
  n_dense = best_results["n_dense"])
hist <- model %>% fit_generator(
  train_images,
  steps_per_epoch = train_samples %/% batch_size, 
  epochs = epochs, 
  validation_data = validation_images,
  validation_steps = valid_samples %/% batch_size,
  verbose = 2
)
model %>% save_model_tf("bird_mod")
