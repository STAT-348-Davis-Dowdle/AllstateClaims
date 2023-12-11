library(doParallel)

cl <- makePSOCKcluster(7)
registerDoParallel(cl)

library(vroom)
library(tidymodels)
library(embed)
library(glmnet)

setwd("C:/Users/davis/OneDrive - Brigham Young University/Documents/skool/new/stat 348/AllstateClaims/AllstateClaims")

set.seed(1)
train <- vroom("train.csv")
train <- train[sort(sample(1:nrow(train), 5000, replace = FALSE)),]
test <- vroom("test.csv")

recipe <- recipe(loss ~ ., data = train) %>%
  step_rm(cat15, cat20, cat21, cat22, cat31, cat34, cat35, cat39, cat46, cat47) %>%
  step_rm(cat48, cat55, cat56, cat58, cat59, cat60, cat61, cat62, cat63, cat64) %>%
  step_rm(cat67, cat68, cat69, cat70, cat74, cat77, cat78, cat93, cat115) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>%
  step_normalize(all_numeric_predictors())

model <- linear_reg(penalty = tune(),
                    mixture = tune()) %>%
  set_engine("glmnet")

wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(model)

grid <- grid_regular(penalty(),
                     mixture(),
                     levels = 5)

folds <- vfold_cv(train, v = 5, repeats = 1)

results <- wf %>%
  tune_grid(grid = grid,
            resamples = folds)

best <- results %>%
  select_best("rmse")

final_wf <- wf %>%
  finalize_workflow(best) %>%
  fit(data = train)

predictions <- final_wf %>%
  predict(new_data = test)

submission <- data.frame(id = as.numeric(test$id), loss = predictions$.pred)

vroom_write(submission, "Penalized.csv", delim = ",")

stopCluster(cl)
