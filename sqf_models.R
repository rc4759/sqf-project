# Rachel Chen

# load libraries
library(tidyverse)
library(ROCR)

# 1. read data
sqf_08_16 <- read_csv('sqf_08_16.csv')

# restrict to 'cpw' stops, remove stops in precinct 121, and select specified variables
sqf <- sqf_08_16 %>% 
  filter(suspected.crime == 'cpw', precinct != 121) %>% 
  select(year, found.weapon, precinct, location.housing, additional.report, additional.investigation,
         additional.proximity, additional.evasive, additional.associating, additional.direction,
         additional.highcrime, additional.time, additional.sights, additional.other,
         stopped.bc.object, stopped.bc.desc, stopped.bc.casing, stopped.bc.lookout, 
         stopped.bc.clothing, stopped.bc.drugs, stopped.bc.furtive, stopped.bc.violent,
         stopped.bc.bulge, stopped.bc.other, suspect.age, suspect.build, suspect.sex,
         suspect.height, suspect.weight, inside, radio.run, observation.period, day, month,
         time.period)

# restrict to complete cases
sqf <- sqf %>% filter(complete.cases(sqf))

# 2. train logistic regression model 

# convert to factor variables as needed
sqf$precinct <- as.factor(sqf$precinct)
sqf$location.housing <- as.factor(sqf$location.housing)
sqf$suspect.build <- as.factor(sqf$suspect.build)
sqf$suspect.sex <- as.factor(sqf$suspect.sex)
sqf$day <- as.factor(sqf$day)
sqf$month <- as.factor(sqf$month)
sqf$time.period <- as.factor(sqf$time.period)

# create training set of stops in 2008
train <- sqf %>% 
  filter(year == 2008) %>% 
  select(-year)

# standardize real-valued attributes
standardize <- function(x) {
  x.std <- (x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE)
  x.std
}

train$suspect.age <- standardize(train$suspect.age)
train$suspect.height <- standardize(train$suspect.height)
train$suspect.weight <- standardize(train$suspect.weight)
train$observation.period <- standardize(train$observation.period)

# fit logistic regression model predicting whether a weapon is found
glm_weapon <- glm(found.weapon ~ ., data = train, family = 'binomial')
summary(glm_weapon)

# 3. compute probability person was carrying a weapon based on given details

# reset training data
train <- sqf %>% 
  filter(year == 2008) %>% 
  select(-year)

# re-standardize attributes based on given information
std.suspect.age <- (30 - mean(train$suspect.age))/(sd(train$suspect.age))
std.suspect.height <- (6 - mean(train$suspect.height))/(sd(train$suspect.height))
std.suspect.weight <- (165 - mean(train$suspect.weight))/(sd(train$suspect.weight))
std.observation.period <- (10 - mean(train$observation.period))/(sd(train$observation.period))

# compute probability based on given information
predict(glm_weapon, newdata = data.frame(precinct = as.factor(6), location.housing = 'transit', additional.report = F,
                                         additional.investigation = F, additional.proximity = F,
                                         additional.evasive = F, additional.associating = F, 
                                         additional.direction = F, additional.highcrime = T, additional.time = F,
                                         additional.sights = F, additional.other = F, stopped.bc.object = F,
                                         stopped.bc.desc = F, stopped.bc.casing = F, stopped.bc.lookout = F,
                                         stopped.bc.clothing = F, stopped.bc.drugs = F, stopped.bc.furtive = F,
                                         stopped.bc.violent = F, stopped.bc.bulge = T, stopped.bc.other = F,
                                         suspect.age = std.suspect.age, suspect.build = 'medium', suspect.sex = 'male', 
                                         suspect.height = std.suspect.height, suspect.weight = std.suspect.weight,
                                         inside = T, radio.run = F, observation.period = std.observation.period, 
                                         day = 'Saturday', month = 'October', time.period = as.factor(6)), type = 'response')

# re-compute probability given the person is a woman, everything else equal
predict(glm_weapon, newdata = data.frame(precinct = as.factor(6), location.housing = 'transit', additional.report = F,
                                         additional.investigation = F, additional.proximity = F,
                                         additional.evasive = F, additional.associating = F, 
                                         additional.direction = F, additional.highcrime = T, additional.time = F,
                                         additional.sights = F, additional.other = F, stopped.bc.object = F,
                                         stopped.bc.desc = F, stopped.bc.casing = F, stopped.bc.lookout = F,
                                         stopped.bc.clothing = F, stopped.bc.drugs = F, stopped.bc.furtive = F,
                                         stopped.bc.violent = F, stopped.bc.bulge = T, stopped.bc.other = F,
                                         suspect.age = std.suspect.age, suspect.build = 'medium', suspect.sex = 'female', 
                                         suspect.height = std.suspect.height, suspect.weight = std.suspect.weight,
                                         inside = T, radio.run = F, observation.period = std.observation.period, 
                                         day = 'Saturday', month = 'October', time.period = as.factor(6)), type = 'response')


# 4. compute AUC on data from 2009
sqf_auc <- sqf %>% 
  filter(year == 2009) %>% 
  select(-year)

# standardize real-valued attributes
sqf_auc$suspect.age <- standardize(sqf_auc$suspect.age)
sqf_auc$suspect.height <- standardize(sqf_auc$suspect.height)
sqf_auc$suspect.weight <- standardize(sqf_auc$suspect.weight)
sqf_auc$observation.period <- standardize(sqf_auc$observation.period)

# find predicted probabilities 
sqf_auc <- sqf_auc %>%
  mutate(predicted.probability = predict(glm_weapon, sqf_auc, type='response'))

# compute AUC
auc.pred <- prediction(sqf_auc$predicted.probability, sqf_auc$found.weapon)
auc.perf <- performance(auc.pred, "auc")
auc.perf@y.values[[1]]

#------------------------------------------------------------------------------#

# 1. choose target variable
# target variable: frisked for whether a suspect is frisked or not

# 2. select set of predictor variables and filter to complete cases
sqf_frisk <- sqf_08_16 %>% 
  select(year, frisked, inside, observation.period, location.housing, suspect.sex,
         suspect.race, suspect.age, suspect.height, suspect.weight, suspect.hair,
         stopped.bc.object, stopped.bc.desc, stopped.bc.casing, stopped.bc.lookout,
         stopped.bc.clothing, stopped.bc.drugs, stopped.bc.furtive, stopped.bc.violent,
         stopped.bc.bulge, stopped.bc.other) %>% 
  filter(year == 2008:2012)

sqf_frisk <- sqf_frisk %>% filter(complete.cases(sqf_frisk))

# 3. create train-test split of data

# training set for data from 2008 to 2011
train_frisk <- sqf_frisk %>% 
  filter(year == 2008:2011) %>% 
  select(-year)
  
# testing set for data in 2012
test_frisk <- sqf_frisk %>% 
  filter(year == 2012) %>% 
  select(-year)

# standardize real-valued attributes
train_frisk$suspect.age <- standardize(train_frisk$suspect.age)
train_frisk$suspect.height <- standardize(train_frisk$suspect.height)
train_frisk$suspect.weight <- standardize(train_frisk$suspect.weight)
train_frisk$observation.period <- standardize(train_frisk$observation.period)

test_frisk$suspect.age <- standardize(test_frisk$suspect.age)
test_frisk$suspect.height <- standardize(test_frisk$suspect.height)
test_frisk$suspect.weight <- standardize(test_frisk$suspect.weight)
test_frisk$observation.period <- standardize(test_frisk$observation.period)

# 4. select classification method, fit data, make predictions 

# fit logistic regression model for whether an individual is frisked or not
glm_frisk <- glm(frisked ~ ., data = train_frisk, family = 'binomial')
summary(glm_frisk)

# make predictions on test set
test_frisk <- test_frisk %>% 
  mutate(predicted.probability = predict(glm_frisk, test_frisk, type = 'response'))
  
# 5. generate recall-at-k% plot 

# create data for plot
plot_data <- test_frisk %>%
  arrange(desc(predicted.probability)) %>%
  mutate(numstops = row_number(),
         percent.outcome = cumsum(frisked)/sum(frisked),
         stops = numstops/n()) %>%
  select(stops, percent.outcome)

# make recall-at-k% plot
p <- ggplot(data = plot_data, aes(x = stops, y = percent.outcome)) +
  geom_line() + scale_x_log10('\nPercent of Stops',
                              limits = c(0.001, 1),
                              breaks = c(.003,.01,.03,.1,.3,1),
                              labels = c('0.3%','1%','3%','10%','30%','100%')) +
  scale_y_continuous("Percent of Frisked", limits = c(0, 1), labels = scales::percent)

# save plot
ggsave(plot = p, file ='figures/recall_at_k.png')





