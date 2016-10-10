#Visa Premier Classification: A Marketing Application

###############################################################################
#                               Description
###############################################################################

From a dataset with the customer's behavior of a bank, four different 
Classification Models were implemented, to predict the probability for a new 
customer to buy a Visa Premier Card. Three of models use only continuous 
variables and one uses heterogenous data (Continuous and Categorial variables).

###############################################################################
#                               Requirements
###############################################################################
- Implementation in R 3.1.2

- Libraries:
    MASS
    mclust
    Rmixmod
    Caret
    pROC

###############################################################################
#                               Files Information
###############################################################################
This repository contains the following three files:

  1. VisaPremier.txt: This dataset provides information about bank costumers. It
     specifies, for instance, their movements, account balances, personal 
     information and whether they have the Visa Premier card or not. This is a
     premium payment card that seeks to strengthen the close relationship with 
     the bank to retain wealthy clients. This dataset describes the behavior 
     of 1073 clients by using 48 variables.
     
  2. VisaPremier_Classification.R: This R script has the following procedures:
    - Data Cleaning: Remove constant or almost constant variables. Split 
                     Continuous and categorical variables.
    - Data Preprocessing: Missing values for the continuous variables are 
                          replaced by the mean and for categorical variables are
                          replaced by the mode. Define correctly the type of the
                          variable.
    - Modeling: Three models with only continuous data: Linear Discriminant
                Analysis, Quadratic Discriminant Analysis, Support Vector 
                Machine. One model with heterogenous data: Mixture Model.

    - Prediction results and comparison.

  3. visa-premier-classification.pdf: Further Description of the tasks, goals,
                                      methodologies, explanation, results & 
                                      conclusions.
