# CLK-MKR
Machine learning toolkit for optimal feature selection and epigenetic clock building 

Clockmaker (CLK-MKR) is a toolkit for biologists and aging scientists to create their own epigenetic clock from their methylation data.
CLK-MKR uses contemporary feature selection methods (Feature selection paper 2020) to select the best CpG sites to use as predictors for aging. 
It then builds a clock tuned with optimal hyperparameters that researchers can download to use to predict the epigenetic age of any future samples.
As an added bonus some visualization and clustering of results are also provided.

It is highly recommended that you use the Python Notebook over the shiny .R/.py web application as it is much faster and more computationally efficient. Similar plots are also available using the notebook. However the webapp is available for those who wish to use the interactive plots. 
For more information on the feature selection methods, step-by-step guide for the shiny app and more please refer to (Webapp paper 2020)

For the CLK-MKR pipeline to successfully read the data correctly and complete its process there are a few criteria that should be met before uploading:
- Methylation levels for CpGs must be normalized prior to uploading.
- Within the DNAm dataset there must be a column named ‘Age’ that contains the corresponding age of  all samples.
- There must be no missing values in the data (e.g. NaN).
- Methylation data must be in .CSV format with the only headers being CpG names and the aforementioned ‘Age’. Any other headers and the pipeline will fail.

# Quick Start Guide (More details available within the notebook)
To begin using CLK-MKR Notebook, make sure you have the methylation data of interest and start by running these two cell blocks to get the necessary dependencies and functions

![part 1](https://user-images.githubusercontent.com/25240354/100318143-d8f21400-2fbd-11eb-948e-fc1bd4c0c9de.png)
