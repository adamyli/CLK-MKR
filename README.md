# CLK-MKR
## Clockmaker: A machine learning toolkit for optimal feature selection and epigenetic clock building 

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

Next simply run the feature selection function below, where the filepath arguement should be a string and the location of your methylation data

![part 2](https://user-images.githubusercontent.com/25240354/100318477-69c8ef80-2fbe-11eb-93c2-017c5d327daa.png)

This concludes the CpG selection and Clock building process. It can take a long time, and you should be able to see the processes running in the output.
When the results are printed and it says 'Finished' you will have the 5 files listed in the notebook available for download.
Thats the bulk of the tool over! Your finished clock and best CpGs are now available for download and study.

The final 'Results' section is optional and only if you want to make use of the two extra files 'labelled_best_cpgs.csv' and 'age_graph.csv'. The following code will plot the graphs described in the notebook along with the age group clusters

![part 3](https://user-images.githubusercontent.com/25240354/100319204-89ace300-2fbf-11eb-97e4-5c7942cc567d.png)
