# CLK-MKR
## Clockmaker: A machine learning toolkit for optimal feature selection and epigenetic clock building 

Clockmaker (CLK-MKR) is a toolkit for biologists and aging scientists to create their own epigenetic clock from their methylation data.
These methods uses contemporary feature selection methods (Feature selection paper 2020) to select the best CpG sites to use as predictors for aging. 

For the pipeline to successfully read the data correctly and complete its process there are a few criteria that should be met before uploading:
- Methylation levels for CpGs must be normalized prior to uploading.
- Within the DNAm dataset there must be a column named ‘Age’ that contains the corresponding age of  all samples.
- There must be no missing values in the data (e.g. NaN).
- Methylation data must be in .CSV format with the only headers being CpG names and the aforementioned ‘Age’. Any other headers and the pipeline will fail.

# Quick Start Guide (More details available within the notebook)
The code in the notebook is the rudimentary base code that outlines the general ideas discussed in the main paper and gives a basic introduction to the feature selection algorithms there. Its highly recommended that you refer to the paper and adjust the algorithms and your code for specific contexts and uses.
To begin using notebook, make sure you have the methylation data of interest and start by running these two cell blocks to get the necessary dependencies and functions

![part 1](https://user-images.githubusercontent.com/25240354/100318143-d8f21400-2fbd-11eb-948e-fc1bd4c0c9de.png)

Next simply run the feature selection function below, where the filepath arguement should be a string and the location of your methylation data

![part 2](https://user-images.githubusercontent.com/25240354/100318477-69c8ef80-2fbe-11eb-93c2-017c5d327daa.png)

This concludes the CpG selection process. It can take a long time, and you should be able to see the processes running in the output.
When the results are printed and it says 'Finished' you will have the results files listed in the notebook available for download.
Thats the bulk of the tool over! Your finished clock and best CpGs are now available for download and study.

### References:

Li, A., Vera, D., Sinclair, D., 2020. Clockmaker (CLK-MKR) : A machine learning toolkit for optimal feature selection and epigenetic clock building. Bioinformatics

Li, A., Vera, D., Sinclair, D., 2020. Constructing more efficient epigenetic clocks and detecting undiscovered predictors of aging through novel feature selection methods. Bioinformatics

Voisin, S., Harvey, N., Haupt, L., Griffiths, L., Ashton, K., Coffey, V., Doering, T., Thompson, J., Benedict, C., Cedernaes, J., Lindholm, M., Craig, J., Rowlands, D., Sharples, A., Horvath, S. and Eynon, N., 2020. An epigenetic clock for human skeletal muscle. Journal of Cachexia, Sarcopenia and Muscle, 11(4), pp.887-898.
