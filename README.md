# ImmunoSEIRS

Contains:
- BA12 for US BA12 scenario projections
- R14 for round 14 of the SMH
- Scripts to generate data (vaccination, cases, hospitalizations, deaths, school and work closures calendar, variant proportions,...)  

##CODE IMPLEMENTATION (FOR BA12 CASE)::

###Prepare:
scripts to prepare vaccination-
https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-Jurisdi/unsk-b7fc
    DONE
scripts to process cases, hospital admission, deaths
https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv
https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv
https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh

DONE
scripts to get work/school data
DONE
scripts with demographics, states,
    DONE
scripts for variants fitting,
https://covid.cdc.gov/covid-data-tracker/#variant-proportions
    DONE

things to do to prepare data:
update the date files: date-fit.xlsx and date-vacc.xlsx
download updated data for hospitalizations, cases, deaths, and vaccination
Update vaccination data in vacc.ipynb script.
Run vaxx.ipynb, fit_data.ipynb, fitter_variant.ipynb, 
    
command to send data to TACC: 
scp data/* abouchni@frontera.tacc.utexas.edu:/work2/08090/abouchni/frontera/seir-austin/seir_regression/ImmunoSEIRS/US-BA12/data/


###Model_function:
DONE
Improved interaction between booster immunity and primary vaccination

###HPC deployment:
parallelize and launch on TACC??
DONE
save status file and import
    DONE

###command to download results:

scp abouchni@frontera.tacc.utexas.edu:/work2/08090/abouchni/frontera/seir-austin/seir_regression/ImmunoSEIRS/US-BA12/saved_data_test/* /media/multiphysics/0A3AF5863AF56ED7/Users/HINNOVIS/Pictures/NN-classification/data/vax_data/BA12/saved_data_test



###TEST;

Scenario assumptions; 

2 immune waning,
2 booster
2 BA.12 transmissibility:
    10% more transmissible and 25% immune escape

