## Extract Masselot data, 23.12.25 SL
##   V4: All city-age combos. Based on Prop Age codes 5, 6 and 7

##    Inputs: Masselot 2023. https://zenodo.org/records/10288665 
##
##    Outputs: "erfs.rds": city-level, age-specific: crossprediction object, 
##                        mmt/p, RRs for slopes 
##             "erf_plots.pdf": city-level ERFs
##______________________________________________________________________________
## Notes:
## - Create folders "indata" for all data input and "outdata" for saving outputs
##
## - For RR slope calculation: specify RR percentile values in "per" 
##______________________________________________________________________________
## Author: Simon Lloyd
##______________________________________________________________________________

# Clear everything______________________________________________________________
rm(list=ls())


# Load libraries and functions__________________________________________________

library(pacman)

p_load(rio,           # to import data
       here,          # to locate files
       tidyverse,     # to clean, handle, and plot the data (includes ggplot2 package)
       janitor,       # to clean column names
       glue,          # for generating file names
       dplyr,
       tidyr
)

p_load(dlnm, 
       mixmeta,
       data.table,
       splines
)


# Setup ________________________________________________________________________
wd <- here()

cnst <- list()                      # store constants

cnst <- within(cnst, {
  path_in = glue("{wd}/data")
  path_out = glue("{wd}/output")
})

data <- list()                      # store all data
parameters <- list()                # store model parameters
# maps <- list()                    # store maps


# Get data _____________________________________________________________________

data$coefs <- import(glue("{cnst$path_in}/coefs.csv")) # columns: city, agegroup, b1, b2, ..., bk
data$vcov <- import(glue("{cnst$path_in}/vcov.csv")) # lower-triangular entries of variance-cov matrices
data$tmean <- import(glue("{cnst$path_in}/tmean_distribution.csv")) # percentiles of temp distribution
data$metadata <- import(glue("{cnst$path_in}/metadata.csv")) # city names, countries, locations + other variables


# Prepare data__________________________________________________________________

# get city names, countries and regions
cnst$lookup <- data$metadata %>%
  select(URAU_CODE, CNTR_CODE, URAU_NAME, region, LABEL, cntr_name)


# # add city names to temperature data
# data$tmean <- data$tmean      %>%
#   left_join(cnst$lookup %>% select(URAU_CODE, LABEL),
#             by = "URAU_CODE") %>%
#   rename(city = LABEL)        %>%
#   relocate(URAU_CODE, city, .before = 1)


# define percentiles for slopes in compact table of RRs
#   (Currently code is not flexible in naming the output object: if percentiles
#     change need to manually change labels for RR_working)
parameters$per <- c(0, 2.5, 97.5, 100)/100


# Generate empty object to store results________________________________________

# Crossprediction objects (city > age group)
crossprediction <- setNames(vector("list", length(data$tmean$URAU_CODE)), 
                      data$tmean$URAU_CODE)

for (city in data$tmean$URAU_CODE) {
  crossprediction[[city]] <- setNames(vector("list", length(unique(data$coefs$agegroup))),
                          unique(data$coefs$agegroup))
}
rm(city)


# mmt and mmp (city > age group)
empty_vec <- setNames(c(NA_real_, NA_real_), c("MMP", "MMT"))

mmtp <- setNames(vector("list", length(data$tmean$URAU_CODE)), 
                 data$tmean$URAU_CODE)

for (city in data$tmean$URAU_CODE) {
  mmtp[[city]] <- setNames(
    replicate(length(unique(data$coefs$agegroup)), empty_vec, simplify = FALSE),
    unique(data$coefs$agegroup)
  )
}
rm(city, empty_vec)


# RR tables for slopes
RR_for_slopes <- setNames(vector("list", length(data$tmean$URAU_CODE)), 
                 data$tmean$URAU_CODE)

for (city in data$tmean$URAU_CODE) {
  RR_for_slopes[[city]] <- setNames(vector("list", length(unique(data$coefs$agegroup))),
                           unique(data$coefs$agegroup))
}
rm(city)


# Get ERFs______________________________________________________________________
# Loop by city then age group

startTime <- Sys.time()

for (iCity in data$tmean$URAU_CODE) {                         # start iCity loop
  
  # get temperature by centile for city (119 percentiles)
  working_tg <- data$tmean     %>%
    filter(URAU_CODE == iCity) %>%
    select(-URAU_CODE)  %>%
    unlist()                   # convert df to named numeric
  
  # get vector of temperature percentiles used for predictions
  predper <- as.numeric(sub("%", "", names(working_tg)))
  
  # generate basis (119 percentiles *5 basis functions)
  working_basis <- onebasis(working_tg, 
                    fun = "bs", 
                    degree = 2, 
                    knots = quantile(working_tg, c(.1, .75, .9)))
  
  
  for (iAge in unique(data$coefs$agegroup)) {                  # start iAge loop
    
    # get coefficients (5 coefs)
    working_coefs <- data$coefs                     %>%
      filter(URAU_CODE == iCity & agegroup == iAge) %>%
      select(-URAU_CODE, -agegroup) %>%
      as.numeric() 
    
    # get vcov matrix (5*5)
    working_vcov <- data$vcov                       %>%
      filter(URAU_CODE == iCity & agegroup == iAge) %>%
      select(-URAU_CODE, -agegroup)                 %>%
      xpndMat()                                     # expand compressed 
    
    # initial (median centered) prediction
    cp_working  <- crosspred(working_basis,
                             coef = working_coefs,
                             vcov = working_vcov, 
                             model.link = "log", 
                             at = working_tg)
    
    # find mmt and mmp
    mmt_working <- cp_working$predvar[which.min(cp_working$allRRfit)]
    mmp_working <- pmin(pmax(predper[which(working_tg == mmt_working)],0),100)
    
    # mmt centered prediction
    cp_working  <- crosspred(working_basis,
                             coef = working_coefs,
                             vcov = working_vcov, 
                             model.link = "log", 
                             at = working_tg, 
                             cen = mmt_working)
    
    # extract RRs
    RR_working <- unlist(crosspred(working_basis,
                                   coef = working_coefs,
                                   vcov = working_vcov,
                                   model.link = "log",
                                   at = quantile(working_tg, parameters$per),
                                   cen = mmt_working)[c("allRRfit", "allRRlow","allRRhigh")])
    
    
    RR_working <- data.frame(
      matrix(RR_working, 1, 12, 
             dimnames = list(c(iAge),
                             c("RR00","RR02.5", "RR97.5", "RR100",
                               "RR00_lwr","RR02.5_lwr", "RR97.5_lwr","RR100_lwr",
                               "RR00_upr","RR02.5_upr", "RR97.5_upr","RR100_upr"))))
    
    
    # Store results___________________
    
    crossprediction[[iCity]][[iAge]] <- cp_working  # crossprediction object
    mmtp[[iCity]][[iAge]][["MMP"]] <- mmp_working   # mmp
    mmtp[[iCity]][[iAge]][["MMT"]] <- mmt_working   # mmt
    RR_for_slopes[[iCity]][[iAge]] <- RR_working    # rr for slope estimation
    
    
    # clean up_______________________
    rm(mmp_working, mmt_working, cp_working, 
       working_coefs, working_vcov, RR_working)

  }                                                              # end iAge loop
  rm(iAge)
  
  # clean up
  rm(working_tg, working_basis, predper)
  
}                                                               # end iCity loop
rm(iCity)

# time taken
endTime <- Sys.time()
round(print(endTime - startTime),2) # about 1.5 mins
rm(startTime, endTime)


# Generate results by country with city names___________________________________
# duplicate
crossprediction_results <- crossprediction
mmtp_results <- mmtp
RR_for_slopes_results <- RR_for_slopes


# Add city names
city_codes <- names(crossprediction_results) # get city codes from results
city_names <- cnst$lookup$LABEL[match(city_codes, cnst$lookup$URAU_CODE)]
                                     # match to city names

# substitute city names into results
names(crossprediction_results) <- city_names
names(mmtp_results) <- city_names
names(RR_for_slopes_results) <- city_names


# Group by country
city_to_country <- cnst$lookup$cntr_name[match(city_names, cnst$lookup$LABEL)]
names(city_to_country) <- city_names
                            # named vector of cities by country
countries <- unique(city_to_country) # unique countries

## add country level to results

# crosspredcition
crossprediction_results <- lapply(countries, function(ctry) {
  # cities in this country
  cities_in_country <- names(city_to_country)[city_to_country == ctry]
  
  # subset the existing renamed list
  crossprediction_results[cities_in_country]
})

names(crossprediction_results) <- countries

# mmtp
mmtp_results <- lapply(countries, function(ctry) {
  # cities in this country
  cities_in_country <- names(city_to_country)[city_to_country == ctry]
  
  # subset the existing renamed list
  mmtp_results[cities_in_country]
})

names(mmtp_results) <- countries

# RR for slopes
RR_for_slopes_results <- lapply(countries, function(ctry) {
  # cities in this country
  cities_in_country <- names(city_to_country)[city_to_country == ctry]
  
  # subset the existing renamed list
  RR_for_slopes_results[cities_in_country]
})

names(RR_for_slopes_results) <- countries

# clean up
rm(city_codes, city_names, city_to_country, countries)


# Save results__________________________________________________________________
results <- list(crossprediction_results = crossprediction_results, 
                mmtp_results = mmtp_results,
                RR_for_slopes_results = RR_for_slopes_results)

saveRDS(results, file = glue('{cnst$path_out}/erfs.rds')) 

rm(results, crossprediction_results, mmtp_results, RR_for_slopes_results)


# Plots_________________________________________________________________________
#   Generate a long PDF, with 9 ERFs per page

startTime <- Sys.time()

# specify x label
xlab <- expression(paste("Temperature (",degree,"C)"))

# send plots to PDF
pdf(glue('{cnst$path_out}/erf_plots.pdf'),width=9,height=9)
layout(matrix(seq(3*3),nrow=3,byrow=T))
par(mex=0.8,mgp=c(2.5,1,0),las=1)


for (iCity in data$tmean$URAU_CODE) {                         # start iCity loop
  
  # get temperature by centile for city (119 percentiles)
  tg_working <- data$tmean     %>%
    filter(URAU_CODE == iCity) %>%
    select(-URAU_CODE)  %>%
    unlist()                   # convert df to named numeric
  
  for (iAge in names(crossprediction[[iCity]])) {              # start iAge loop
    
    # get working crossprediction
    cp_working <- crossprediction[[iCity]][[iAge]]
    
    # get working mmt
    mmt_working <- mmtp[[iCity]][[iAge]][["MMT"]]
    
    # get name of plot
    city_name <- cnst$lookup     %>%
      filter(URAU_CODE == iCity) %>%
      pull(LABEL)
    
    name <- str_glue("{city_name}, {iAge}")
    
    rm(city_name)
    
    # plot
    plot(cp_working, 
         xlab = xlab, ylab = "RR", ylim=c(0.8,2.0),
         lwd = 2, col = 1, bty = "o",
         main=name)
    abline(v = c(mmt_working, quantile(tg_working,c(0.01,0.99))),
           lty=c(1,2,2))
    
    # clean up
    rm(cp_working, mmt_working)
  }                                                              # end iAge loop
  rm(iAge)
  rm(tg_working)

}                                                               # end iCity loop
rm(iCity)

# close PDF
dev.off()

# time taken
endTime <- Sys.time()
round(print(endTime - startTime),2) # ~30s
rm(startTime, endTime)


