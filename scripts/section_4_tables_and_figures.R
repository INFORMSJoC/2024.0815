# Clear workspace

rm(list = ls())

# Define exit function

exit <- function() { invokeRestart("abort") } 

# Load libraries

library(readr)
library(janitor)
library(dplyr)
library(ggplot2)
library(reshape2)
library(tidyr)
library(scales)
library(xtable)
library(ggridges)
library(gridExtra)
library(forcats)

###############################################################################
#   Part 1 of 3: Analysis of "small n" instances. Matches paper v0
###############################################################################

# Read data from output.csv

if(!file.exists("../results/output.csv")) {
    list_of_files <- list.files(path = "../results/hpc_csv", recursive = TRUE,
                                pattern = "\\.csv$", full.names = TRUE)
    tic <- 1
    ind <- 1
    df <- c()
    while(tic <= length(list_of_files)) {
        toc <- min(tic + 500, length(list_of_files))
        df[[ind]] <- read_csv(list_of_files[tic : toc])
        tic <- toc + 1
        ind <- ind + 1
    }
    df <- do.call(rbind, df)
    write_csv(df, "../results/output.csv")
} else {
    df <- read_csv("../results/output.csv")
}

# Filter as necessary. As of 2024-05-23, expected number of rows in df
# is 37800 * 4 = 151200

# Remove rastrigin2d instances
# Remove bestsample v-domain
# Remove svm v-domain
# Remove svmplus v-domain

df <- df %>% filter(func != "rotated_hyper_ellipsoid")
df <- df %>% filter(func != "rastrigin2d")
df <- df %>% filter(v_domain != "chp.05")
df <- df %>% filter(v_domain != "chp.1")
#df <- df %>% filter(v_domain != "bestsample")
#df <- df %>% filter(v_domain != "svm")
#df <- df %>% filter(v_domain != "svmplus")

# Sort (not really necessary, just for cleanliness)

df <- df %>% arrange(seed, func, noise, v_domain)

# Factor v_domain column

df$v_domain <- factor(df$v_domain, c("box", "ch", "isofor", "chplus"))

# Create plot function (as of 2024-05-23, this is not being used in this
# file). And not updated during revision (2025-01-23).

my_plot_function <- function(my_func, my_sampling, my_learning) {
    
    tmp <- df %>%
        filter(func == my_func, sampling == my_sampling, learning == my_learning) %>%
        select(
            func, noise, sampling, sample_sz, seed, learning, v_domain,
            funval_err, optval_err, optsol_err
        ) %>%
        group_by(v_domain) %>%
        summarize(med_funval_err = median(funval_err),
                  med_optval_err = median(optval_err),
                  med_optsol_err = median(optsol_err))
    
    val <- tmp$med_funval_err[1] # box
    df$funval_err <- df$funval_err / val
    # tmp[[2]] <- formatC(tmp[[2]] / val, format = "f", digits = 2)
    
    val <- tmp$med_optval_err[1] # box
    df$optval_err <- df$optval_err / val
    # tmp[[3]] <- formatC(tmp[[3]] / val, format = "f", digits = 2)

    val <- tmp$med_optsol_err[1] # box
    df$optsol_err <- df$optsol_err / val
    # tmp[[4]] <- formatC(tmp[[4]] / val, format = "f", digits = 2)

    tmp$v_domain <- as.character(tmp$v_domain)
    tmp$v_domain[tmp$v_domain == "box"] <- "Box"
    tmp$v_domain[tmp$v_domain == "ch"] <- "CH"
    tmp$v_domain[tmp$v_domain == "chplus"] <- "CH+"
    tmp$v_domain[tmp$v_domain == "isofor"] <- "IsoFor"
    names(tmp) <- c("Validity\nDomain", "Median Function\nValue Error", "Median Optimal\nValue Error", "Median Optimal\nSolution Error")

    print(tmp %>% xtable(digits = 2))
    
}

# Create table printing function

my_print_function <- function(my_sampling) {
    
    tmp <- df %>%
        filter(sampling == my_sampling) %>%
        select(
            func, noise, sampling, sample_sz, seed, learning, v_domain,
            funval_err, optval_err, optsol_err
        ) %>%
        group_by(func, v_domain) %>%
        summarize(med_funval_err = median(funval_err),
                  med_optval_err = median(optval_err),
                  med_optsol_err = median(optsol_err))
    
    # Be careful with this code! Is there a better way to do this?
    
    for(j in 1 : (nrow(tmp)/4)) {
        
        i <- (j - 1)*4 + 1
        
        val <- tmp$med_funval_err[i] # box
        tmp$med_funval_err[i + 0] <- tmp$med_funval_err[i + 0] / val
        tmp$med_funval_err[i + 1] <- tmp$med_funval_err[i + 1] / val
        tmp$med_funval_err[i + 2] <- tmp$med_funval_err[i + 2] / val
        tmp$med_funval_err[i + 3] <- tmp$med_funval_err[i + 3] / val
        
        val <- tmp$med_optval_err[i] # box
        tmp$med_optval_err[i + 0] <- tmp$med_optval_err[i + 0] / val
        tmp$med_optval_err[i + 1] <- tmp$med_optval_err[i + 1] / val
        tmp$med_optval_err[i + 2] <- tmp$med_optval_err[i + 2] / val
        tmp$med_optval_err[i + 3] <- tmp$med_optval_err[i + 3] / val
        
        val <- tmp$med_optsol_err[i] # box
        tmp$med_optsol_err[i + 0] <- tmp$med_optsol_err[i + 0] / val
        tmp$med_optsol_err[i + 1] <- tmp$med_optsol_err[i + 1] / val
        tmp$med_optsol_err[i + 2] <- tmp$med_optsol_err[i + 2] / val
        tmp$med_optsol_err[i + 3] <- tmp$med_optsol_err[i + 3] / val
        
    }
    
    tmp$med_funval_err <- formatC(tmp$med_funval_err, format = "f", digits = 2)
    tmp$med_optval_err <- formatC(tmp$med_optval_err, format = "f", digits = 2)
    tmp$med_optsol_err <- formatC(tmp$med_optsol_err, format = "f", digits = 2)
    
    tmp$v_domain <- as.character(tmp$v_domain)
    tmp$v_domain[tmp$v_domain == "box"] <- "Box"
    tmp$v_domain[tmp$v_domain == "ch"] <- "CH"
    tmp$v_domain[tmp$v_domain == "chplus"] <- "CH+"
    tmp$v_domain[tmp$v_domain == "isofor"] <- "IsoFor"
    names(tmp) <- c("Func", "V Dom", "Med FunValErr", "Med OptValErr", "Med OptSolErr")
    
    names(tmp) <- c("Function", "Validity Domain", "Median Function Value Error",
                    "Median Optimal Value Error", "Median Optimal Solution Error")
    
    tmp$Function[tmp$Function == "beale"] <- "Beale"
    tmp$Function[tmp$Function == "griewank"] <- "Griewank"
    tmp$Function[tmp$Function == "peaks"] <- "Peaks"
    tmp$Function[tmp$Function == "powell"] <- "Powell"
    tmp$Function[tmp$Function == "qing"] <- "Qing"
    tmp$Function[tmp$Function == "quintic"] <- "Quintic"
    tmp$Function[tmp$Function == "rastrigin10d"] <- "Rastrigin"
    
    tmp$`Validity Domain`[tmp$`Validity Domain` == "Box"] <- "{\\sc Box}"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "IsoFor"] <- "{\\sc IsoFor}"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "CH+"] <- "$\\CH^+$"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "CH"] <- "$\\CH$"
    
    # print(
    #     tmp %>% xtable(digits = 2),
    #     include.rownames = FALSE,
    #     sanitize.text.function = function(x){x},
    #     file = paste0("../results/tables/", my_sampling, ".tex")
    #     )
    
    tmp
}

# Save Tables XXX and YYY (2024-05-23) to file. These tables show how we perform
# on the various functions. Table XXX shows the uniform sampling rule. Table YYY
# shows the normal_at_min sampling rule.

df_uniform <- my_print_function(my_sampling = "uniform")
df_normal <- my_print_function(my_sampling = "normal_at_min")
#my_print_function(my_sampling = "all")

# 2024-06-12: Attempt to combine uniform and normal_at_min tables

df_all <- cbind(
    df_uniform[, c(1, 2, 3)],
    df_normal[, 3],
    df_uniform[, 4],
    df_normal[, 4],
    df_uniform[, 5],
    df_normal[, 5]
)
print(
    df_all %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/section_4_errors.tex"
)

# Save table showing R2 scores, similar to table 1 in Shi et al.

tmp <- df %>%
    filter(v_domain == "box") %>%
    group_by(func, learning) %>%
    summarize(avgR2 = mean(R2_score)) %>%
    pivot_wider(names_from = func, values_from = avgR2)

tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"

tmp <- tmp %>%
    rename(Beale = beale) %>%
    rename(Griewank = griewank) %>%
    rename(Peaks = peaks) %>%
    rename(Powell = powell) %>%
    rename(Qing = qing) %>%
    rename(Quintic = quintic) %>%
    rename(Rastrigin = rastrigin10d) %>%
    rename(`ML Technique` = learning)

print(
    tmp %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/section_4_r2.tex"
)

# Save table showing median ML training times (similarly constructed as table for R2 scores)

tmp <- df %>%
    filter(v_domain == "box") %>%
    group_by(func, learning) %>%
    summarize(medtime = median(ML_train_time)) %>%
    pivot_wider(names_from = func, values_from = medtime)

tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"

tmp <- tmp %>%
    rename(Beale = beale) %>%
    rename(Griewank = griewank) %>%
    rename(Peaks = peaks) %>%
    rename(Powell = powell) %>%
    rename(Qing = qing) %>%
    rename(Quintic = quintic) %>%
    rename(Rastrigin = rastrigin10d) %>%
    rename(`ML Technique` = learning)

print(
    tmp %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/section_4_ml_train_times.tex"
)

# Save table showing median opt setup times

# This requires re-reading df and this time including the enlarged versions
# of CH+

df <- read_csv("../results/output.csv")
df <- df %>% filter(func != "rotated_hyper_ellipsoid")
df <- df %>% filter(func != "rastrigin2d")
df <- df %>% arrange(seed, func, noise, v_domain)
df$v_domain <- factor(df$v_domain, c("box", "ch", "isofor", "chplus", "chp.05", "chp.1"))

tmp <- df %>%
    group_by(v_domain, learning) %>%
    summarize(myval = median(opt_setup_time)) %>%
    pivot_wider(names_from = v_domain, values_from = myval)

tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"

tmp <- tmp %>%
    rename(`{\\sc Box}` = box) %>%
    rename(`$\\CH$` = ch) %>%
    rename(`{\\sc IsoFor}` = isofor) %>%
    rename(`$\\CH^+$` = chplus) %>%
    rename(`ML Technique` = learning)

print(
    tmp %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/section_s1_opt_setup_times.tex"
)

# Save table showing median opt solve times (similarly constructed as table for setup times)

tmp <- df %>%
    group_by(v_domain, learning) %>%
    summarize(myval = median(opt_opt_time)) %>%
    pivot_wider(names_from = v_domain, values_from = myval)

tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"

tmp <- tmp %>%
    rename(`{\\sc Box}` = box) %>%
    rename(`$\\CH$` = ch) %>%
    rename(`{\\sc IsoFor}` = isofor) %>%
    rename(`$\\CH^+$` = chplus) %>%
    rename(`ML Technique` = learning)

print(
    tmp %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/section_s1_opt_solve_times.tex"
)

###################################################################################
#   Part 2 of 3: Analysis of CH+(eps) for "smaller n" instances. New in v1 of paper
###################################################################################

# Read data from output.csv

df <- read_csv("../results/output.csv")

# Filter as necessary. As of 2025-02-20, expected number of rows in df
# is 37800 * 4 = 151200

df <- df %>%
    filter(func != "rotated_hyper_ellipsoid") %>%
    filter(func != "rastrigin2d") %>%
    filter(v_domain %in% c("box", "chp.05", "chp.1", "chplus")) %>%
    arrange(seed, func, noise, v_domain)

# Factor v_domain column

df$v_domain <- factor(df$v_domain, c("box", "chplus", "chp.05", "chp.1"))

# Create table printing function

my_print_function <- function(my_sampling) {
    
    tmp <- df %>%
        filter(sampling == my_sampling) %>%
        select(
            func, noise, sampling, sample_sz, seed, learning, v_domain,
            funval_err, optval_err, optsol_err
        ) %>%
        group_by(func, v_domain) %>%
        summarize(med_funval_err = median(funval_err),
                  med_optval_err = median(optval_err),
                  med_optsol_err = median(optsol_err))
    
    # Be careful with this code! Is there a better way to do this?
    
    for(j in 1 : (nrow(tmp)/4)) {
        
        i <- (j - 1)*4 + 1
        
        val <- tmp$med_funval_err[i] # box
        tmp$med_funval_err[i + 0] <- tmp$med_funval_err[i + 0] / val
        tmp$med_funval_err[i + 1] <- tmp$med_funval_err[i + 1] / val
        tmp$med_funval_err[i + 2] <- tmp$med_funval_err[i + 2] / val
        tmp$med_funval_err[i + 3] <- tmp$med_funval_err[i + 3] / val
        
        val <- tmp$med_optval_err[i] # box
        tmp$med_optval_err[i + 0] <- tmp$med_optval_err[i + 0] / val
        tmp$med_optval_err[i + 1] <- tmp$med_optval_err[i + 1] / val
        tmp$med_optval_err[i + 2] <- tmp$med_optval_err[i + 2] / val
        tmp$med_optval_err[i + 3] <- tmp$med_optval_err[i + 3] / val
        
        val <- tmp$med_optsol_err[i] # box
        tmp$med_optsol_err[i + 0] <- tmp$med_optsol_err[i + 0] / val
        tmp$med_optsol_err[i + 1] <- tmp$med_optsol_err[i + 1] / val
        tmp$med_optsol_err[i + 2] <- tmp$med_optsol_err[i + 2] / val
        tmp$med_optsol_err[i + 3] <- tmp$med_optsol_err[i + 3] / val
        
    }
    
    tmp$med_funval_err <- formatC(tmp$med_funval_err, format = "f", digits = 2)
    tmp$med_optval_err <- formatC(tmp$med_optval_err, format = "f", digits = 2)
    tmp$med_optsol_err <- formatC(tmp$med_optsol_err, format = "f", digits = 2)
    
    tmp$v_domain <- as.character(tmp$v_domain)
    tmp$v_domain[tmp$v_domain == "chplus"] <- "CH+"
    names(tmp) <- c("Func", "V Dom", "Med FunValErr", "Med OptValErr", "Med OptSolErr")
    
    names(tmp) <- c("Function", "Validity Domain", "Median Function Value Error",
                    "Median Optimal Value Error", "Median Optimal Solution Error")
    
    tmp$Function[tmp$Function == "rotated_hyper_ellipsoid"] <- "Rotated Hyper Ellipsoid"

    tmp$`Validity Domain`[tmp$`Validity Domain` == "box"] <- "{\\sc Box}"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "CH+"] <- "$\\CH^+$"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "chp.05"] <- "$\\CH^+_{0.05}$"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "chp.1"] <- "$\\CH^+_{0.10}$"

    tmp
}

df_uniform <- my_print_function(my_sampling = "uniform")
df_normal <- my_print_function(my_sampling = "normal_at_min")
# my_print_function(my_sampling = "all")

# 2024-06-12: Attempt to combine uniform and normal_at_min tables

df_all <- cbind(
    df_uniform[, c(1, 2, 3)],
    df_normal[, 3],
    df_uniform[, 4],
    df_normal[, 4],
    df_uniform[, 5],
    df_normal[, 5]
)
print(
    df_all %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/section_4_errors_enlarged.tex"
)


# df <- read_csv("../results/output.csv")
#
# # Filter as necessary. As of 2025-01-28, expected number of
# # rows in df is 900 * 5 = 4500
#
# # Remove rastrigin2d instances
# # Remove bestsample v-domain
# # Remove svm v-domain
# # Remove svmplus v-domain
#
# df <- df %>% filter(func == "rotated_hyper_ellipsoid")
# df <- df %>% filter(func != "rastrigin2d")
# df <- df %>% filter(v_domain != "bestsample")
# df <- df %>% filter(v_domain != "svm")
# df <- df %>% filter(v_domain != "svmplus")
#
# # Sort (not really necessary, just for cleanliness)
#
# df <- df %>% arrange(seed, func, noise, v_domain)
#
# # Factor v_domain column
#
# df$v_domain <- factor(df$v_domain, c("box", "ch", "chplus", "chp.05", "chp.1"))
#
# # Create table printing function (without IsoFor)
#
# my_print_function <- function(my_sampling) {
#
#     tmp <- df %>%
#         filter(sampling == my_sampling) %>%
#         select(
#             func, noise, sampling, sample_sz, seed, learning, v_domain,
#             funval_err, optval_err, optsol_err
#         ) %>%
#         group_by(func, v_domain) %>%
#         summarize(med_funval_err = median(funval_err),
#                   med_optval_err = median(optval_err),
#                   med_optsol_err = median(optsol_err))
#
#     # Be careful with this code! Is there a better way to do this?
#
#     for(j in 1 : (nrow(tmp)/5)) {
#
#         i <- (j - 1)*5 + 1
#
#         val <- tmp$med_funval_err[i] # box
#         tmp$med_funval_err[i + 0] <- tmp$med_funval_err[i + 0] / val
#         tmp$med_funval_err[i + 1] <- tmp$med_funval_err[i + 1] / val
#         tmp$med_funval_err[i + 2] <- tmp$med_funval_err[i + 2] / val
#         tmp$med_funval_err[i + 3] <- tmp$med_funval_err[i + 3] / val
#         tmp$med_funval_err[i + 4] <- tmp$med_funval_err[i + 4] / val
#
#         val <- tmp$med_optval_err[i] # box
#         tmp$med_optval_err[i + 0] <- tmp$med_optval_err[i + 0] / val
#         tmp$med_optval_err[i + 1] <- tmp$med_optval_err[i + 1] / val
#         tmp$med_optval_err[i + 2] <- tmp$med_optval_err[i + 2] / val
#         tmp$med_optval_err[i + 3] <- tmp$med_optval_err[i + 3] / val
#         tmp$med_optval_err[i + 4] <- tmp$med_optval_err[i + 4] / val
#
#         val <- tmp$med_optsol_err[i] # box
#         tmp$med_optsol_err[i + 0] <- tmp$med_optsol_err[i + 0] / val
#         tmp$med_optsol_err[i + 1] <- tmp$med_optsol_err[i + 1] / val
#         tmp$med_optsol_err[i + 2] <- tmp$med_optsol_err[i + 2] / val
#         tmp$med_optsol_err[i + 3] <- tmp$med_optsol_err[i + 3] / val
#         tmp$med_optsol_err[i + 4] <- tmp$med_optsol_err[i + 4] / val
#
#     }
#
#     tmp$med_funval_err <- formatC(tmp$med_funval_err, format = "f", digits = 2)
#     tmp$med_optval_err <- formatC(tmp$med_optval_err, format = "f", digits = 2)
#     tmp$med_optsol_err <- formatC(tmp$med_optsol_err, format = "f", digits = 2)
#
#     tmp$v_domain <- as.character(tmp$v_domain)
#     tmp$v_domain[tmp$v_domain == "box"] <- "Box"
#     tmp$v_domain[tmp$v_domain == "ch"] <- "CH"
#     tmp$v_domain[tmp$v_domain == "chplus"] <- "CH+"
#     tmp$v_domain[tmp$v_domain == "chp.05"] <- "CH+(0.05)"
#     tmp$v_domain[tmp$v_domain == "chp.1"] <- "CH+(0.10)"
#     names(tmp) <- c("Func", "V Dom", "Med FunValErr", "Med OptValErr", "Med OptSolErr")
#
#     names(tmp) <- c("Function", "Validity Domain", "Median Function Value Error",
#                     "Median Optimval Value Error", "Median Optimal Solution Error")
#
#     tmp$Function[tmp$Function == "beale"] <- "Beale"
#     tmp$Function[tmp$Function == "griewank"] <- "Griewank"
#     tmp$Function[tmp$Function == "peaks"] <- "Peaks"
#     tmp$Function[tmp$Function == "powell"] <- "Powell"
#     tmp$Function[tmp$Function == "qing"] <- "Qing"
#     tmp$Function[tmp$Function == "quintic"] <- "Quintic"
#     tmp$Function[tmp$Function == "rastrigin10d"] <- "Rastrigin"
#     tmp$Function[tmp$Function == "rotated_hyper_ellipsoid"] <- "Rotated Hyperellipsoid"
#
#     tmp$`Validity Domain`[tmp$`Validity Domain` == "Box"] <- "{\\sc Box}"
#     #tmp$`Validity Domain`[tmp$`Validity Domain` == "IsoFor"] <- "{\\sc IsoFor}"
#     tmp$`Validity Domain`[tmp$`Validity Domain` == "CH+"] <- "$\\CH^+$"
#     tmp$`Validity Domain`[tmp$`Validity Domain` == "CH"] <- "$\\CH$"
#
#     # print(
#     #     tmp %>% xtable(digits = 2),
#     #     include.rownames = FALSE,
#     #     sanitize.text.function = function(x){x},
#     #     file = paste0("../results/tables/", my_sampling, ".tex")
#     #     )
#
#     tmp
# }
#
#
# # Save Tables XXX and YYY (2024-05-23) to file. These tables show how we
# # perform on the various functions. Table XXX shows the uniform sampling
# # rule. Table YYY shows the normal_at_min sampling rule.
#
# #df_uniform <- my_print_function(my_sampling = "uniform")
# df_normal <- my_print_function(my_sampling = "normal_at_min")
# #my_print_function(my_sampling = "all")
#
# # 2024-06-12: Attempt to combine uniform and normal_at_min tables
#
# #df_all <- cbind(
# #    df_uniform[, c(1, 2, 3)],
# #    df_normal[, 3],
# #    df_uniform[, 4],
# #    df_normal[, 4],
# #    df_uniform[, 5],
# #    df_normal[, 5]
# #)
# df_all <- df_normal
# print(
#     df_all %>% xtable(digits = 2),
#     include.rownames = FALSE,
#     sanitize.text.function = function(x){x},
#     file = "../results/tables/all_larger_n.tex"
# )
#
# # Save table showing R2 scores, similar to table 1 in Shi et al.
#
# tmp <- df %>%
#     filter(v_domain == "box") %>%
#     group_by(func, learning) %>%
#     summarize(avgR2 = mean(R2_score)) %>%
#     pivot_wider(names_from = func, values_from = avgR2)
#
# tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
# tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
# tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"
#
# tmp <- tmp %>%
#     rename(`Rotated Hyperellipsoid` = rotated_hyper_ellipsoid) %>%
#     rename(`ML Technique` = learning)
#
# print(
#     tmp %>% xtable(digits = 2),
#     include.rownames = FALSE,
#     sanitize.text.function = function(x){x},
#     file = "../results/tables/r2_larger_n.tex"
# )
#
# # Save table showing median ML training times (similarly constructed as table for R2 scores)
#
# tmp <- df %>%
#     filter(v_domain == "box") %>%
#     group_by(func, learning) %>%
#     summarize(medtime = median(ML_train_time)) %>%
#     pivot_wider(names_from = func, values_from = medtime)
#
# tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
# tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
# tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"
#
# tmp <- tmp %>%
#     rename(`Rotated Hyperellipsoid` = rotated_hyper_ellipsoid) %>%
#     rename(`ML Technique` = learning)
#
# print(
#     tmp %>% xtable(digits = 2),
#     include.rownames = FALSE,
#     sanitize.text.function = function(x){x},
#     file = "../results/tables/ml_train_times_larger_n.tex"
# )
#
# # Save table showing median opt setup times
#
# tmp <- df %>%
#     group_by(v_domain, learning) %>%
#     summarize(myval = median(opt_setup_time)) %>%
#     pivot_wider(names_from = v_domain, values_from = myval)
#
# tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
# tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
# tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"
#
# tmp <- tmp %>%
#     rename(`{\\sc Box}` = box) %>%
#     rename(`$\\CH$` = ch) %>%
#     rename(`$\\CH^+$` = chplus) %>%
#     rename(`ML Technique` = learning)
#
# print(
#     tmp %>% xtable(digits = 2),
#     include.rownames = FALSE,
#     sanitize.text.function = function(x){x},
#     file = "../results/tables/opt_setup_times_larger_n.tex"
# )
#
# # Save table showing median opt solve times (similarly constructed as table for setup times)
#
# tmp <- df %>%
#     group_by(v_domain, learning) %>%
#     summarize(myval = median(opt_opt_time)) %>%
#     pivot_wider(names_from = v_domain, values_from = myval)
#
# tmp$learning[tmp$learning == "forest"] <- "{\\em RandomForestRegressor}"
# tmp$learning[tmp$learning == "gb"] <- "{\\em GradientBoostingRegressor}"
# tmp$learning[tmp$learning == "net"] <- "{\\em MLPRegressor}"
#
# tmp <- tmp %>%
#     rename(`{\\sc Box}` = box) %>%
#     rename(`$\\CH$` = ch) %>%
#     rename(`$\\CH^+$` = chplus) %>%
#     rename(`ML Technique` = learning)
#
# print(
#     tmp %>% xtable(digits = 2),
#     include.rownames = FALSE,
#     sanitize.text.function = function(x){x},
#     file = "../results/tables/opt_solve_times_larger_n.tex"
# )


###############################################################################
#   Part 3 of 3: Analysis of "larger n". New in v1 of paper
###############################################################################

# Read data from output.csv

df <- read_csv("../results/output.csv")

# Filter as necessary. As of 2025-02-05, expected number of rows in df
# is 900 * 5 = 2700

df <- df %>%
    filter(func == "rotated_hyper_ellipsoid") %>%
    filter(v_domain %in% c("box", "ch", "chp.05", "chp.1", "chplus")) %>%
    arrange(seed, func, noise, v_domain)

# Factor v_domain column

df$v_domain <- factor(df$v_domain, c("box", "ch", "chplus", "chp.05", "chp.1"))

# Create table printing function

my_print_function <- function(my_sampling) {
    
    tmp <- df %>%
        filter(sampling == my_sampling) %>%
        select(
            func, noise, sampling, sample_sz, seed, learning, v_domain,
            funval_err, optval_err, optsol_err
        ) %>%
        group_by(func, v_domain) %>%
        summarize(med_funval_err = median(funval_err),
                  med_optval_err = median(optval_err),
                  med_optsol_err = median(optsol_err))
    
    # Be careful with this code! Is there a better way to do this?
    
    for(j in 1 : (nrow(tmp)/4)) {
        
        i <- (j - 1)*5 + 1
        
        val <- tmp$med_funval_err[i] # box
        tmp$med_funval_err[i + 0] <- tmp$med_funval_err[i + 0] / val
        tmp$med_funval_err[i + 1] <- tmp$med_funval_err[i + 1] / val
        tmp$med_funval_err[i + 2] <- tmp$med_funval_err[i + 2] / val
        tmp$med_funval_err[i + 3] <- tmp$med_funval_err[i + 3] / val
        tmp$med_funval_err[i + 4] <- tmp$med_funval_err[i + 4] / val
        
        val <- tmp$med_optval_err[i] # box
        tmp$med_optval_err[i + 0] <- tmp$med_optval_err[i + 0] / val
        tmp$med_optval_err[i + 1] <- tmp$med_optval_err[i + 1] / val
        tmp$med_optval_err[i + 2] <- tmp$med_optval_err[i + 2] / val
        tmp$med_optval_err[i + 3] <- tmp$med_optval_err[i + 3] / val
        tmp$med_optval_err[i + 4] <- tmp$med_optval_err[i + 4] / val
        
        val <- tmp$med_optsol_err[i] # box
        tmp$med_optsol_err[i + 0] <- tmp$med_optsol_err[i + 0] / val
        tmp$med_optsol_err[i + 1] <- tmp$med_optsol_err[i + 1] / val
        tmp$med_optsol_err[i + 2] <- tmp$med_optsol_err[i + 2] / val
        tmp$med_optsol_err[i + 3] <- tmp$med_optsol_err[i + 3] / val
        tmp$med_optsol_err[i + 4] <- tmp$med_optsol_err[i + 4] / val
        
    }
    
    tmp$med_funval_err <- formatC(tmp$med_funval_err, format = "f", digits = 2)
    tmp$med_optval_err <- formatC(tmp$med_optval_err, format = "f", digits = 2)
    tmp$med_optsol_err <- formatC(tmp$med_optsol_err, format = "f", digits = 2)
    
    tmp$v_domain <- as.character(tmp$v_domain)
    tmp$v_domain[tmp$v_domain == "chplus"] <- "CH+"
    names(tmp) <- c("Func", "V Dom", "Med FunValErr", "Med OptValErr", "Med OptSolErr")
    
    names(tmp) <- c("Function", "Validity Domain", "Median Function Value Error",
                    "Median Optimal Value Error", "Median Optimal Solution Error")
    
    tmp$Function[tmp$Function == "rotated_hyper_ellipsoid"] <- "Rotated Hyper Ellipsoid"

    tmp$`Validity Domain`[tmp$`Validity Domain` == "box"] <- "{\\sc Box}"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "ch"] <- "$\\CH$"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "CH+"] <- "$\\CH^+$"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "chp.05"] <- "$\\CH^+_{0.05}$"
    tmp$`Validity Domain`[tmp$`Validity Domain` == "chp.1"] <- "$\\CH^+_{0.10}$"

    tmp
}

# df_uniform <- my_print_function(my_sampling = "uniform") # We don't do uniform here
df_normal <- my_print_function(my_sampling = "normal_at_min")
# my_print_function(my_sampling = "all")

print(
    df_normal %>% xtable(digits = 2),
    include.rownames = FALSE,
    sanitize.text.function = function(x){x},
    file = "../results/tables/section_4_errors_larger_n.tex"
)

