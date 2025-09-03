library(readr)
library(rvest)
library(dplyr)

data_url <- "https://raw.githubusercontent.com/Gurobi/modeling-examples/master/price_optimization/"
data_url <- paste0(data_url, "HAB_data_2015to2022.csv")
avocado <- read_csv(data_url) %>% arrange(date)

regions = c(
    "Great_Lakes",
    "Midsouth",
    "Northeast",
    "Northern_New_England",
    "SouthCentral",
    "Southeast",
    "West",
    "Plains"
)

df <- avocado[avocado$region %in% regions, ]

df$revenue = df$units_sold * df$price

print(mean(df$units_sold))
print(mean(df$price))
print(mean(df$revenue))

tmp <- df %>%
    group_by(date) %>%
    summarize(units_sold = sum(units_sold),
              price = sum(price)/8,
              revenue = sum(revenue))
print(mean(tmp$units_sold))
print(mean(tmp$price))
print(mean(tmp$revenue))

tmp <- df %>%
    filter(peak == 0) %>%
    group_by(date) %>%
    summarize(units_sold = sum(units_sold),
              price = sum(price)/8,
              revenue = sum(revenue))
print(mean(tmp$units_sold))
print(mean(tmp$price))
print(mean(tmp$revenue))