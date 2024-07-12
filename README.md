# Indexed-Expenditures

This is a program that takes house price, rent, groceries, natural gas, electricity, healthcare, tuition, and median income expenditures/prices, indexes them to 1980 and creates a linear regression model based off of the past data.

This project is made to accompany a research paper discussing the growths in expenditures and wage stagnation since 1980 in the United States in real dollars to more clearly illustrate cost of living growth. This program requires multiple files to download from FRED, BLS, CMS, and NCES. Using the data, this program indexes cost to 1980, compiles the datasets, fits one or two regression lines depending on volatility of data to each area, provides R^2, adjusted R^2, coefficients, intercepts, and standard error, and finally visualizes them in comparison to each other. 
The program requires the pandas, pip, openpyxl, matplotlib, numpy, statsmodels, and scikitlearn packages.

Following are necessary downloads for the data and thus for the program to run.
Centers for Medicare & Medicaid Services. (2023, December 13). Historical | CMS. Www.cms.gov. https://www.cms.gov/data-research/statistics-trends-and-reports/national-health-expenditure-data/historical
Federal Reserve Economic Data. (2023, September 12). Mean Personal Income in the United States. FRED, Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/MAPAINUSA646N
Federal Reserve Economic Data. (2024a, April 26). Population. FRED, Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/POPTHM
Federal Reserve Economic Data. (2024b, May 15). Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average. Stlouisfed.org. https://fred.stlouisfed.org/series/CPILFESL
Federal Reserve Economic Data. (2024c, May 15). Consumer Price Index for All Urban Consumers: Food at Home in U.S. City Average. FRED, Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/CUSR0000SAF11
Federal Reserve Economic Data. (2024d, May 23). Median Sales Price of Houses Sold for the United States. FRED, Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/series/MSPUS
National Center for Education Statistics. (2023, December). Average undergraduate tuition, fees, room, and board rates charged for full-time students in degree-granting postsecondary institutions, by level and control of institution: Selected academic years, 1963-64 through 2022-23. Nces.ed.gov; Digest of Education Statistics. https://nces.ed.gov/programs/digest/d23/tables/dt23_330.10.asp


Some portions of the data, like the rent, are found through other websites due to not finding consistent downloadable data from the previous websites - here is where they were found.
Statista Research Department. (2023, September 4). Asking rent for U.S. unfurnished apartments 1980-2018. Statista. https://www.statista.com/statistics/200223/median-apartment-rent-in-the-us-since-1980/
United States Census Bureau. (2021, October 8). Historical Census of Housing Tables: Home Values. Census.gov. https://www.census.gov/data/tables/time-series/dec/coh-values.html
RentCafe. (2023, March). Average Rent in the U.S. & Rent Prices by State - RentCafe. Www.rentcafe.com. https://www.rentcafe.com/average-rent-market-trends/us/
