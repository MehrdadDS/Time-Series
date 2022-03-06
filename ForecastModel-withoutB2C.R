options(java.parameters = "-Xmx2048m")
library(forecast)
library(xlsx)
library(readxl)
library(writexl)
library(RMySQL)
library(DBI)
library(odbc)

driver = dbDriver("MySQL")
mydb = dbConnect(driver, user = 'root', password = '123456', dbname  = 'forecasting_schema')

query = dbSendQuery(mydb, "Select * from daily_forecast")
data = fetch(query, n=-1)

weekly_data = data.frame(data)

query = dbSendQuery(mydb, "Select * from terminal_list")
data = fetch(query, n=-1)

terminal_data = data.frame(data)

dbDisconnect(mydb)

#External Variables
amazonHistorical = readxl::read_xlsx("C:/Forecast tools/P&D Forecast/P&D Forecast v2/Excel Input/AmazonData.xlsx", sheet = "Historical Input")
amazonData = readxl::read_xlsx("C:/Forecast tools/P&D Forecast/P&D Forecast v2/Excel Input/AmazonData.xlsx", 7)
amazonDataStops = readxl::read_xlsx("C:/Forecast tools/P&D Forecast/P&D Forecast v2/Excel Input/AmazonData.xlsx", 8)
cyber = read_xlsx("C:/Forecast tools/P&D Forecast/P&D Forecast v2/Excel Input/CyberWeekRegressor.xlsx", 1)
b_2_c = read_xlsx("C:/Forecast tools/P&D Forecast/P&D Forecast v2/Excel Input/B2CRegressor.xlsx", sheet = "Weekly")





list_of_terminals = sort(unique(terminal_data$Terminal),decreasing = FALSE)

for (sheet in 0:7){
  
  for (terminal in 0:(length(list_of_terminals)-1)){
    
    current_terminal = list_of_terminals[terminal+1]
    
    opt_start_year = as.numeric(terminal_data[terminal_data$Terminal == current_terminal,9])
    opt_start_week = as.numeric(terminal_data[terminal_data$Terminal == current_terminal,10])
    adjustment = 0
    
    start_year = min(as.numeric(weekly_data[weekly_data$Terminal == current_terminal,6]))
    start_week = min(as.numeric(weekly_data[weekly_data$Terminal == current_terminal & weekly_data$Year == start_year,9]))
    end_year = max(as.numeric(weekly_data[weekly_data$Terminal == current_terminal,6]))
    end_week = max(as.numeric(weekly_data[weekly_data$Terminal == current_terminal & weekly_data$Year == end_year,9]))
    
    #Change the ts values when changing actuals
    values = ts((weekly_data[weekly_data$Terminal == current_terminal,10+sheet]), start = c(start_year,start_week), end=c(end_year,end_week), frequency=52)
    values = window(values, start = c(opt_start_year,opt_start_week))
    values = tsclean(values)
    
    current_cyber = cyber[cyber$Year >= opt_start_year,]
    current_cyber = current_cyber[opt_start_week:nrow(current_cyber),]
    
    amazonWeeklyHistorical = amazonHistorical[amazonHistorical$Terminal == current_terminal,]
    amazonWeeklyHistorical = amazonWeeklyHistorical[!is.na(amazonWeeklyHistorical$Week),]
    
    #amazon_edd_reg = amazon_edd_actuals[amazon_edd_actuals$Terminal == current_terminal,]
    
    if (opt_start_year>2017){
      adjustment = 1+adjustment + (opt_start_year-2017)*52
    }
    if (opt_start_week>1){
      adjustment = adjustment + opt_start_week
    }
    amazonWeeklyHistorical = amazonWeeklyHistorical[(adjustment):263,4]#Update plus number of new weeks
    colnames(amazonWeeklyHistorical)[1] = "Amazon"
    
    current_b2c = b_2_c[b_2_c$Terminal == current_terminal,]
    current_b2c = current_b2c[(adjustment):nrow(current_b2c),]
    
    if ((sheet == 0 &&  !(current_terminal %in% c(66,132,133,135,138,141,142,143,148,182,183,385,386,387,405,406,407,415,416,417,418,420,422,423,424,430,431,440,441,442,443,444,445,446,451,456,543,544))) || (sheet == 4 &&  (current_terminal %in% c(66,132,133,135,138,141,142,143,148,182,183,385,386,387,405,406,407,415,416,417,418,420,422,423,424,430,431,440,441,442,443,444,445,446,451,456,543,544)))){
      
      
      if (current_terminal %in% c(207,234,526,131,53)){
        amazonWeeklyExpected = amazonData[10]
        amazonWeeklyExpected = amazonWeeklyExpected[1:(52-end_week+52),]
        colnames(amazonWeeklyExpected)[1] = "Amazon"  
      }else{
        amazonWeeklyExpected = amazonData[as.character(current_terminal)]
        amazonWeeklyExpected = amazonWeeklyExpected[1:(52-end_week+52),]
        colnames(amazonWeeklyExpected)[1] = "Amazon"    
      }
      
      
      
      if (current_terminal %in% c(207,234,526,131,53)){
        modl = stlf(y = values, method = "arima", h = 52-end_week+52, s.window = 52, robust = FALSE, biasadj = FALSE, xreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Actual",c("Cyber")])), newxreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Forecast",c("Cyber")])))
      }else{
        tryCatch({
          modl = stlf(y = values, method = "arima", h = 52-end_week+52, s.window = 52, robust = FALSE, biasadj = FALSE, xreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Actual",c("Cyber")], amazonWeeklyHistorical)), newxreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Forecast",c("Cyber")], amazonWeeklyExpected)))  
        },error=function(e){
          print(paste0(current_terminal," did not use B2C regressor because of thrown error. Forecasted without B2C regressor"))
          modl = stlf(y = values, method = "arima", h = 52-end_week+52, s.window = 52, robust = FALSE, biasadj = FALSE, xreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Actual",c("Cyber")], amazonWeeklyHistorical)), newxreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Forecast",c("Cyber")], amazonWeeklyExpected)))
        })
        
      }
      
      fcast = modl$mean
      
      forecastedMean = as.numeric(fcast)
      
    }
    else if ((sheet == 1 && !(current_terminal %in% c(66,132,133,135,138,141,142,143,148,182,183,385,386,387,405,406,407,415,416,417,418,420,422,423,424,430,431,440,441,442,443,444,445,446,451,456,543,544)))){
      
      if (current_terminal %in% c(207,234,526,131,53)){
        amazonWeeklyExpected = amazonDataStops[10]
        amazonWeeklyExpected = amazonWeeklyExpected[1:(52-end_week+52),]
        colnames(amazonWeeklyExpected)[1] = "Amazon"
      }else{
        amazonWeeklyExpected = amazonDataStops[as.character(current_terminal)]
        amazonWeeklyExpected = amazonWeeklyExpected[1:(52-end_week+52),]
        colnames(amazonWeeklyExpected)[1] = "Amazon"  
      }
      
      
      if (current_terminal %in% c(207,234,526,131,53)){
        modl = stlf(y = values, method = "arima", h = 52-end_week+52, s.window = 52, robust = FALSE, biasadj = FALSE, xreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Actual",c("Cyber")])), newxreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Forecast",c("Cyber")])))  
      }else{
        tryCatch({
          modl = stlf(y = values, method = "arima", h = 52-end_week+52, s.window = 52, robust = FALSE, biasadj = FALSE, xreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Actual",c("Cyber")], amazonWeeklyHistorical/1.07)), newxreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Forecast",c("Cyber")], amazonWeeklyExpected)))    
        },error=function(e){
          print(paste0(current_terminal," did not use B2C regressor because of thrown error. Forecasted without B2C regressor"))
          modl = stlf(y = values, method = "arima", h = 52-end_week+52, s.window = 52, robust = FALSE, biasadj = FALSE, xreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Actual",c("Cyber")], amazonWeeklyHistorical/1.07)), newxreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Forecast",c("Cyber")], amazonWeeklyExpected)))    
        })
        
      }
      
      fcast = modl$mean
      
      forecastedMean = as.numeric(fcast)
      
    }
    else if (sheet == 6){
      
      if (current_terminal %in% c(207,234,526,131,53)){
        amazonWeeklyExpected = amazonDataStops[10]
        amazonWeeklyExpected = amazonWeeklyExpected[1:(52-end_week+52),]
        colnames(amazonWeeklyExpected)[1] = "Amazon"
      }else{
        amazonWeeklyExpected = amazonDataStops[as.character(current_terminal)]
        amazonWeeklyExpected = amazonWeeklyExpected[1:(52-end_week+52),]
        colnames(amazonWeeklyExpected)[1] = "Amazon"  
      }
      
      
      if (current_terminal %in% c(207,234,526,131,53)){
        modl = stlf(y = values, method = "arima", h = 52-end_week+52, s.window = 52, robust = FALSE, biasadj = FALSE, xreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Actual",c("Cyber")])), newxreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Forecast",c("Cyber")])))  
      }else{
        tryCatch({
          modl = stlf(y = values, method = "arima", h = 52-end_week+52, s.window = 52, robust = FALSE, biasadj = FALSE, xreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Actual",c("Cyber")], amazonWeeklyHistorical/1.07)), newxreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Forecast",c("Cyber")], amazonWeeklyExpected)))    
        },error=function(e){
          print(paste0(current_terminal," did not use B2C regressor because of thrown error. Forecasted without B2C regressor"))
          modl = stlf(y = values, method = "arima", h = 52-end_week+52, s.window = 52, robust = FALSE, biasadj = FALSE, xreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Actual",c("Cyber")], amazonWeeklyHistorical/1.07)), newxreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Forecast",c("Cyber")], amazonWeeklyExpected)))    
        })
        
      }
      
      fcast = modl$mean
      
      forecastedMean = as.numeric(fcast)
    }
    else{
      tryCatch({
        modl = stlf(y = values, method = "arima", h = 52-end_week+52, s.window = 52, robust = FALSE, biasadj = FALSE, xreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Actual",c("Cyber")])), newxreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Forecast",c("Cyber")])))  
      },error=function(e){
        print(paste0(current_terminal," did not use B2C regressor because of thrown error. Forecasted without B2C regressor"))
        modl = stlf(y = values, method = "arima", h = 52-end_week+52, s.window = 52, robust = FALSE, biasadj = FALSE, xreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Actual",c("Cyber")])), newxreg = as.matrix(cbind(current_cyber[current_cyber$Type == "Forecast",c("Cyber")])))
      })
      
      
      fcast = modl$mean
      
      forecastedMean = as.numeric(fcast)
    }
    
    current_result = data.frame(Terminal = current_terminal,
                                Year = 2022,
                                Week = seq(end_week+1,52+52,1),
                                Forecast  = forecastedMean)
    #2022
    current_result[current_result$Week >= 53,2] = current_result[current_result$Week >= 53,2]+1
    current_result[current_result$Week >= 53,3] = current_result[current_result$Week >= 53,3] - 52
    #2023
    current_result[current_result$Week >= 53,2] = current_result[current_result$Week >= 53,2]+1
    current_result[current_result$Week >= 53,3] = current_result[current_result$Week >= 53,3] - 52
    # decomposition = data.frame(Year = opt_start_year,
    #                            Week = seq(opt_start_week,opt_start_week+length(values)-1,1),
    #                            Terminal = current_terminal, Trend = as.numeric(decompose(values)$trend), Seasonal = as.numeric(decompose(values)$seasonal), Random = as.numeric(decompose(values)$random), Values = as.numeric(values))
    # for (i in 1:4){
    #   decomposition[decomposition$Week > 52, "Year"] = decomposition[decomposition$Week > 52, "Year"] + 1
    #   decomposition[decomposition$Week > 52, "Week"] = decomposition[decomposition$Week > 52, "Week"] - 52
    # }
    
    if (terminal == 0){
      full_result = current_result
      #full_decomposition = decomposition
    }
    else{
      full_result = rbind(full_result,current_result)
      #full_decomposition = rbind(full_decomposition, decomposition)
    }
    
  }
  write.xlsx(full_result, paste("C:/Forecast tools/P&D Forecast/P&D Forecast v2/Excel Output/ForecastResults.xlsx",sep=""), sheetName = paste("Type",sheet), append = TRUE, row.names = FALSE)
  #write.xlsx(full_decomposition, paste("C:/Users/nikola.ivanovic/Documents/P&D Forecast v2/Excel Output/ForecastDecomposition.xlsx",sep=""), sheetName = paste("Type",sheet), append = TRUE, row.names = FALSE)
  if (sheet ==0){
    full_result_new = full_result
  }else{
    full_result_new = cbind(full_result_new,full_result[,"Forecast"])
  }
}
write.xlsx(full_result_new, paste("C:/Forecast tools/P&D Forecast/P&D Forecast v2/Excel Output/ForecastResults_onesheet.xlsx",sep=""), sheetName = "Forecast", append = TRUE, row.names = FALSE)

openxlsx::write.xlsx(weekly_data, paste("C:/Forecast tools/P&D Forecast/P&D Forecast v2/Excel Output/ForecastResults-Actual.xlsx",sep=""), sheetName = paste("Actual"), append = TRUE, row.names = FALSE)
#openxlsx::write.xlsx(weekly_data, paste("C:/Forecast tools/P&D Forecast/P&D Forecast v2/Excel Output/DailyForecast3.xlsx",sep=""), append = TRUE, row.names = FALSE)
