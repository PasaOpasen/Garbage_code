
library(tidyr)

facilities = 0:5
all_combs = crossing(var1 = facilities, var2 = facilities, var3 = facilities, var4 = facilities, var5 = facilities)

is5 = apply(all_combs, 1, function(row) sum(row) == 5)

all_combs = all_combs[is5, ]



probs = c(0.8, 0.7, 0.6, 0.5, 0.3)


get_best_combination = function(count){
  
  sim_results = sapply(probs, function(p) rbinom(count,1,p))

  
  answers = numeric(nrow(all_combs))
  
  for(i in 1:nrow(all_combs)){
    
    comb = all_combs[i,]
    
    nums = (sim_results %*% t(comb)) > 2 
    
    answers[i] = mean(nums)
  
  }
  
  result = as.numeric(all_combs[which.max(answers),])
  
  cat('best accuracy:',max(answers),'on',result,'combination\n')
  
  return(result)
}













