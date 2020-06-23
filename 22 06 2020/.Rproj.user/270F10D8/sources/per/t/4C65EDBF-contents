
se <- function(x) 
{
  sd(x, na.rm = TRUE)/sqrt(length(x))
}

median.my <- function (vec, na.rm.my = TRUE)
{
  if(is.factor(vec)) return (NA)
  return (median(vec, na.rm=TRUE))
}

quantile.my <- function(vec, probs = 0.25, na.rm = TRUE)
{
  if(is.factor(vec)) return (NA)
  return (quantile(vec, probs, na.rm = TRUE ))
  
}

quantile.interval.my <- function(vec, 
                                 prob.min    = 0.25, 
                                 prob.max    = 0.75, 
                                 na.rm       = TRUE, 
                                 pos         = 2
)
{
  require(plyr)
  require(magrittr)
  
  qu.min = vec %>% quantile(prob.min, na.rm = TRUE) %>% round(digits = pos)
  qu.max = vec %>% quantile(prob.max, na.rm = TRUE) %>% round(digits = pos)
  
  return (paste(c( '[', as.character(qu.min),'; ',as.character(qu.max), ']'), collapse=''))
}

min.my <- function(vec, na.rm = TRUE)
{
  if(is.factor(vec)) return (NA)
  return (min(vec, na.rm=TRUE))
  
}

max.my <- function(vec, na.rm = TRUE)
{
  if(is.factor(vec)) return (NA)
  return (max(vec, na.rm=TRUE))
  
}

bin.pos.my <- function(vec)
{
  return(sum(vec == 1, na.rm=TRUE))
}

bin.neg.my <- function(vec)
{
  return(length(vec)-bin.pos.my(vec))
  
}

bin.med.my <- function(vec)
{
  require("binom")
  res <- binom.confint(bin.pos.my(vec),length(vec),methods = "wilson")
  return((100.0*as.numeric(res$mean)) %>% round(digits = 2))
  
}

bin.ci.my <- function(vec, pos = 2, type.brackets = "()")
{
  res <- binom.confint(bin.pos.my(vec), length(vec),methods = "wilson")
  left   = (100.0*as.numeric(res$lower)) %>% round(digits = pos)
  right  = (100.0*as.numeric(res$upper)) %>% round(digits = pos)
  
  if (type.brackets == "()")
  {
    return (paste(c( '(', as.character(left),'; ',as.character(right), ')'), collapse=''))
  }
  
  if (type.brackets == "[]")
  {
    return (paste(c( '[', as.character(left),'; ',as.character(right), ']'), collapse=''))
  }
  if (type.brackets == "[%]")
  {
    return (paste(c( '[', as.character(left),'%; ',as.character(right), '%]'), collapse=''))
  }
}


bin.ci.my.num <- function(vec, b = "l")
{
  res <- binom.confint(bin.pos.my(vec), length(vec),methods = "wilson")
  left   = (as.numeric(res$lower)) %>% round(digits = 2)
  right  = (as.numeric(res$upper)) %>% round(digits = 2)
  
  if(b=="l"){
    return (left)
  }else{
    return (right)   
  }
  
}

bin.ci.my.l<- function(vec)
{
  return (bin.ci.my.num(vec, b="l"))
}
bin.ci.my.r<- function(vec)
{
  return (bin.ci.my.num(vec, b="r"))
}

bin.my = function(vec, digits=2)
{
  if ( all(is.na(vec)) == TRUE ) return("NA")
  res = paste0( round(bin.pos.my(vec),digits),", ",
                as.character(round(bin.med.my(vec),digits)), '% ', # %,
                bin.ci.my(vec, pos = digits, type.brackets = "[%]") 
  )  
  return(res)
}




# returns number of non NA's in vector:
# #non NA's(% non NA's)
na.count.string = function(x)
{
  return(
    paste0(as.character(sum(!is.na(x))),
           " (",
           as.character(round(sum(!is.na(x))/length(x)*100,0)),
           "%)"
    )
  )
}

# returns formatted value rounded to 3 digits:
# "0.001*"  if value < 0.001
# "0.xxx*"  if value < 0.05
# "0.xxx"   if 0.05 <= value < 1
# "1"       if value = 1
p_value_formatted = function(p_value)
{
  # number of digits for round
  nod = 3
  
  suffix_char = ""
  if (p_value < 0.001) {
    p_val = "<0.001"
  } else {
    p_val = round(p_value, nod)
  }
  
  if (p_value < 0.05){
    p_val = paste0(p_val, "*")
  }
  
  if (p_value == 1.0){
    p_val = ">0.999"
  }
  
  return (p_val)
}

#' Title
#' parse data.frame into two column and remove NA's  
#' first column contain group labels
#' second column contain values for comparing  
#'
#' @param dt            data frame for parsing
#' @param groups.column column contained group labels
#' @param values.column values column
#'
#' @return data frame
#' @export dpyr
#'
#' @examples  parse.data.frame(mtcars, 2, c(4,6), 3)
#' 
parse.data.frame = function(
  dt = NULL,
  groups.column = NULL,
  group.labels  = NULL,
  values.column = NULL
)
{
  require(dplyr)
  dt %>% 
    select(c(groups.column,values.column)) %>% 
    filter(dt[[groups.column]] %in% group.labels) %>% 
    na.omit() %>%
    return ()
}

parse.data.frame.with.na = function(
  dt = NULL,
  groups.column = NULL,
  group.labels  = NULL,
  values.column = NULL
)
{
  require(dplyr)
  dt %>% 
    select(c(groups.column,values.column)) %>% 
    filter(dt[[groups.column]] %in% group.labels) %>% 
    #na.omit() %>%
    return ()
}







calc.NA = function(dat = NULL, path.file = NULL)
{
  require(dplyr)
  res = summarise_if(dat,is.numeric,funs(sum((is.na(.)))))
  res = t(res)
  res = cbind(res,res)
  res[,2] = paste0(as.character(round((res[,2]/length(dat[[1]]))*100,0)), "%")
  res[,3] = round((res[,2]/length(dat[[1]]))*100,0)
  colnames(res) = c("?????????? ?????????","% ?????????","% ?????????")
  res %>% write.csv2(file=path.file)
  res = as.data.frame(res)
  res[,1] = as.integer(res[,1])
  res[,3] = as.integer(res[,3])
  return(res)
}


#########################3
summ.bin.date = function(
  dt       = NULL, # dt.diam
  offset   = 0,    # 2
  range    = 0,    # 1:19
  date.col = 0,    # 19
  stages   = 0,     # 5 length(offsets)
  format.date = "%d.%m.%Y"
)
{
  # remove id of date column
  k.range = range[!range %in% date.col] 
  
  k.range = k.range + offset
  
  for ( k in k.range )
  {
    #add summary columns
    bin.name             = paste0("total.bin ",  colnames(dt)[k])
    date.name            = paste0("first.date ", colnames(dt)[k])
    dt[ ,bin.name ]      = NA
    dt[ ,date.name]      = as.Date(dt[ ,offset+date.col],format.date)
    
    len.rows             = dim(dt)[1]
    
    for(id.row in 1:len.rows)  
    {
      #to know there was an event or not
      event = 0
      
      for(j in (1:stages - 1))  
      {
        offset.stage = j*length(range)
        id.col = k + offset.stage
        
        # current stage date
        cur.date = as.Date(dt[id.row, offset+date.col+offset.stage],format.date)
        
        if(event == 0)
        {
          
          if( 
            ( !is.na(dt[id.row,id.col]) ) 
            & 
            ( dt[id.row,id.col] == 1    )
          )
          {
            dt[id.row, bin.name]       = 1
            dt.diam[id.row, date.name] = cur.date
            event                      = 1 
          } else
          {
            if( 
              ( !is.na(cur.date)          )
              &
              ( !is.na(dt[id.row,id.col]) )
              &
              ( dt[id.row,id.col] == 0    )
            )
            {
              dt[id.row, bin.name]  = 0
              dt[id.row, date.name] = cur.date
            }
          }
          
        }
      }
    }
  }
  return (dt)
}



